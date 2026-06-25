"""End-to-end table extraction pipeline.

Render a PDF page to an image, detect table regions with YOLOv3, map the boxes
into PDF coordinate space and hand them to Camelot for cell-level extraction.

Heavy/optional dependencies (``camelot``, ``pdf2image``, ``PyPDF2``) are imported
lazily inside the functions that need them, so this module can be imported — and
its pure logic exercised under test — without those packages installed.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_CONFIG, YoloConfig
from .detector import TableDetector
from .geometry import bbox_to_pdf, to_camelot_area


@dataclass
class ExtractionResult:
    """Result of running the pipeline on a single PDF page."""

    pdf_path: str
    page: int
    areas: list[str] = field(default_factory=list)  # Camelot table_areas strings
    tables: list[object] = field(default_factory=list)  # list[pandas.DataFrame]

    @property
    def num_tables(self) -> int:
        return len(self.tables)

    def to_records(self) -> list[list[dict]]:
        """Serialise every table to a list of row dicts (JSON friendly)."""
        return [table.to_dict(orient="records") for table in self.tables]

    def save_excel(self, out_dir: str = ".") -> list[str]:
        """Write each table to ``<stem>-<page>-table-<i>.xlsx``; return paths."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        stem = Path(self.pdf_path).stem
        written: list[str] = []
        for i, table in enumerate(self.tables):
            dest = out_path / f"{stem}-{self.page}-table-{i}.xlsx"
            table.to_excel(dest)
            written.append(str(dest))
        return written


def get_pdf_page_size(pdf_path: str, page: int) -> tuple:
    """Return the ``(width, height)`` of a 1-indexed PDF page in points."""
    from PyPDF2 import PdfReader

    reader = PdfReader(pdf_path)
    media_box = reader.pages[page - 1].mediabox
    return float(media_box.width), float(media_box.height)


def render_page_to_image(pdf_path: str, page: int, dest_path: str) -> tuple:
    """Render a PDF page to a JPEG and return ``(path, height, width)``."""
    import numpy as np
    from pdf2image import convert_from_path

    image = convert_from_path(pdf_path, first_page=page, last_page=page)[0]
    image.save(dest_path)
    height, width = np.array(image).shape[:2]
    return dest_path, int(height), int(width)


def extract_tables(
    pdf_path: str,
    page: int,
    detector: TableDetector | None = None,
    config: YoloConfig = DEFAULT_CONFIG,
) -> ExtractionResult:
    """Detect and extract every table on ``page`` of ``pdf_path``."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    detector = detector or TableDetector(config)
    pdf_width, pdf_height = get_pdf_page_size(pdf_path, page)

    with tempfile.TemporaryDirectory(prefix="pdf_page_") as tmp:
        image_path = str(Path(tmp) / f"{Path(pdf_path).stem}-{page}.jpg")
        image_path, img_h, img_w = render_page_to_image(pdf_path, page, image_path)
        detections = detector.detect(image_path)

    areas = [
        to_camelot_area(
            bbox_to_pdf(
                pdf_width,
                pdf_height,
                img_h,
                img_w,
                detection,
                correction=config.bbox_correction,
            )
        )
        for detection in detections
    ]

    tables = _read_camelot(pdf_path, page, areas) if areas else []
    return ExtractionResult(pdf_path=pdf_path, page=page, areas=areas, tables=tables)


def _read_camelot(pdf_path: str, page: int, areas: list[str]) -> list:
    """Run Camelot over the detected regions and return a list of DataFrames."""
    from camelot import io as camelot

    parsed = camelot.read_pdf(
        filepath=pdf_path,
        pages=str(page),
        flavor="stream",
        table_areas=areas,
    )
    return [table.df for table in parsed]
