"""Pure geometry helpers for mapping YOLO detections into PDF coordinates.

These functions contain no I/O, no torch and no Camelot dependency, which makes
them fully unit-testable in isolation. They reproduce the coordinate math of the
original script while being explicit about their inputs.

Detection rows are ``[x1, y1, x2, y2, cls, conf]`` in *image pixel* space, with
the origin at the top-left. PDF space has its origin at the bottom-left, hence
the ``1 - y`` inversion in :func:`bbox_to_pdf`.
"""

from __future__ import annotations

from collections.abc import Sequence

# A parsed detection: x1, y1, x2, y2, class, confidence.
Detection = list[float]


def parse_yolo_output(raw: str) -> list[Detection]:
    """Parse the whitespace-separated detection block into numeric rows.

    Each non-empty line is expected to hold six values
    (``x1 y1 x2 y2 cls conf``). Extra surrounding whitespace is tolerated.
    """
    detections: list[Detection] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        detections.append([float(value) for value in line.split()])
    return detections


def box_metrics(img_height: float, img_width: float, bbox: Sequence[float]) -> dict:
    """Return absolute box coordinates, size and the image dimensions."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return {
        "box": (x1, y1, x2, y2),
        "size": (x2 - x1, y2 - y1),
        "image": (float(img_height), float(img_width)),
    }


def normalize_bbox(
    img_height: float,
    img_width: float,
    bbox: Sequence[float],
    correction: float = 0.05,
) -> list[float]:
    """Normalise a pixel-space box to ``[0, 1]`` and expand it slightly.

    The expansion (``correction``) widens the box so Camelot reliably captures
    the whole table. This mirrors the original asymmetric correction: the box is
    grown by ``correction`` on the left/right and bottom, and by half that on the
    top.
    """
    metrics = box_metrics(img_height, img_width, bbox)
    x1, y1, x2, y2 = metrics["box"]
    w_table, h_table = metrics["size"]
    height, width = metrics["image"]

    x1_norm, y1_norm = x1 / width, y1 / height
    x2_norm, y2_norm = x2 / width, y2 / height
    w_norm, h_norm = w_table / width, h_table / height

    w_corr = w_norm * correction
    h_corr = h_norm * correction

    return [
        x1_norm - w_corr,
        y1_norm - h_corr / 2,
        x2_norm + w_corr,
        y2_norm + 2 * h_corr,
    ]


def bbox_to_pdf(
    pdf_width: float,
    pdf_height: float,
    img_height: float,
    img_width: float,
    bbox: Sequence[float],
    correction: float = 0.05,
) -> list[float]:
    """Map an image-space detection to a PDF-space ``[x1, y1, x2, y2]`` box.

    The returned coordinates use the PDF convention where ``(x1, y1)`` is the
    top-left and ``(x2, y2)`` the bottom-right, which is what Camelot's
    ``table_areas`` expects.
    """
    x1_norm, y1_norm, x2_norm, y2_norm = normalize_bbox(
        img_height, img_width, bbox, correction
    )
    x1 = x1_norm * pdf_width
    y1 = (1 - y1_norm) * pdf_height
    x2 = x2_norm * pdf_width
    y2 = (1 - y2_norm) * pdf_height
    return [x1, y1, x2, y2]


def to_camelot_area(pdf_bbox: Sequence[float]) -> str:
    """Format a PDF-space box as a Camelot ``table_areas`` string."""
    return ",".join(str(coord) for coord in pdf_bbox)
