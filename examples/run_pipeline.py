"""Run the full extraction pipeline on the sample PDF and save the results.

Run:  python examples/run_pipeline.py
Outputs (under examples/output/):
  * <stem>-<page>-table-<i>.xlsx / .csv  — one per detected table
  * results.json                         — machine-readable summary
  * detected_boxes.png                   — page image with detection overlay
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

# Allow running directly (`python examples/run_pipeline.py`) without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from pdf_table_extractor import extract_tables  # noqa: E402
from pdf_table_extractor.detector import TableDetector  # noqa: E402
from pdf_table_extractor.pipeline import render_page_to_image  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PDF = HERE / "sample_invoice.pdf"
PAGE = 1
OUT_DIR = HERE / "output"


def save_overlay(pdf_path: Path, page: int, dest: Path) -> int:
    """Render the page, run detection and draw the boxes. Returns box count."""
    image_path = str(OUT_DIR / "_page.jpg")
    render_page_to_image(str(pdf_path), page, image_path)
    detections = TableDetector().detect(image_path)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det[:4]
        draw.rectangle([x1, y1, x2, y2], outline=(231, 76, 60), width=4)
        draw.text((x1 + 4, y1 + 4), f"table {i} ({det[5]:.2f})", fill=(231, 76, 60))
    img.save(dest)
    Path(image_path).unlink(missing_ok=True)
    return len(detections)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    result = extract_tables(str(PDF), page=PAGE)
    print(f"Detected {result.num_tables} table(s) on page {PAGE} of {PDF.name}")

    xlsx_paths = result.save_excel(OUT_DIR)
    for i, df in enumerate(result.tables):
        csv_path = OUT_DIR / f"{PDF.stem}-{PAGE}-table-{i}.csv"
        df.to_csv(csv_path, index=False, header=False)
        print(
            f"  table {i}: shape={df.shape} -> {Path(xlsx_paths[i]).name}, {csv_path.name}"
        )

    num_boxes = save_overlay(PDF, PAGE, OUT_DIR / "detected_boxes.png")

    summary = {
        "pdf": PDF.name,
        "page": PAGE,
        "num_tables": result.num_tables,
        "camelot_areas": result.areas,
        "tables": [
            {"index": i, "shape": list(df.shape), "preview": df.head(2).values.tolist()}
            for i, df in enumerate(result.tables)
        ],
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"  overlay: detected_boxes.png ({num_boxes} boxes)")
    print("  summary: results.json")
    print(f"\nAll artifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
