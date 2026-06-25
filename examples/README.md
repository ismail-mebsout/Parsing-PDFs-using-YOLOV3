# Example: end-to-end table extraction

A self-contained demo of the pipeline on a generated PDF.

## Files

| File | Description |
|------|-------------|
| `make_sample_pdf.py` | Generates `sample_invoice.pdf` (2 tables) with reportlab. |
| `sample_invoice.pdf` | Input PDF: an invoice line-items table + a quarterly summary table. |
| `run_pipeline.py` | Runs `extract_tables()` and writes all artifacts to `output/`. |
| `output/` | Committed results (see below). |

## Reproduce

```bash
pip install -r requirements.txt reportlab   # reportlab only needed to regenerate the PDF
python examples/make_sample_pdf.py          # (optional) regenerate the input PDF
python examples/run_pipeline.py             # detect + extract + save artifacts
```

## Results

The YOLOv3 detector found **2 tables** and Camelot extracted both correctly.

![detected boxes](output/detected_boxes.png)

`output/` contains:

- `detected_boxes.png` — the page with the detected bounding boxes drawn on top.
- `sample_invoice-1-table-0.{xlsx,csv}` — invoice line items (8×4).
- `sample_invoice-1-table-1.{xlsx,csv}` — quarterly summary (5×4).
- `results.json` — machine-readable summary (page, Camelot areas, shapes, previews).

Extracted table 0 (`sample_invoice-1-table-0.csv`):

```
Item,Qty,Unit Price,Total
Widget A,10,12.50,125.00
Widget B,4,30.00,120.00
Gadget X,2,75.25,150.50
Service Fee,1,40.00,40.00
Subtotal,,,435.50
Tax (8%),,,34.84
Grand Total,,,470.34
```
