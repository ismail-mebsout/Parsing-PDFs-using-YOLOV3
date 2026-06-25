"""Generate a sample PDF containing tables, used to demo the pipeline.

Run:  python examples/make_sample_pdf.py
Produces: examples/sample_invoice.pdf
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent / "sample_invoice.pdf"


def _styled_table(data):
    table = Table(data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#ECF0F1")],
                ),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build(path: Path = OUT) -> Path:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER)
    story = []

    story.append(Paragraph("ACME Corporation — Invoice #2026-0412", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "The following items were purchased during the billing period. "
            "Totals are shown in USD.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 18))

    line_items = [
        ["Item", "Qty", "Unit Price", "Total"],
        ["Widget A", "10", "12.50", "125.00"],
        ["Widget B", "4", "30.00", "120.00"],
        ["Gadget X", "2", "75.25", "150.50"],
        ["Service Fee", "1", "40.00", "40.00"],
        ["Subtotal", "", "", "435.50"],
        ["Tax (8%)", "", "", "34.84"],
        ["Grand Total", "", "", "470.34"],
    ]
    story.append(_styled_table(line_items))
    story.append(Spacer(1, 30))

    story.append(Paragraph("Quarterly Summary", styles["Heading2"]))
    story.append(Spacer(1, 10))
    summary = [
        ["Quarter", "Revenue", "Expenses", "Net"],
        ["Q1", "120,000", "80,000", "40,000"],
        ["Q2", "150,000", "95,000", "55,000"],
        ["Q3", "138,000", "90,500", "47,500"],
        ["Q4", "172,000", "101,000", "71,000"],
    ]
    story.append(_styled_table(summary))

    doc.build(story)
    return path


if __name__ == "__main__":
    out = build()
    print(f"Wrote {out}")
