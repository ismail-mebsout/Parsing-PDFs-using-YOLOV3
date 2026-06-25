"""Command-line interface for the table extractor.

Usage::

    python -m pdf_table_extractor.cli --pdf-path doc.pdf --page 2 --out-dir results
"""

from __future__ import annotations

import argparse
import sys

from .pipeline import extract_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-table-extractor",
        description="Detect and extract tables from a PDF page into Excel files.",
    )
    parser.add_argument(
        "--pdf-path",
        required=True,
        help="Path to the PDF file to parse.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="1-indexed page to parse (default: 1).",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory where the .xlsx tables are written (default: current).",
    )
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = extract_tables(args.pdf_path, page=args.page)
    if result.num_tables == 0:
        print(f"No tables detected on page {args.page} of {args.pdf_path}.")
        return 0
    written = result.save_excel(args.out_dir)
    print(f"Extracted {result.num_tables} table(s):")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
