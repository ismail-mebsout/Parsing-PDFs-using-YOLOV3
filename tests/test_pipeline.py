"""Tests for the orchestration pipeline with detection/Camelot mocked out."""

from pathlib import Path

import pandas as pd
import pytest

from pdf_table_extractor import pipeline
from pdf_table_extractor.pipeline import ExtractionResult, extract_tables


def test_extraction_result_num_tables(fake_tables):
    result = ExtractionResult(pdf_path="doc.pdf", page=2, tables=fake_tables)
    assert result.num_tables == 2


def test_extraction_result_to_records(fake_tables):
    result = ExtractionResult(pdf_path="doc.pdf", page=2, tables=fake_tables)
    records = result.to_records()
    assert records[0] == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
    assert records[1] == [{"x": "p"}, {"x": "q"}]


def test_extraction_result_save_excel(tmp_path, fake_tables):
    result = ExtractionResult(pdf_path="/some/doc.pdf", page=3, tables=fake_tables)
    written = result.save_excel(tmp_path)
    assert [Path(p).name for p in written] == [
        "doc-3-table-0.xlsx",
        "doc-3-table-1.xlsx",
    ]
    for path in written:
        assert Path(path).exists()
        # Round-trip the first table to prove the file is a valid workbook.
    reloaded = pd.read_excel(written[0], index_col=0)
    assert list(reloaded.columns) == ["a", "b"]


def test_extract_tables_missing_pdf():
    with pytest.raises(FileNotFoundError):
        extract_tables("does-not-exist.pdf", page=1)


class _FakeDetector:
    def __init__(self, detections):
        self._detections = detections
        self.seen_image = None

    def detect(self, image_path):
        self.seen_image = image_path
        return self._detections


def test_extract_tables_happy_path(tmp_path, monkeypatch, fake_tables):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    detector = _FakeDetector([[100.0, 200.0, 300.0, 500.0, 0.0, 0.95]])

    monkeypatch.setattr(pipeline, "get_pdf_page_size", lambda p, pg: (600.0, 792.0))
    monkeypatch.setattr(
        pipeline,
        "render_page_to_image",
        lambda p, pg, dest: (dest, 1000, 800),
    )
    captured = {}

    def fake_camelot(pdf_path, page, areas):
        captured["areas"] = areas
        return fake_tables

    monkeypatch.setattr(pipeline, "_read_camelot", fake_camelot)

    result = extract_tables(str(pdf), page=2, detector=detector)

    assert result.num_tables == 2
    assert result.page == 2
    # One detection -> one Camelot area string with four comma-separated coords.
    assert len(result.areas) == 1
    assert captured["areas"] == result.areas
    coords = [float(c) for c in result.areas[0].split(",")]
    assert coords == pytest.approx([67.5, 639.54, 232.5, 372.24])


def test_extract_tables_no_detections(tmp_path, monkeypatch):
    pdf = tmp_path / "blank.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(pipeline, "get_pdf_page_size", lambda p, pg: (600.0, 792.0))
    monkeypatch.setattr(
        pipeline, "render_page_to_image", lambda p, pg, dest: (dest, 1000, 800)
    )
    # Camelot must never be called when there are no detections.
    monkeypatch.setattr(
        pipeline,
        "_read_camelot",
        lambda *a, **k: pytest.fail("camelot should not run without detections"),
    )

    result = extract_tables(str(pdf), page=1, detector=_FakeDetector([]))
    assert result.num_tables == 0
    assert result.areas == []
