"""Unit tests for the pure coordinate-mapping logic."""

import pytest

from pdf_table_extractor.geometry import (
    bbox_to_pdf,
    box_metrics,
    normalize_bbox,
    parse_yolo_output,
    to_camelot_area,
)


def test_parse_yolo_output_two_rows(raw_yolo_output):
    rows = parse_yolo_output(raw_yolo_output)
    assert rows == [
        [100.0, 200.0, 300.0, 500.0, 0.0, 0.95],
        [10.0, 20.0, 30.0, 40.0, 0.0, 0.80],
    ]


def test_parse_yolo_output_empty():
    assert parse_yolo_output("") == []
    assert parse_yolo_output("\n  \n") == []


def test_parse_yolo_output_tolerates_extra_whitespace():
    assert parse_yolo_output("  1   2  3 4 0 0.5  ") == [[1, 2, 3, 4, 0, 0.5]]


def test_box_metrics(sample_detection):
    m = box_metrics(1000, 800, sample_detection)
    assert m["box"] == (100.0, 200.0, 300.0, 500.0)
    assert m["size"] == (200.0, 300.0)
    assert m["image"] == (1000.0, 800.0)


def test_normalize_bbox_known_values(sample_detection):
    result = normalize_bbox(1000, 800, sample_detection, correction=0.05)
    assert result == pytest.approx([0.1125, 0.1925, 0.3875, 0.53])


def test_normalize_bbox_no_correction(sample_detection):
    result = normalize_bbox(1000, 800, sample_detection, correction=0.0)
    # With no correction this is just plain min/max normalisation.
    assert result == pytest.approx([0.125, 0.2, 0.375, 0.5])


def test_bbox_to_pdf_inverts_y(sample_detection):
    result = bbox_to_pdf(600, 792, 1000, 800, sample_detection, correction=0.05)
    assert result == pytest.approx([67.5, 639.54, 232.5, 372.24])
    # PDF origin is bottom-left: the top edge (y1) sits above the bottom edge.
    assert result[1] > result[3]


def test_to_camelot_area_roundtrip():
    area = to_camelot_area([67.5, 639.54, 232.5, 372.24])
    assert area == "67.5,639.54,232.5,372.24"
    assert len(area.split(",")) == 4
