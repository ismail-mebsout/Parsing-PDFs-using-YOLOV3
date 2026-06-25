"""Shared pytest fixtures.

The suite is designed to run without heavy/optional dependencies (torch,
camelot, pdf2image): those boundaries are mocked. Pure geometry is tested for
real.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_detection():
    """A single image-space detection: x1, y1, x2, y2, cls, conf."""
    return [100.0, 200.0, 300.0, 500.0, 0.0, 0.95]


@pytest.fixture
def raw_yolo_output():
    """Raw text block as returned by the YOLO runner (two detections)."""
    return "100 200 300 500 0 0.95 \n10 20 30 40 0 0.80 \n"


@pytest.fixture
def fake_tables():
    """Two small DataFrames standing in for Camelot output."""
    return [
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        pd.DataFrame({"x": ["p", "q"]}),
    ]
