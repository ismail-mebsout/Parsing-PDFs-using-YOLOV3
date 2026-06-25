"""Vendored YOLOv3 engine (adapted from ``ultralytics/yolov3``).

This subpackage contains third-party model code and is intentionally excluded
from the project's formatting/linting configuration. Treat it as a black box:
the only supported entry point is :func:`detect_table`.
"""

from .detect import detect_table

__all__ = ["detect_table"]
