"""Thin, testable wrapper around the vendored YOLOv3 inference runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .config import DEFAULT_CONFIG, YoloConfig
from .geometry import Detection, parse_yolo_output


class TableDetector:
    """Detect table bounding boxes in a page image.

    The heavy ``torch`` import is deferred until :meth:`detect` is called so that
    importing this module (and the rest of the package) stays cheap and does not
    require torch to be installed — useful for tests and for the API process that
    may run detection out-of-band.
    """

    def __init__(self, config: YoloConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def _check_assets(self) -> None:
        for label, path in (
            ("cfg", self.config.cfg),
            ("names", self.config.names),
            ("weights", self.config.weights),
        ):
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing model {label} file: {path}")

    def detect(self, image_path: str) -> list[Detection]:
        """Run inference on ``image_path`` and return parsed detections.

        Each detection is ``[x1, y1, x2, y2, cls, conf]`` in image-pixel space.
        """
        self._check_assets()
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Import lazily: this is the only place torch is actually needed.
        from .yolo import detect_table

        with tempfile.TemporaryDirectory(prefix="yolo_out_") as out_dir:
            opt = self.config.for_source(image_path, out_dir)
            raw = detect_table(opt)
        return parse_yolo_output(raw)
