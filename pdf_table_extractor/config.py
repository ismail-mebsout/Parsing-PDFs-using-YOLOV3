"""Configuration for the table-detection model and pipeline.

Replaces the old ad-hoc ``parameters`` class with a typed, immutable dataclass
whose defaults point at the bundled model assets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

# Repository layout: <root>/pdf_table_extractor/config.py -> <root>/assets
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
ASSETS_DIR = ROOT_DIR / "assets"

DEFAULT_CFG = ASSETS_DIR / "yolov3-tiny_table.cfg"
DEFAULT_NAMES = ASSETS_DIR / "table.names"
DEFAULT_WEIGHTS = ASSETS_DIR / "best_v2.weights"


@dataclass(frozen=True)
class YoloConfig:
    """Inference configuration consumed by the vendored YOLOv3 runner.

    The attribute names mirror what :func:`pdf_table_extractor.yolo.detect_table`
    expects, so an instance can be passed straight through to the engine.
    """

    cfg: str = str(DEFAULT_CFG)
    names: str = str(DEFAULT_NAMES)
    weights: str = str(DEFAULT_WEIGHTS)
    img_size: int = 416
    conf_thres: float = 0.2
    iou_thres: float = 0.6
    device: str = "cpu"
    half: bool = False
    classes: list | None = None
    agnostic_nms: bool = False

    # Per-call fields (set by the detector, not by the user).
    source: str = ""
    output: str = "outputs"

    # Bounding-box expansion factor applied when mapping detections back to the
    # PDF coordinate space (guarantees the full table is enclosed).
    bbox_correction: float = 0.05

    def for_source(self, source: str, output: str) -> YoloConfig:
        """Return a copy bound to a specific input image and output folder."""
        return replace(self, source=str(source), output=str(output))


# A ready-to-use default instance.
DEFAULT_CONFIG: YoloConfig = YoloConfig()
