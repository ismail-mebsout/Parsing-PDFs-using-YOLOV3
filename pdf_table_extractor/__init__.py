"""pdf_table_extractor — detect and extract tables from PDFs with YOLOv3.

Public API:
    >>> from pdf_table_extractor import extract_tables
    >>> result = extract_tables("doc.pdf", page=2)
    >>> result.num_tables
    1

Only lightweight modules are imported here; torch/camelot are pulled in lazily
when detection/extraction actually runs.
"""

from .config import DEFAULT_CONFIG, YoloConfig
from .detector import TableDetector
from .pipeline import ExtractionResult, extract_tables

__version__ = "1.0.0"

__all__ = [
    "YoloConfig",
    "DEFAULT_CONFIG",
    "TableDetector",
    "ExtractionResult",
    "extract_tables",
    "__version__",
]
