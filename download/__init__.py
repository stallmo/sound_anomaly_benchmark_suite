"""
download — dataset acquisition and preprocessing.

Public API
----------
DatasetDownloader  : Protocol for fetching raw source files
DatasetPreprocessor: Protocol for converting raw files into the canonical layout
DatasetInfo        : Metadata + acquisition components tied to one dataset
DownloadError      : Raised on network / I/O failures during download
PreprocessingError : Raised on format / I/O failures during preprocessing

DATASET_REGISTRY   : dict[str, DatasetInfo] — all registered datasets
get_dataset_info   : look up a DatasetInfo by name, raises KeyError if unknown
"""

from download.protocol import (
    DatasetDownloader,
    DatasetInfo,
    DatasetPreprocessor,
    DownloadError,
    PreprocessingError,
)
from download.registry import DATASET_REGISTRY, get_dataset_info

__all__ = [
    "DatasetDownloader",
    "DatasetPreprocessor",
    "DatasetInfo",
    "DownloadError",
    "PreprocessingError",
    "DATASET_REGISTRY",
    "get_dataset_info",
]

