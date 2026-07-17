"""
utilities — shared helper utilities for the audio_processing project.

Sub-modules
-----------
device  : PyTorch device detection (:func:`~utilities.device.detect_device`).
paths   : Canonical data-directory constants and path resolution helpers
          (:data:`~utilities.paths.DEFAULT_DATA_DIR`, :func:`~utilities.paths.entity_dir`,
          :func:`~utilities.paths.effective_data_dir`).
data    : Data-availability checking and download orchestration
          (:func:`~utilities.data.ensure_data`).
"""

from utilities.data import ensure_data
from utilities.device import detect_device
from utilities.paths import DEFAULT_DATA_DIR, DEFAULT_MEL_CACHE_DIR, effective_data_dir, entity_dir

__all__ = [
    "detect_device",
    "DEFAULT_DATA_DIR",
    "DEFAULT_MEL_CACHE_DIR",
    "effective_data_dir",
    "entity_dir",
    "ensure_data",
]
