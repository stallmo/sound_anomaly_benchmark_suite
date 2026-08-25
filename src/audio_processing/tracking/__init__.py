"""
tracking package — experiment tracking protocol and back-end implementations.

Designed as a cross-cutting module so that training, inference, and
evaluation can all share the same tracker interface without any of them
depending on each other.

Public API
----------
ExperimentTracker : structural protocol — implement to add a new back-end
NullTracker       : silent no-op (default when no tracker is supplied)
WandbTracker      : Weights & Biases back-end (requires ``wandb``)
"""

from audio_processing.tracking.base import ExperimentTracker, NullTracker
from audio_processing.tracking.wandb import WandbTracker

__all__ = [
    "ExperimentTracker",
    "NullTracker",
    "WandbTracker",
]

