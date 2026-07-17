"""
training package — training loop, configuration, and experiment tracking.

Public API
----------
TrainingConfig      : typed hyperparameter container
PtTrainer           : lightweight PyTorch training loop
ExperimentTracker   : structural protocol for tracking back-ends
NullTracker         : silent no-op tracker (default)
WandbTracker        : Weights & Biases tracker (requires ``wandb``)
"""

from training.config import TrainingConfig
from training.tracking import ExperimentTracker, NullTracker, WandbTracker
from training.trainer import PtTrainer

__all__ = [
    "TrainingConfig",
    "PtTrainer",
    "ExperimentTracker",
    "NullTracker",
    "WandbTracker",
]

