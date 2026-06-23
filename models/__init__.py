"""
models package — neural network architectures and checkpoint utilities.

Public API
----------
Autoencoder          : fully-connected autoencoder baseline (DCASE baseline)
load_from_checkpoint : restore model weights from a PtTrainer checkpoint file
"""

from models.autoencoder import Autoencoder
from models.checkpoint import load_from_checkpoint

__all__ = ["Autoencoder", "load_from_checkpoint"]

