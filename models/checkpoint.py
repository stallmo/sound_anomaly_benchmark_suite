"""
models/checkpoint.py — utilities for persisting and restoring model weights.

The checkpoint format is a plain ``torch.save`` dict with the keys written
by :meth:`~training.trainer.PtTrainer.save_checkpoint`::

    {
        "epoch":               int,
        "model_state_dict":    dict,
        "optimizer_state_dict": dict,
        "best_val_loss":       float,
    }

Only ``model_state_dict`` is required for inference; the remaining keys
are used when resuming training.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def load_from_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: str = "cpu",
) -> nn.Module:
    """
    Load model weights from a checkpoint file produced by
    :meth:`~training.trainer.PtTrainer.save_checkpoint` and return the model.

    The model is moved to *device* before the state dict is loaded, so the
    returned model is already on the target device and ready for inference.

    :param path: Path to the ``.pt`` checkpoint file.
    :type path: str or Path
    :param model: An *un-initialised* or *pre-constructed* model instance
        whose architecture matches the saved weights.  Weights are loaded
        in-place.
    :type model: nn.Module
    :param device: PyTorch device string to move the model to before loading
        (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).
    :type device: str
    :returns: The same *model* instance with weights restored and moved to
        *device*, set to ``eval()`` mode.
    :rtype: nn.Module
    :raises FileNotFoundError: If *path* does not exist.
    :raises RuntimeError: If the state dict is incompatible with *model*.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model.to(device)
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

