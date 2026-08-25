"""
utilities/device.py — PyTorch device detection.

Provides a single helper that returns the best available ``torch.device``
string in priority order: CUDA → MPS → CPU.

Usage::

    from audio_processing.utilities.device import detect_device

    device = detect_device()   # e.g. "mps" on Apple Silicon
"""

from __future__ import annotations

import torch


def detect_device() -> str:
    """
    Return the best available PyTorch device string.

    Priority order:

    1. ``"cuda"``  — NVIDIA GPU via CUDA (checked with :func:`torch.cuda.is_available`).
    2. ``"mps"``   — Apple Silicon GPU via Metal (checked with
       :attr:`torch.backends.mps.is_available`).
    3. ``"cpu"``   — Fallback; prints a warning.

    :returns: One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
    :rtype: str
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    print("No GPU found — falling back to CPU.")
    return "cpu"

