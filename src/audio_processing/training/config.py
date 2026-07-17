"""
training/config.py — hyperparameter container for :class:`~training.trainer.PtTrainer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from audio_processing.features.hpss import SignalTransform


@dataclass
class TrainingConfig:
    """
    All hyperparameters that govern a single training run.

    Keeping every knob in one typed dataclass gives a single source of truth
    that can be serialised, logged to an experiment tracker, and diffed
    between runs.

    :param epochs: Maximum number of training epochs.
    :type epochs: int
    :param learning_rate: Informational — used to construct the optimiser
        *before* passing it to :class:`~training.trainer.PtTrainer`.
        Logged automatically at the start of :meth:`~training.trainer.PtTrainer.train`.
    :type learning_rate: float
    :param weight_decay: Informational — same convention as ``learning_rate``.
    :type weight_decay: float
    :param batch_size: Informational — the actual batch size is determined by
        the :class:`~torch.utils.data.DataLoader` passed to the trainer.
        Stored here for logging completeness.
    :type batch_size: int
    :param device: PyTorch device string (e.g. ``"cpu"``, ``"cuda"``,
        ``"mps"``). The trainer moves the model to this device on
        initialisation.
    :type device: str
    :param checkpoint_dir: Directory where checkpoints are written.
        Created automatically if it does not exist.
    :type checkpoint_dir: Path
    :param early_stopping_patience: Stop training if the monitored metric
        (validation loss, or training loss when no validation loader is
        provided) does not improve for this many consecutive epochs.
        ``None`` disables early stopping.
    :type early_stopping_patience: int or None
    :param sample_rate: Audio sample rate in Hz.  Files are resampled to
        this rate before feature extraction.
    :type sample_rate: int
    :param n_mels: Number of mel filter banks (F in the DCASE baseline).
    :type n_mels: int
    :param n_fft: STFT window size in samples.
    :type n_fft: int
    :param mel_hop_length: STFT hop size in samples.
    :type mel_hop_length: int
    :param n_frames: Context window width in mel time steps (P in the DCASE
        baseline).  ``input_dim = n_frames × n_mels``.
    :type n_frames: int
    """

    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    device: str = "cpu"
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    early_stopping_patience: int | None = None

    # ── Feature extraction — DCASE baseline ──────────────────────────────
    # Must match the parameters passed to MelFrameDataset and the
    # Autoencoder input_dim:  input_dim = n_frames × n_mels = 5 × 64 = 320
    sample_rate: int = 16_000
    n_mels: int = 64
    n_fft: int = 1_024
    mel_hop_length: int = 512
    n_frames: int = 5        # P — context window width in mel time steps

    # ── Optional signal pre-processing ───────────────────────────────────
    # Applied to each raw waveform before mel feature extraction.
    # Not serialised by dataclasses.asdict() — log via HpssTransform.to_config()
    # in the runner instead.
    signal_transform: Callable[[np.ndarray], np.ndarray] | None = field(
        default=None, repr=False
    )

