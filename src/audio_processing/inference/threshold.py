"""
inference/threshold.py — threshold calibration from normal training scores.

After training, the detection threshold is derived by running the fully
trained model over all normal training files and taking a high percentile
of the resulting score distribution.  Any file whose score exceeds this
threshold at test time is flagged as anomalous.

Placing this here — rather than in ``training/`` or ``models/`` — keeps the
boundary clean: threshold calibration *is* inference applied to training
data.  It uses the exact same :class:`~inference.file_scorer.FileScorer`
machinery as test-time scoring, and produces a ``float`` consumed by
:class:`~inference.detector.AnomalyDetector`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from audio_processing.inference.file_scorer import FileScorer


def calibrate_threshold(
    normal_file_paths: list[Path],
    file_scorer: FileScorer,
    percentile: float = 95.0,
) -> float:
    """
    Score every normal training file and return the *percentile*-th
    percentile of those scores as the detection threshold.

    A score at ``percentile=95`` means that 95 % of normal training files
    score at or below the threshold — so roughly 5 % of normal files would
    be false-positives at test time.  Increase *percentile* to reduce
    false-positive rate at the cost of missing more anomalies.

    Usage::

        # After PtTrainer.train() completes:
        file_scorer = FileScorer(frame_scorer, aggregation_fn=mean_score, ...)
        threshold   = calibrate_threshold(split.train_paths, file_scorer, percentile=95.0)
        detector    = AnomalyDetector(threshold)

    :param normal_file_paths: Paths to all normal training WAV files — the
        same files used during :meth:`~training.trainer.PtTrainer.train`.
        Must be non-empty.
    :type normal_file_paths: list[Path]
    :param file_scorer: A fully configured :class:`~inference.file_scorer.FileScorer`
        wrapping the trained model, using the same framing and feature
        parameters as training.
    :type file_scorer: FileScorer
    :param percentile: Percentile in ``[0, 100]`` used to select the
        threshold from the distribution of normal training scores.
        Default: ``95.0``.
    :type percentile: float
    :returns: Scalar threshold.  Pass directly to
        :class:`~inference.detector.AnomalyDetector`.
    :rtype: float
    :raises ValueError: If *normal_file_paths* is empty or *percentile* is
        outside ``[0, 100]``.
    """
    if not normal_file_paths:
        raise ValueError("normal_file_paths must not be empty")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")

    scores = file_scorer.score_files(normal_file_paths)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    return torch.quantile(scores_tensor, percentile / 100.0).item()

