"""
inference/detector.py — threshold-based anomaly detection.

``AnomalyDetector`` converts a file-level anomaly score (a plain float)
into a binary anomaly decision by comparing it against a fixed threshold.
It is entirely decoupled from the model, feature extraction, and data
loading — it operates solely on floats.

This separation means the same detector can be used in unit tests with
hard-coded scores and in production with scores coming from
:class:`~inference.file_scorer.FileScorer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnomalyResult:
    """
    Result of a single file anomaly detection decision.

    :param path: Path to the scored WAV file.  ``None`` when scoring
        synthetic or in-memory data (e.g. in unit tests).
    :type path: Path or None
    :param score: Aggregated file-level reconstruction error produced by
        :class:`~inference.file_scorer.FileScorer`.
    :type score: float
    :param is_anomaly: ``True`` if *score* exceeds the detector threshold.
    :type is_anomaly: bool
    """

    path: Path | None
    score: float
    is_anomaly: bool


class AnomalyDetector:
    """
    Applies a fixed threshold to file-level anomaly scores.

    A file is flagged as anomalous when its score **strictly exceeds**
    *threshold*.  The threshold is typically chosen on a held-out normal
    validation set (e.g. as a high percentile of normal reconstruction
    errors).

    :param threshold: Decision boundary.  Scores > threshold → anomaly.
    :type threshold: float
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def detect(self, score: float, path: Path | None = None) -> AnomalyResult:
        """
        Apply the threshold to a single file score.

        :param score: Aggregated file-level reconstruction error.
        :type score: float
        :param path: Optional path to attach to the result for traceability.
        :type path: Path or None
        :returns: :class:`AnomalyResult` with ``is_anomaly`` set according
            to whether *score* exceeds the threshold.
        :rtype: AnomalyResult
        """
        result = AnomalyResult(path=path, score=score, is_anomaly=score >= self.threshold)
        return result

    def detect_batch(
        self,
        scores: list[float],
        paths: list[Path] | None = None,
    ) -> list[AnomalyResult]:
        """
        Apply the threshold to a list of file scores.

        :param scores: Ordered list of file-level scores, e.g. from
            :meth:`~inference.file_scorer.FileScorer.score_files`.
        :type scores: list[float]
        :param paths: Optional list of file paths parallel to *scores*.
            When provided, each :class:`AnomalyResult` carries its
            corresponding path.  ``None`` leaves ``path`` unset in all
            results.
        :type paths: list[Path] or None
        :returns: List of :class:`AnomalyResult` in the same order as
            *scores*.
        :rtype: list[AnomalyResult]
        :raises ValueError: If *paths* is provided but its length differs
            from *scores*.
        """
        anomaly_results = [self.detect(score, path) for score, path in zip(scores, paths)]

        return anomaly_results

