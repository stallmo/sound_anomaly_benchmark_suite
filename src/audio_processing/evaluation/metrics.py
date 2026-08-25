"""
evaluation/metrics.py — threshold-dependent and threshold-free evaluation metrics.

Two levels of evaluation are supported:

* **Threshold-free**: :func:`compute_auc_roc` — the primary DCASE Task 2
  metric; measures ranking quality across all possible thresholds.
* **Threshold-dependent**: :func:`compute_metrics` — collapses scores to
  binary decisions at a fixed threshold (e.g. from
  :func:`~inference.threshold.calibrate_threshold`) and reports precision,
  recall, F1, and accuracy alongside AUC-ROC.

All functions operate on plain Python lists and floats so they can be used
independently of the rest of the pipeline (no model or DataLoader required).
scikit-learn is used for metric computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    """
    Container for all evaluation metrics produced by :func:`compute_metrics`.

    AUC-ROC is the primary DCASE Task 2 ranking metric and requires no
    threshold.  Precision, recall, F1, and accuracy are binary-decision
    metrics computed at the supplied *threshold*.

    :param auc_roc: Area under the ROC curve in ``[0, 1]``.
        ``0.5`` = random; ``1.0`` = perfect separation.
    :type auc_roc: float
    :param precision: Fraction of flagged files that are truly anomalous.
    :type precision: float
    :param recall: Fraction of anomalous files that were flagged.
        Also known as the true-positive rate / sensitivity.
    :type recall: float
    :param f1_score: Harmonic mean of precision and recall.
    :type f1_score: float
    :param accuracy: Fraction of all files classified correctly.
    :type accuracy: float
    :param threshold: Decision threshold used for binary predictions.
    :type threshold: float
    """

    auc_roc: float
    partial_auc_roc: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    threshold: float
    max_fpr: float


def compute_auc_roc(scores: list[float], labels: list[int], max_fpr: float = None) -> float:
    """
    Compute the Area Under the ROC Curve (AUC-ROC).

    This is the primary evaluation metric for DCASE Task 2 anomaly detection.
    It is threshold-free and measures how well the score *ranks* anomalous
    files above normal ones.

    :param scores: File-level anomaly scores from
        :meth:`~inference.file_scorer.FileScorer.score_files`.
        Higher score = more anomalous.
    :type scores: list[float]
    :param labels: Ground-truth binary labels in the same order as *scores*.
        ``0`` = normal, ``1`` = abnormal.  At least one sample of each
        class must be present.
    :type labels: list[int]
    :param max_fpr: Maximum false positive rate, used for the calculation of the partial AUC-ROC.
        Must be between 0 and 1.
        If None, the full AUC-ROC is computed.
    :returns: AUC-ROC in ``[0, 1]``.
    :rtype: float
    :raises ValueError: If *scores* and *labels* differ in length, if either
        is empty, or if *labels* contains only one class.
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores and labels must have the same length, "
            f"got {len(scores)} and {len(labels)}"
        )
    if len(scores) == 0:
        raise ValueError("scores and labels must not be empty.")
    unique = set(labels)
    if len(unique) < 2:
        raise ValueError(
            f"AUC-ROC is undefined when only one class is present in labels. "
            f"Got classes: {unique}."
        )

    if max_fpr is not None:
        assert 0 <= max_fpr <= 1, "The maximum false positive rate must be between 0 and 1."

    return float(roc_auc_score(labels, scores, max_fpr=max_fpr))

def compute_metrics(
    scores: list[float],
    labels: list[int],
    threshold: float,
    max_fpr: float = None,
) -> EvaluationResult:
    """
    Compute the full suite of evaluation metrics at a fixed *threshold*.

    Files with ``score >= threshold`` are predicted as anomalous.  AUC-ROC
    is included as a threshold-free companion metric.

    :param scores: File-level anomaly scores from
        :meth:`~inference.file_scorer.FileScorer.score_files`.
    :type scores: list[float]
    :param labels: Ground-truth binary labels in the same order as *scores*.
        ``0`` = normal, ``1`` = abnormal.
    :type labels: list[int]
    :param threshold: Decision boundary.  Typically obtained from
        :func:`~inference.threshold.calibrate_threshold`.
    :type threshold: float
    :param max_fpr: Maximum false positive rate, used for the calculation of the partial AUC-ROC.
        Must be between 0 and 1.
        If None, the full AUC-ROC is computed and will be stored as partial AUC-ROC.
    :returns: :class:`EvaluationResult` with all metrics populated.
    :rtype: EvaluationResult
    :raises ValueError: If *scores* and *labels* differ in length, if either
        is empty, or if *labels* contains only one class (AUC-ROC is undefined).
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores and labels must have the same length, "
            f"got {len(scores)} and {len(labels)}"
        )
    if len(scores) == 0:
        raise ValueError(
            "scores and labels must not be empty. "
            "This entity may have no test data (e.g. MIMII DUE id_03–id_05). "
            "Check split.test_paths before calling compute_metrics."
        )

    scores_arr = np.array(scores, dtype=np.float64)
    predictions = (scores_arr >= threshold).astype(int).tolist()

    auc_roc = compute_auc_roc(scores, labels)
    if max_fpr is not None:
        partial_auc_roc = compute_auc_roc(scores, labels, max_fpr)
    else:
        partial_auc_roc = auc_roc
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    accuracy = float(accuracy_score(labels, predictions))

    return EvaluationResult(
        auc_roc=auc_roc,
        partial_auc_roc=partial_auc_roc,
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        accuracy=accuracy,
        threshold=threshold,
        max_fpr=max_fpr if max_fpr is not None else 1.0,
    )


