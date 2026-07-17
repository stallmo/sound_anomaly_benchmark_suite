"""
evaluation/evaluator.py — orchestrated evaluation with experiment tracking.

:class:`Evaluator` is the single entry point for the evaluation pipeline.
It mirrors the injectable-dependency pattern of
:class:`~training.trainer.PtTrainer`: all collaborators are passed at
construction time, and the class owns one public method — :meth:`evaluate`.

Typical usage after training::

    file_scorer = FileScorer(frame_scorer, ...)
    threshold   = calibrate_threshold(split.train_paths, file_scorer)
    evaluator   = Evaluator(file_scorer, threshold, tracker=WandbTracker(...))
    result      = evaluator.evaluate(split.test_paths)
    print_report(result)
"""

from __future__ import annotations

from pathlib import Path

from evaluation.labels import extract_labels
from evaluation.metrics import EvaluationResult, compute_metrics
from evaluation.report import print_report
from inference.file_scorer import FileScorer
from tracking.base import ExperimentTracker, NullTracker


class Evaluator:
    """
    Orchestrates the full evaluation pipeline for a single test set.

    Internally calls, in order:

    1. :meth:`~inference.file_scorer.FileScorer.score_files` — score every
       test file with the trained model.
    2. :func:`~evaluation.labels.extract_labels` — derive ground-truth labels
       from directory names.
    3. :func:`~evaluation.metrics.compute_metrics` — compute AUC-ROC,
       precision, recall, F1, and accuracy.
    4. Tracker logging — ``threshold`` as a param; all metric values as
       metrics.

    :param file_scorer: Configured :class:`~inference.file_scorer.FileScorer`
        wrapping the trained model, using the same framing and feature
        parameters used during training.
    :type file_scorer: FileScorer
    :param threshold: Decision boundary used for binary predictions.
        Typically obtained from
        :func:`~inference.threshold.calibrate_threshold`.
    :type threshold: float
    :param tracker: Experiment tracking back-end.  Defaults to
        :class:`~tracking.base.NullTracker` (silent no-op) so callers
        never need to pass ``None``.
    :type tracker: ExperimentTracker
    """

    def __init__(
        self,
        file_scorer: FileScorer,
        threshold: float,
        max_fpr: float = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.file_scorer = file_scorer
        self.threshold = threshold
        self.max_fpr = max_fpr
        self.tracker: ExperimentTracker = tracker if tracker is not None else NullTracker()

    def evaluate(self, test_paths: list[Path], print_report_to_console: bool = True) -> EvaluationResult:
        """
        Score *test_paths*, compute all metrics, and log them to the tracker.

        Ground-truth labels are inferred from the directory names of
        *test_paths* following the MIMII convention (``normal/`` → 0,
        ``abnormal/`` → 1), so *test_paths* must contain at least one file
        from each class for AUC-ROC to be defined.

        The threshold is logged once as a hyperparameter; AUC-ROC, precision,
        recall, F1, and accuracy are logged as scalar metrics.

        :param test_paths: Ordered list of WAV file paths to evaluate.
            Typically ``split.test_paths`` from
            :func:`~data.splitting.make_train_test_split`.
        :type test_paths: list[Path]
        :param print_report_to_console: Controls whether report is printed.
        :returns: Fully populated :class:`~evaluation.metrics.EvaluationResult`.
        :rtype: EvaluationResult
        :raises ValueError: If *test_paths* contains only one class label, or
            if labels cannot be parsed from any path.
        """
        scores = self.file_scorer.score_files(test_paths)
        labels = extract_labels(test_paths)
        result = compute_metrics(scores, labels, self.threshold)

        self._log(result)

        if print_report_to_console:
            print_report(result)

        return result

    # ── private helpers ───────────────────────────────────────────────────

    def _log(self, result: EvaluationResult) -> None:
        """Log threshold as a param and all metric values to the tracker."""
        self.tracker.log_param("eval_threshold", self.threshold)
        self.tracker.log_metric("eval/auc_roc",   result.auc_roc)
        self.tracker.log_metric("eval/precision", result.precision)
        self.tracker.log_metric("eval/recall",    result.recall)
        self.tracker.log_metric("eval/f1_score",  result.f1_score)
        self.tracker.log_metric("eval/accuracy",  result.accuracy)

