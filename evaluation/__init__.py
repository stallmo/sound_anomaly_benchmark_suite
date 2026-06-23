"""
evaluation — metrics and reporting for the MIMII anomaly detection pipeline.

Public API
----------
labels.extract_labels           path list → binary label list (0=normal, 1=abnormal)
metrics.EvaluationResult        dataclass holding all computed metrics
metrics.compute_metrics         scores + labels + threshold → EvaluationResult
report.format_report            EvaluationResult → human-readable string
report.print_report             convenience wrapper that prints format_report
evaluator.Evaluator             orchestrates scoring → metrics → tracker logging
"""

from evaluation.evaluator import Evaluator
from evaluation.labels import extract_labels
from evaluation.metrics import EvaluationResult, compute_metrics
from evaluation.report import format_report, print_report

__all__ = [
    "Evaluator",
    "extract_labels",
    "EvaluationResult",
    "compute_metrics",
    "format_report",
    "print_report",
]

