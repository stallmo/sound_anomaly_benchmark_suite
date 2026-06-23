"""Tests for evaluation/report.py — format_report and print_report."""

from __future__ import annotations

import pytest

from evaluation.metrics import EvaluationResult
from evaluation.report import format_report, print_report


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def result() -> EvaluationResult:
    return EvaluationResult(
        auc_roc=0.923,
        partial_auc_roc=0.891,
        precision=0.867,
        recall=0.929,
        f1_score=0.897,
        accuracy=0.900,
        threshold=0.042,
        max_fpr=0.1,
    )


# ── format_report ─────────────────────────────────────────────────────────

class TestFormatReport:

    def test_returns_string(self, result):
        assert isinstance(format_report(result), str)

    def test_contains_auc_roc_value(self, result):
        assert "0.9230" in format_report(result)

    def test_contains_threshold_value(self, result):
        assert "0.0420" in format_report(result)

    def test_contains_precision_label(self, result):
        assert "Precision" in format_report(result)

    def test_contains_recall_label(self, result):
        assert "Recall" in format_report(result)

    def test_contains_f1_label(self, result):
        assert "F1" in format_report(result)

    def test_contains_accuracy_label(self, result):
        assert "Accuracy" in format_report(result)

    def test_is_multiline(self, result):
        assert "\n" in format_report(result)


# ── print_report ──────────────────────────────────────────────────────────

class TestPrintReport:

    def test_prints_to_stdout(self, result, capsys):
        print_report(result)
        captured = capsys.readouterr()
        assert "AUC-ROC" in captured.out

