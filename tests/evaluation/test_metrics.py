"""Tests for evaluation/metrics.py — compute_auc_roc and compute_metrics."""

from __future__ import annotations

import pytest

from evaluation.metrics import EvaluationResult, compute_auc_roc, compute_metrics

# ── shared fixtures ───────────────────────────────────────────────────────

# Perfect separation: anomalous scores are strictly above normal scores.
SCORES_PERFECT  = [0.1, 0.2, 0.9, 0.8]
LABELS_PERFECT  = [0,   0,   1,   1  ]

# Random / no signal.
SCORES_RANDOM   = [0.5, 0.5, 0.5, 0.5]
LABELS_RANDOM   = [0,   1,   0,   1  ]

THRESHOLD = 0.5


# ── compute_auc_roc ───────────────────────────────────────────────────────

class TestComputeAucRoc:

    def test_perfect_separation_gives_one(self):
        assert compute_auc_roc(SCORES_PERFECT, LABELS_PERFECT) == pytest.approx(1.0)

    def test_returns_float(self):
        assert isinstance(compute_auc_roc(SCORES_PERFECT, LABELS_PERFECT), float)

    def test_value_in_unit_interval(self):
        auc = compute_auc_roc(SCORES_RANDOM, LABELS_RANDOM)
        assert 0.0 <= auc <= 1.0

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError):
            compute_auc_roc([0.1, 0.9], [0])

    def test_raises_on_single_class(self):
        with pytest.raises(ValueError):
            compute_auc_roc([0.1, 0.2], [0, 0])


# ── compute_metrics ───────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_returns_evaluation_result(self):
        result = compute_metrics(SCORES_PERFECT, LABELS_PERFECT, threshold=THRESHOLD)
        assert isinstance(result, EvaluationResult)

    def test_threshold_stored_in_result(self):
        result = compute_metrics(SCORES_PERFECT, LABELS_PERFECT, threshold=THRESHOLD)
        assert result.threshold == THRESHOLD

    def test_perfect_separation_auc_roc(self):
        result = compute_metrics(SCORES_PERFECT, LABELS_PERFECT, threshold=THRESHOLD)
        assert result.auc_roc == pytest.approx(1.0)

    def test_perfect_separation_recall(self):
        result = compute_metrics(SCORES_PERFECT, LABELS_PERFECT, threshold=THRESHOLD)
        assert result.recall == pytest.approx(1.0)

    def test_perfect_separation_precision(self):
        result = compute_metrics(SCORES_PERFECT, LABELS_PERFECT, threshold=THRESHOLD)
        assert result.precision == pytest.approx(1.0)

    def test_perfect_separation_f1(self):
        result = compute_metrics(SCORES_PERFECT, LABELS_PERFECT, threshold=THRESHOLD)
        assert result.f1_score == pytest.approx(1.0)

    def test_perfect_separation_accuracy(self):
        result = compute_metrics(SCORES_PERFECT, LABELS_PERFECT, threshold=THRESHOLD)
        assert result.accuracy == pytest.approx(1.0)

    def test_all_fields_are_floats(self):
        result = compute_metrics(SCORES_PERFECT, LABELS_PERFECT, threshold=THRESHOLD)
        for field in (result.auc_roc, result.precision, result.recall,
                      result.f1_score, result.accuracy, result.threshold):
            assert isinstance(field, float)

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError):
            compute_metrics([0.1, 0.9], [0], threshold=0.5)

    def test_metric_values_in_unit_interval(self):
        result = compute_metrics(SCORES_RANDOM, LABELS_RANDOM, threshold=THRESHOLD)
        for val in (result.auc_roc, result.precision, result.recall,
                    result.f1_score, result.accuracy):
            assert 0.0 <= val <= 1.0

