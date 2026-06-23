"""Tests for evaluation/evaluator.py — Evaluator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from evaluation.evaluator import Evaluator
from evaluation.metrics import EvaluationResult
from tracking.base import NullTracker


# ── helpers ───────────────────────────────────────────────────────────────

def _path(label: str, n: int = 0) -> Path:
    return Path(f"audio_data/fan/id_00/{label}/{n:08d}.wav")


# Balanced test set: 3 normal + 3 abnormal
TEST_PATHS = (
    [_path("normal", i) for i in range(3)]
    + [_path("abnormal", i) for i in range(3)]
)

# Scores that perfectly separate the two classes
PERFECT_SCORES = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]

THRESHOLD = 0.5


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_file_scorer():
    """FileScorer stub that returns PERFECT_SCORES for any input."""
    scorer = MagicMock()
    scorer.score_files.return_value = PERFECT_SCORES
    return scorer


@pytest.fixture()
def evaluator(mock_file_scorer):
    return Evaluator(mock_file_scorer, threshold=THRESHOLD)


# ── construction ──────────────────────────────────────────────────────────

class TestEvaluatorConstruction:

    def test_default_tracker_is_null_tracker(self, mock_file_scorer):
        ev = Evaluator(mock_file_scorer, threshold=THRESHOLD)
        assert isinstance(ev.tracker, NullTracker)

    def test_custom_tracker_is_stored(self, mock_file_scorer):
        tracker = NullTracker()
        ev = Evaluator(mock_file_scorer, threshold=THRESHOLD, tracker=tracker)
        assert ev.tracker is tracker

    def test_threshold_is_stored(self, mock_file_scorer):
        ev = Evaluator(mock_file_scorer, threshold=0.42)
        assert ev.threshold == pytest.approx(0.42)


# ── evaluate ──────────────────────────────────────────────────────────────

class TestEvaluate:

    def test_returns_evaluation_result(self, evaluator):
        result = evaluator.evaluate(TEST_PATHS)
        assert isinstance(result, EvaluationResult)

    def test_delegates_scoring_to_file_scorer(self, evaluator, mock_file_scorer):
        evaluator.evaluate(TEST_PATHS)
        mock_file_scorer.score_files.assert_called_once_with(TEST_PATHS)

    def test_perfect_scores_give_auc_roc_one(self, evaluator):
        result = evaluator.evaluate(TEST_PATHS)
        assert result.auc_roc == pytest.approx(1.0)

    def test_threshold_stored_in_result(self, evaluator):
        result = evaluator.evaluate(TEST_PATHS)
        assert result.threshold == pytest.approx(THRESHOLD)


# ── tracker logging ───────────────────────────────────────────────────────

class TestEvaluatorTrackerLogging:

    def test_logs_threshold_as_param(self, mock_file_scorer):
        tracker = MagicMock()
        ev = Evaluator(mock_file_scorer, threshold=THRESHOLD, tracker=tracker)
        ev.evaluate(TEST_PATHS)
        tracker.log_param.assert_called_once_with("eval_threshold", THRESHOLD)

    def test_logs_auc_roc_metric(self, mock_file_scorer):
        tracker = MagicMock()
        ev = Evaluator(mock_file_scorer, threshold=THRESHOLD, tracker=tracker)
        ev.evaluate(TEST_PATHS)
        logged_keys = {c.args[0] for c in tracker.log_metric.call_args_list}
        assert "eval/auc_roc" in logged_keys

    def test_logs_all_five_metrics(self, mock_file_scorer):
        tracker = MagicMock()
        ev = Evaluator(mock_file_scorer, threshold=THRESHOLD, tracker=tracker)
        ev.evaluate(TEST_PATHS)
        logged_keys = {c.args[0] for c in tracker.log_metric.call_args_list}
        assert logged_keys == {
            "eval/auc_roc",
            "eval/precision",
            "eval/recall",
            "eval/f1_score",
            "eval/accuracy",
        }

    def test_null_tracker_does_not_raise(self, mock_file_scorer):
        ev = Evaluator(mock_file_scorer, threshold=THRESHOLD, tracker=NullTracker())
        ev.evaluate(TEST_PATHS)  # must not raise

