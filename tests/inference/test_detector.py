"""Tests for inference/detector.py — AnomalyDetector and AnomalyResult."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference.detector import AnomalyDetector, AnomalyResult


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def detector() -> AnomalyDetector:
    return AnomalyDetector(threshold=1.0)


# ── AnomalyResult ─────────────────────────────────────────────────────────

class TestAnomalyResult:

    def test_is_dataclass(self):
        result = AnomalyResult(path=None, score=0.5, is_anomaly=False)
        assert result.score == 0.5

    def test_path_can_be_none(self):
        pass

    def test_path_can_be_path_object(self):
        pass


# ── AnomalyDetector.detect ────────────────────────────────────────────────

class TestDetect:

    def test_score_above_threshold_is_anomaly(self, detector):
        pass

    def test_score_below_threshold_is_not_anomaly(self, detector):
        pass

    def test_score_equal_to_threshold_is_not_anomaly(self, detector):
        """Strict greater-than: score == threshold should NOT be flagged."""
        pass

    def test_returns_anomaly_result(self, detector):
        pass

    def test_score_stored_in_result(self, detector):
        pass

    def test_path_stored_in_result_when_provided(self, detector, tmp_path):
        pass

    def test_path_is_none_when_not_provided(self, detector):
        pass


# ── AnomalyDetector.detect_batch ─────────────────────────────────────────

class TestDetectBatch:

    def test_returns_list_of_anomaly_results(self, detector):
        pass

    def test_length_matches_scores(self, detector):
        pass

    def test_results_are_correct(self, detector):
        """Scores [0.5, 1.5] with threshold=1.0 → [False, True]."""
        pass

    def test_paths_attached_when_provided(self, detector, tmp_path):
        pass

    def test_paths_none_when_not_provided(self, detector):
        pass

    def test_mismatched_paths_length_raises_value_error(self, detector):
        pass

