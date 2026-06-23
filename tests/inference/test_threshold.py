"""Tests for inference/threshold.py — calibrate_threshold."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from inference.threshold import calibrate_threshold


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def fake_scorer():
    """FileScorer stub whose score_files returns a fixed list of floats."""
    scorer = MagicMock()
    scorer.score_files.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    return scorer


@pytest.fixture()
def normal_paths(tmp_path) -> list[Path]:
    """Five dummy paths — content irrelevant; scorer is mocked."""
    return [tmp_path / f"normal_{i:02d}.wav" for i in range(5)]


# ── calibrate_threshold ───────────────────────────────────────────────────

class TestCalibrateThreshold:

    def test_returns_float(self, normal_paths, fake_scorer):
        pass

    def test_default_percentile_is_95(self, normal_paths, fake_scorer):
        pass

    def test_custom_percentile(self, normal_paths, fake_scorer):
        pass

    def test_percentile_100_returns_max(self, normal_paths, fake_scorer):
        pass

    def test_percentile_0_returns_min(self, normal_paths, fake_scorer):
        pass

    def test_raises_on_empty_paths(self, fake_scorer):
        pass

    def test_raises_on_percentile_below_0(self, normal_paths, fake_scorer):
        pass

    def test_raises_on_percentile_above_100(self, normal_paths, fake_scorer):
        pass

    def test_delegates_scoring_to_file_scorer(self, normal_paths, fake_scorer):
        pass

