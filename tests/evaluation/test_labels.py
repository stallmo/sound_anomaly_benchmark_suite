"""Tests for evaluation/labels.py — extract_labels."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_processing.evaluation.labels import extract_labels


# ── helpers ───────────────────────────────────────────────────────────────

def _path(label: str) -> Path:
    """Construct a fake MIMII-style path with the given label directory."""
    return Path(f"audio_data/fan/id_00/{label}/00000000.wav")


# ── extract_labels ────────────────────────────────────────────────────────

class TestExtractLabels:

    def test_normal_path_gives_zero(self):
        assert extract_labels([_path("normal")]) == [0]

    def test_abnormal_path_gives_one(self):
        assert extract_labels([_path("abnormal")]) == [1]

    def test_preserves_order(self):
        paths = [_path("normal"), _path("abnormal"), _path("normal")]
        assert extract_labels(paths) == [0, 1, 0]

    def test_empty_list_returns_empty(self):
        assert extract_labels([]) == []

    def test_raises_on_unrecognised_label(self):
        with pytest.raises(ValueError, match="normal.*abnormal"):
            extract_labels([Path("audio_data/fan/id_00/unknown/file.wav")])

    def test_length_matches_input(self):
        paths = [_path("normal")] * 3 + [_path("abnormal")] * 2
        assert len(extract_labels(paths)) == 5

