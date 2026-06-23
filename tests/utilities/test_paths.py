"""
tests/utilities/test_paths.py — unit tests for utilities.paths helpers.

Covers :func:`~utilities.paths.effective_data_dir` for all meaningful input
combinations:
- negative noise level  → suffix with leading minus sign (e.g. ``"-6db"``)
- zero noise level      → ``"0db"``
- positive noise level  → suffix without sign (e.g. ``"6db"``)
- ``None`` noise level  → base directory returned unchanged
"""

from __future__ import annotations

from pathlib import Path

import pytest

from utilities.paths import effective_data_dir


BASE = Path("audio_data/downmixed")


class TestEffectiveDataDir:
    """Unit tests for :func:`effective_data_dir`."""

    # ── noise level cases ─────────────────────────────────────────────────

    def test_negative_noise_level(self):
        """Negative SNR should produce a directory with a leading '-'."""
        result = effective_data_dir(BASE, -6)
        assert result == BASE / "-6db"

    def test_zero_noise_level(self):
        """SNR of 0 dB should produce the '0db' subdirectory."""
        result = effective_data_dir(BASE, 0)
        assert result == BASE / "0db"

    def test_positive_noise_level(self):
        """Positive SNR should produce a directory without a sign prefix."""
        result = effective_data_dir(BASE, 6)
        assert result == BASE / "6db"

    def test_none_noise_level_returns_base_unchanged(self):
        """When noise_level_db is None the base path must be returned as-is."""
        result = effective_data_dir(BASE, None)
        assert result == BASE

    # ── suffix format ─────────────────────────────────────────────────────

    def test_suffix_uses_db_lowercase(self):
        """The suffix must use lower-case 'db' to match the on-disk convention."""
        result = effective_data_dir(BASE, -6)
        assert result.name == "-6db"

    @pytest.mark.parametrize("level,expected_name", [
        (-6,  "-6db"),
        ( 0,   "0db"),
        ( 6,   "6db"),
        (-12, "-12db"),
        ( 12,  "12db"),
    ])
    def test_suffix_format_parametrized(self, level: int, expected_name: str):
        result = effective_data_dir(BASE, level)
        assert result.name == expected_name

    # ── base directory is preserved ───────────────────────────────────────

    def test_base_directory_is_parent(self):
        """The base directory must always be the parent of the resolved path."""
        result = effective_data_dir(BASE, -6)
        assert result.parent == BASE

    def test_base_directory_preserved_for_none(self):
        """With None, the returned object must be identical to the base."""
        result = effective_data_dir(BASE, None)
        assert result is BASE   # same object, not just equal

    # ── different base paths ──────────────────────────────────────────────

    def test_cache_base_path(self):
        """Works correctly with an absolute cache-style base path."""
        base = Path("/Users/user/Library/Caches/audio-processing/audio_data")
        result = effective_data_dir(base, -6)
        assert result == base / "-6db"

    def test_relative_base_path(self):
        """Works correctly with a plain relative base path (no sub-folder)."""
        base = Path("audio_data")
        result = effective_data_dir(base, 6)
        assert result == Path("audio_data/6db")

