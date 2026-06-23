"""Tests for data/loader.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.loader import (
    AudioFile1D,
    find_wav_files,
    load_wav,
    parse_label_from_path,
    parse_machine_id_from_path,
)
from tests.conftest import SAMPLE_RATE, N_SAMPLES


# ── load_wav ──────────────────────────────────────────────────────────────

class TestLoadWav:

    def test_returns_audio_file1d(self, wav_mono):
        result = load_wav(wav_mono)
        assert isinstance(result, AudioFile1D)

    def test_signal_is_1d(self, wav_mono):
        result = load_wav(wav_mono, mono=True)
        assert result.signal.ndim == 1

    def test_signal_length_matches_duration(self, wav_mono):
        result = load_wav(wav_mono)
        # librosa normalises to float32 but preserves sample count
        assert len(result.signal) == N_SAMPLES

    def test_native_samplerate_preserved(self, wav_mono):
        result = load_wav(wav_mono, target_samplerate=None)
        assert result.samplerate == SAMPLE_RATE

    def test_resamples_to_target_samplerate(self, wav_mono):
        target = SAMPLE_RATE // 2  # 8 000 Hz
        result = load_wav(wav_mono, target_samplerate=target)
        assert result.samplerate == target
        # Signal length should reflect the new rate (allow ±1 for rounding)
        assert abs(len(result.signal) - target) <= 1

    def test_multichannel_mixed_down_to_1d(self, wav_multichannel):
        result = load_wav(wav_multichannel, mono=True)
        assert result.signal.ndim == 1

    def test_path_stored_on_result(self, wav_mono):
        result = load_wav(wav_mono)
        assert result.path == Path(wav_mono)

    def test_label_and_machine_id_populated(self, mimii_dir):
        path = mimii_dir / "normal" / "00000000.wav"
        result = load_wav(path)
        assert result.label == "normal"
        assert result.machine_id == "id_00"

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_wav(tmp_path / "does_not_exist.wav")


# ── find_wav_files ────────────────────────────────────────────────────────

class TestFindWavFiles:

    def test_discovers_all_wavs(self, mimii_dir):
        # 2 normal + 1 abnormal = 3 total
        result = find_wav_files(mimii_dir)
        assert len(result) == 3

    def test_result_is_sorted(self, mimii_dir):
        result = find_wav_files(mimii_dir)
        assert result == sorted(result)

    def test_returns_path_objects(self, mimii_dir):
        result = find_wav_files(mimii_dir)
        assert all(isinstance(p, Path) for p in result)

    def test_ignores_non_wav_files(self, tmp_path):
        (tmp_path / "note.txt").write_text("hello")
        (tmp_path / "audio.wav").write_bytes(b"RIFF")  # dummy WAV header presence
        result = find_wav_files(tmp_path)
        assert all(p.suffix == ".wav" for p in result)

    def test_empty_dir_returns_empty_list(self, tmp_path):
        result = find_wav_files(tmp_path)
        assert result == []

    def test_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_wav_files(tmp_path / "nonexistent")


# ── parse_label_from_path ─────────────────────────────────────────────────

class TestParseLabelFromPath:

    def test_normal(self):
        path = Path("/data/fan/id_00/normal/00000010.wav")
        assert parse_label_from_path(path) == "normal"

    def test_abnormal(self):
        path = Path("/data/fan/id_00/abnormal/00000010.wav")
        assert parse_label_from_path(path) == "abnormal"

    def test_unrecognised_returns_none(self):
        path = Path("/some/flat/directory/file.wav")
        assert parse_label_from_path(path) is None


# ── parse_machine_id_from_path ────────────────────────────────────────────

class TestParseMachineIdFromPath:

    def test_known_id(self):
        path = Path("/data/fan/id_00/normal/00000010.wav")
        assert parse_machine_id_from_path(path) == "id_00"

    @pytest.mark.parametrize("machine_id", ["id_00", "id_02", "id_04", "id_06"])
    def test_all_mimii_ids(self, machine_id):
        path = Path(f"/data/fan/{machine_id}/normal/00000000.wav")
        assert parse_machine_id_from_path(path) == machine_id

    def test_unrecognised_returns_none(self):
        path = Path("/some/flat/directory/file.wav")
        assert parse_machine_id_from_path(path) is None

