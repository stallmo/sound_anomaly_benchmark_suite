"""Tests for data/splitting.py — make_train_test_split and DataSplit."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.dataset import AudioFrameDataset
from data.loader import find_wav_files, parse_label_from_path
from data.splitting import DataSplit, make_train_test_split
from tests.conftest import SAMPLE_RATE, _make_wav

# ── helpers ───────────────────────────────────────────────────────────────

def _all_normal(paths: list[Path]) -> bool:
    return all(parse_label_from_path(p) == "normal" for p in paths)


# ── make_train_test_split — split correctness ─────────────────────────────

class TestMakeTrainTestSplit:

    def test_all_abnormal_in_test(self, mimii_dir):
        split = make_train_test_split(mimii_dir)
        all_files = find_wav_files(mimii_dir)
        abnormal = [p for p in all_files if parse_label_from_path(p) == "abnormal"]
        assert set(split.test_abnormal_paths) == set(abnormal)

    def test_test_normal_count_equals_abnormal_count(self, mimii_dir):
        split = make_train_test_split(mimii_dir)
        assert len(split.test_normal_paths) == len(split.test_abnormal_paths)

    def test_train_contains_only_normal_files(self, mimii_dir):
        split = make_train_test_split(mimii_dir)
        assert _all_normal(split.train_paths)

    def test_test_normal_contains_only_normal_files(self, mimii_dir):
        split = make_train_test_split(mimii_dir)
        assert _all_normal(split.test_normal_paths)

    def test_no_overlap_between_train_and_test(self, mimii_dir):
        split = make_train_test_split(mimii_dir)
        assert set(split.train_paths).isdisjoint(set(split.test_paths))

    def test_all_files_accounted_for(self, mimii_dir):
        split = make_train_test_split(mimii_dir)
        all_files = set(find_wav_files(mimii_dir))
        split_files = set(split.train_paths) | set(split.test_paths)
        assert split_files == all_files

    def test_no_abnormal_files_gives_empty_test(self, tmp_path):
        """If there are no abnormal files the test set is empty — not an error."""
        normal_dir = tmp_path / "id_00" / "normal"
        normal_dir.mkdir(parents=True)
        _make_wav(normal_dir / "00000000.wav", SAMPLE_RATE, 1)
        _make_wav(normal_dir / "00000001.wav", SAMPLE_RATE, 1)

        split = make_train_test_split(tmp_path / "id_00")

        assert split.test_abnormal_paths == []
        assert split.test_normal_paths == []
        assert len(split.train_paths) == 2

    def test_more_abnormal_than_normal_raises(self, tmp_path):
        normal_dir = tmp_path / "id_00" / "normal"
        abnormal_dir = tmp_path / "id_00" / "abnormal"
        normal_dir.mkdir(parents=True)
        abnormal_dir.mkdir(parents=True)
        _make_wav(normal_dir / "00000000.wav", SAMPLE_RATE, 1)
        _make_wav(abnormal_dir / "00000000.wav", SAMPLE_RATE, 1)
        _make_wav(abnormal_dir / "00000001.wav", SAMPLE_RATE, 1)

        with pytest.raises(ValueError, match="balanced test set"):
            make_train_test_split(tmp_path / "id_00")


# ── make_train_test_split — reproducibility ───────────────────────────────

class TestMakeTrainTestSplitReproducibility:

    def test_same_seed_gives_same_split(self, mimii_dir):
        s1 = make_train_test_split(mimii_dir, seed=42)
        s2 = make_train_test_split(mimii_dir, seed=42)
        assert s1.train_paths == s2.train_paths
        assert s1.test_normal_paths == s2.test_normal_paths

    def test_different_seeds_can_give_different_splits(self, tmp_path):
        """With enough normal files, different seeds should produce different samples."""
        normal_dir = tmp_path / "id_00" / "normal"
        abnormal_dir = tmp_path / "id_00" / "abnormal"
        normal_dir.mkdir(parents=True)
        abnormal_dir.mkdir(parents=True)
        for i in range(10):
            _make_wav(normal_dir / f"{i:08d}.wav", SAMPLE_RATE, 1)
        _make_wav(abnormal_dir / "00000000.wav", SAMPLE_RATE, 1)

        splits = [
            make_train_test_split(tmp_path / "id_00", seed=s) for s in range(20)
        ]
        unique_test_normal = {tuple(s.test_normal_paths) for s in splits}
        assert len(unique_test_normal) > 1


# ── DataSplit.test_paths property ─────────────────────────────────────────

class TestDataSplitProperty:

    def test_test_paths_is_union_of_normal_and_abnormal(self, mimii_dir):
        split = make_train_test_split(mimii_dir, seed=0)
        expected = set(split.test_normal_paths) | set(split.test_abnormal_paths)
        assert set(split.test_paths) == expected

    def test_test_paths_is_sorted(self, mimii_dir):
        split = make_train_test_split(mimii_dir, seed=0)
        assert split.test_paths == sorted(split.test_paths)


# ── AudioFrameDataset.from_file_paths ─────────────────────────────────────

class TestAudioFrameDatasetFromFilePaths:

    def test_creates_valid_dataset(self, mimii_dir):
        split = make_train_test_split(mimii_dir, seed=0)
        ds = AudioFrameDataset.from_file_paths(
            split.train_paths, frame_length=8_000, hop_length=4_000
        )
        assert len(ds) > 0

    def test_len_matches_expected_frame_count(self, mimii_dir):
        from data.framing import count_frames
        import soundfile as sf

        split = make_train_test_split(mimii_dir, seed=0)
        frame_length, hop_length = 8_000, 4_000

        expected = sum(
            count_frames(sf.info(p).frames, frame_length, hop_length)
            for p in split.train_paths
        )
        ds = AudioFrameDataset.from_file_paths(
            split.train_paths, frame_length=frame_length, hop_length=hop_length
        )
        assert len(ds) == expected

    def test_getitem_returns_tensor(self, mimii_dir):
        import torch
        split = make_train_test_split(mimii_dir, seed=0)
        ds = AudioFrameDataset.from_file_paths(
            split.train_paths, frame_length=8_000, hop_length=4_000
        )
        assert isinstance(ds[0], torch.Tensor)

    def test_train_and_test_datasets_cover_all_files(self, mimii_dir):
        split = make_train_test_split(mimii_dir, seed=0)
        train_ds = AudioFrameDataset.from_file_paths(
            split.train_paths, frame_length=8_000, hop_length=4_000
        )
        test_ds = AudioFrameDataset.from_file_paths(
            split.test_paths, frame_length=8_000, hop_length=4_000
        )
        all_files = set(find_wav_files(mimii_dir))
        ds_files = set(train_ds._file_paths) | set(test_ds._file_paths)
        assert ds_files == all_files

