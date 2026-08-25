"""Tests for data/dataset.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from audio_processing.data.dataset import AudioFrameDataset, MelFrameDataset
from audio_processing.data.framing import count_frames
from tests.conftest import SAMPLE_RATE, N_SAMPLES

# ── test-local constants ──────────────────────────────────────────────────
# Each fixture WAV file is N_SAMPLES (16 000) samples long.
FRAME_LENGTH = 8_000   # 0.5 s at 16 kHz
HOP_LENGTH = 4_000     # 0.25 s  →  50 % overlap
FRAMES_PER_FILE = count_frames(N_SAMPLES, FRAME_LENGTH, HOP_LENGTH)  # 3
N_FILES = 3            # 2 normal + 1 abnormal in mimii_dir
TOTAL_FRAMES = N_FILES * FRAMES_PER_FILE  # 9


@pytest.fixture()
def dataset(mimii_dir) -> AudioFrameDataset:
    return AudioFrameDataset(mimii_dir, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)


# ── __len__ ───────────────────────────────────────────────────────────────

class TestDatasetLen:

    def test_len_matches_total_frame_count(self, dataset):
        assert len(dataset) == TOTAL_FRAMES

    def test_empty_dir_has_zero_length(self, tmp_path):
        ds = AudioFrameDataset(tmp_path, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        assert len(ds) == 0


# ── __getitem__ ───────────────────────────────────────────────────────────

class TestDatasetGetItem:

    def test_returns_tensor(self, dataset):
        item = dataset[0]
        assert isinstance(item, torch.Tensor)

    def test_default_shape(self, dataset):
        # Without a transform the raw frame is returned as a 1-D tensor
        item = dataset[0]
        assert item.shape == (FRAME_LENGTH,)

    def test_default_dtype_is_float32(self, dataset):
        item = dataset[0]
        assert item.dtype == torch.float32

    def test_first_index_accessible(self, dataset):
        _ = dataset[0]  # must not raise

    def test_last_index_accessible(self, dataset):
        _ = dataset[len(dataset) - 1]  # must not raise

    def test_out_of_range_raises_index_error(self, dataset):
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]

    def test_negative_index_raises_index_error(self, dataset):
        with pytest.raises(IndexError):
            _ = dataset[-1]


# ── transform ─────────────────────────────────────────────────────────────

class TestDatasetTransform:

    def test_transform_is_applied(self, mimii_dir):
        def double(x: np.ndarray) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.float32) * 2

        ds_raw = AudioFrameDataset(mimii_dir, FRAME_LENGTH, HOP_LENGTH)
        ds_transformed = AudioFrameDataset(mimii_dir, FRAME_LENGTH, HOP_LENGTH, transform=double)

        torch.testing.assert_close(ds_transformed[0], ds_raw[0] * 2)

    def test_transform_can_change_shape(self, mimii_dir):
        # A transform may reshape the frame (e.g., feature extraction)
        def reshape(x: np.ndarray) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, frame_length)

        ds = AudioFrameDataset(mimii_dir, FRAME_LENGTH, HOP_LENGTH, transform=reshape)
        assert ds[0].shape == (1, FRAME_LENGTH)


# ── multi-file coverage ───────────────────────────────────────────────────

class TestDatasetMultiFile:

    def test_index_spans_multiple_files(self, dataset):
        # Three files each contributing FRAMES_PER_FILE frames.
        assert len(dataset._file_paths) == N_FILES
        assert len(dataset._frame_counts) == N_FILES
        # Every file contributes at least one frame.
        assert all(c > 0 for c in dataset._frame_counts)

    def test_cumulative_counts_length(self, dataset):
        assert len(dataset._cumulative_counts) == N_FILES

    def test_cumulative_counts_values(self, dataset):
        expected = [FRAMES_PER_FILE * i for i in range(1, N_FILES + 1)]
        assert dataset._cumulative_counts == expected


# ── DataLoader compatibility ───────────────────────────────────────────────

class TestDatasetDataLoaderCompatibility:

    def test_dataloader_iterates_without_error(self, dataset):
        loader = DataLoader(dataset, batch_size=4)
        batch = next(iter(loader))
        assert isinstance(batch, torch.Tensor)

    def test_dataloader_batch_shape(self, dataset):
        batch_size = 4
        loader = DataLoader(dataset, batch_size=batch_size)
        batch = next(iter(loader))
        assert batch.shape == (batch_size, FRAME_LENGTH)


# ── MelFrameDataset constants ─────────────────────────────────────────────
# 1-second file at 16 kHz, mel_hop=512 → T = 16000//512 + 1 = 32 time bins
# n_windows per file = 1 + (32 - 5) // 1 = 28
MEL_HOP = 512
N_FRAMES = 5
MEL_N_MELS = 16
WINDOWS_PER_FILE = 1 + (N_SAMPLES // MEL_HOP + 1 - N_FRAMES) // 1  # 28
MEL_INPUT_DIM = N_FRAMES * MEL_N_MELS  # 80


@pytest.fixture()
def mel_dataset(mimii_dir) -> MelFrameDataset:
    from audio_processing.data.loader import find_wav_files
    paths = find_wav_files(mimii_dir)
    return MelFrameDataset(
        file_paths=paths,
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        mel_hop_length=MEL_HOP,
        n_mels=MEL_N_MELS,
        n_frames=N_FRAMES,
        context_hop=1,
    )


# ── MelFrameDataset: length ───────────────────────────────────────────────

class TestMelFrameDatasetLen:

    def test_total_windows_equals_windows_per_file_times_n_files(self, mel_dataset):
        assert len(mel_dataset) == WINDOWS_PER_FILE * N_FILES

    def test_empty_file_list_has_zero_length(self):
        ds = MelFrameDataset(file_paths=[], n_frames=N_FRAMES)
        assert len(ds) == 0


# ── MelFrameDataset: __getitem__ ──────────────────────────────────────────

class TestMelFrameDatasetGetItem:

    def test_returns_tensor(self, mel_dataset):
        import torch
        assert isinstance(mel_dataset[0], torch.Tensor)

    def test_output_shape_is_n_frames_times_n_mels(self, mel_dataset):
        assert mel_dataset[0].shape == (MEL_INPUT_DIM,)

    def test_output_dtype_is_float32(self, mel_dataset):
        import torch
        assert mel_dataset[0].dtype == torch.float32

    def test_first_index_accessible(self, mel_dataset):
        _ = mel_dataset[0]

    def test_last_index_accessible(self, mel_dataset):
        _ = mel_dataset[len(mel_dataset) - 1]

    def test_out_of_range_raises_index_error(self, mel_dataset):
        with pytest.raises(IndexError):
            _ = mel_dataset[len(mel_dataset)]

    def test_consecutive_windows_overlap(self, mel_dataset):
        """Windows at index 0 and 1 share n_frames-1 mel columns."""
        w0 = mel_dataset[0]
        w1 = mel_dataset[1]
        # Last (n_frames-1)*n_mels elements of w0 == first (n_frames-1)*n_mels of w1
        tail = w0[MEL_N_MELS:]
        head = w1[: (N_FRAMES - 1) * MEL_N_MELS]
        import torch
        torch.testing.assert_close(tail, head)


# ── MelFrameDataset: context_hop ──────────────────────────────────────────

class TestMelFrameDatasetContextHop:

    def test_larger_context_hop_reduces_window_count(self, mimii_dir):
        from audio_processing.data.loader import find_wav_files
        paths = find_wav_files(mimii_dir)
        ds_hop1 = MelFrameDataset(paths, n_mels=MEL_N_MELS, mel_hop_length=MEL_HOP,
                                  n_frames=N_FRAMES, context_hop=1)
        ds_hop2 = MelFrameDataset(paths, n_mels=MEL_N_MELS, mel_hop_length=MEL_HOP,
                                  n_frames=N_FRAMES, context_hop=2)
        assert len(ds_hop2) < len(ds_hop1)


# ── MelFrameDataset: DataLoader compatibility ─────────────────────────────

class TestMelFrameDatasetDataLoader:

    def test_dataloader_iterates_without_error(self, mel_dataset):
        loader = DataLoader(mel_dataset, batch_size=8)
        batch = next(iter(loader))
        import torch
        assert isinstance(batch, torch.Tensor)

    def test_dataloader_batch_shape(self, mel_dataset):
        loader = DataLoader(mel_dataset, batch_size=8)
        batch = next(iter(loader))
        assert batch.shape == (8, MEL_INPUT_DIM)
