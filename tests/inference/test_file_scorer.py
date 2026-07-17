"""Tests for inference/file_scorer.py — FileScorer and make_mel_dataset_factory."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import Dataset

from audio_processing.models.autoencoder import Autoencoder
from audio_processing.inference.aggregation import mean_score
from audio_processing.inference.frame_scorer import ReconstructionFrameScorer
from audio_processing.inference.file_scorer import FileScorer, make_mel_dataset_factory

# ── test-local constants ──────────────────────────────────────────────────
# Small values so tests stay fast with synthetic 1-second WAV files.
N_MELS = 16
N_FRAMES = 5
MEL_HOP = 512
INPUT_DIM = N_FRAMES * N_MELS   # 5 × 16 = 80


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def mel_factory():
    return make_mel_dataset_factory(
        n_mels=N_MELS,
        n_fft=512,
        mel_hop_length=MEL_HOP,
        n_frames=N_FRAMES,
        sample_rate=16_000,
    )


@pytest.fixture()
def frame_scorer() -> ReconstructionFrameScorer:
    model = Autoencoder(input_dim=INPUT_DIM, hidden_dims=(32,), bottleneck_dim=4)
    return ReconstructionFrameScorer(model, device="cpu")


@pytest.fixture()
def file_scorer(frame_scorer, mel_factory) -> FileScorer:
    return FileScorer(
        frame_scorer=frame_scorer,
        dataset_factory=mel_factory,
        aggregation_fn=mean_score,
    )


# ── make_mel_dataset_factory ──────────────────────────────────────────────

class TestMakeMelDatasetFactory:

    def test_returns_callable(self):
        assert callable(make_mel_dataset_factory())

    def test_factory_produces_dataset(self, mel_factory, wav_mono):
        ds = mel_factory([wav_mono])
        assert isinstance(ds, Dataset)

    def test_factory_output_has_windows(self, mel_factory, wav_mono):
        ds = mel_factory([wav_mono])
        assert len(ds) > 0

    def test_factory_window_dim_matches_n_frames_times_n_mels(self, mel_factory, wav_mono):
        ds = mel_factory([wav_mono])
        assert ds[0].shape == (INPUT_DIM,)


# ── score_file ────────────────────────────────────────────────────────────

class TestScoreFile:

    def test_returns_float(self, file_scorer, wav_mono):
        pass

    def test_score_is_non_negative(self, file_scorer, wav_mono):
        pass

    def test_score_is_finite(self, file_scorer, wav_mono):
        pass

    def test_return_frame_scores_false_gives_float(self, file_scorer, wav_mono):
        pass

    def test_return_frame_scores_true_gives_tuple(self, file_scorer, wav_mono):
        pass

    def test_frame_scores_tensor_is_1d(self, file_scorer, wav_mono):
        pass

    def test_frame_scores_match_aggregated_score(self, file_scorer, wav_mono):
        pass


# ── score_files ───────────────────────────────────────────────────────────

class TestScoreFiles:

    def test_returns_list(self, file_scorer, mimii_dir):
        pass

    def test_length_matches_input(self, file_scorer, mimii_dir):
        pass

    def test_scores_are_floats(self, file_scorer, mimii_dir):
        pass

    def test_order_is_preserved(self, file_scorer, mimii_dir):
        pass

