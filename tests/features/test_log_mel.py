"""Tests for features/log_mel.py — MelSpectrogramTransform."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from audio_processing.features.log_mel import MelSpectrogramTransform, compute_log_mel_spectrogram
from tests.conftest import N_SAMPLES, SAMPLE_RATE

# ── test-local constants ──────────────────────────────────────────────────
N_FFT = 512
HOP_LENGTH = 160    # transform hop_length
N_MELS = 64
TOP_DB = 80.0

# With center=True (torchaudio default):
#   time_bins = frame_length // hop_length + 1
TIME_BINS = N_SAMPLES // HOP_LENGTH + 1        # 16 000 // 160 + 1 = 101
EXPECTED_OUTPUT_SIZE = N_MELS * TIME_BINS      # 64 × 101 = 6 464


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def frame() -> np.ndarray:
    """Synthetic 1-second 440 Hz sine wave at 16 kHz — matches the WAV fixtures."""
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture()
def transform() -> MelSpectrogramTransform:
    """MelSpectrogramTransform with the test-local default parameters."""
    return MelSpectrogramTransform(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )


# ── output type, rank and dtype ───────────────────────────────────────────

class TestOutputTypeAndRank:

    def test_returns_tensor(self, transform, frame):
        assert isinstance(transform(frame), torch.Tensor)

    def test_output_is_1d(self, transform, frame):
        assert transform(frame).ndim == 1

    def test_output_dtype_is_float32(self, transform, frame):
        assert transform(frame).dtype == torch.float32


# ── output shape ──────────────────────────────────────────────────────────

class TestOutputShape:

    def test_output_size(self, transform, frame):
        assert transform(frame).shape == (EXPECTED_OUTPUT_SIZE,)

    def test_output_size_formula(self, frame):
        """n_mels × (frame_length // hop_length + 1)."""
        n_mels, hop_length = 32, 256
        expected = n_mels * (N_SAMPLES // hop_length + 1)
        t = MelSpectrogramTransform(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=hop_length, n_mels=n_mels,
        )
        assert t(frame).shape == (expected,)

    def test_n_fft_does_not_affect_output_shape(self, frame):
        """n_fft controls frequency resolution only, not the output vector size."""
        t512 = MelSpectrogramTransform(
            sample_rate=SAMPLE_RATE, n_fft=512, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        t1024 = MelSpectrogramTransform(
            sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        assert t512(frame).shape == t1024(frame).shape

    @pytest.mark.parametrize("n_mels,hop_length", [
        (64, 160),
        (128, 160),
        (64, 320),
        (128, 320),
    ])
    def test_parametrized_output_size(self, frame, n_mels, hop_length):
        t = MelSpectrogramTransform(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=hop_length, n_mels=n_mels,
        )
        expected = n_mels * (N_SAMPLES // hop_length + 1)
        assert t(frame).shape == (expected,)


# ── output values ─────────────────────────────────────────────────────────

class TestOutputValues:

    def test_values_are_finite(self, transform, frame):
        assert torch.isfinite(transform(frame)).all()

    def test_silent_frame_is_finite(self, transform):
        """AmplitudeToDB clamps power to amin=1e-10, so silence produces no -inf."""
        silent = np.zeros(N_SAMPLES, dtype=np.float32)
        assert torch.isfinite(transform(silent)).all()

    def test_dynamic_range_within_top_db(self, transform, frame):
        """top_db=80 guarantees max(output) - min(output) ≤ 80 dB."""
        result = transform(frame)
        assert (result.max() - result.min()).item() <= TOP_DB + 1e-4

    def test_louder_signal_has_higher_db_values(self, transform):
        """Scaling amplitude up shifts the dB spectrogram up."""
        t = np.arange(N_SAMPLES) / SAMPLE_RATE
        quiet = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.01
        loud = quiet * 100.0
        assert transform(loud).mean() > transform(quiet).mean()

    def test_deterministic(self, transform, frame):
        """Same input always produces identical output."""
        torch.testing.assert_close(transform(frame), transform(frame))


# ── compute_log_mel_spectrogram ───────────────────────────────────────────

class TestComputeLogMelSpectrogram:
    """Tests for the file-level spectrogram function used by MelFrameDataset."""

    @pytest.fixture()
    def signal(self) -> np.ndarray:
        t = np.arange(N_SAMPLES) / SAMPLE_RATE
        return np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def test_returns_ndarray(self, signal):
        result = compute_log_mel_spectrogram(signal, sample_rate=SAMPLE_RATE)
        assert isinstance(result, np.ndarray)

    def test_output_shape_rows_equal_n_mels(self, signal):
        result = compute_log_mel_spectrogram(signal, sample_rate=SAMPLE_RATE, n_mels=N_MELS)
        assert result.shape[0] == N_MELS

    def test_output_shape_cols_equal_time_bins(self, signal):
        hop = 512
        result = compute_log_mel_spectrogram(signal, sample_rate=SAMPLE_RATE, hop_length=hop)
        expected_cols = N_SAMPLES // hop + 1
        assert result.shape[1] == expected_cols

    def test_output_dtype_is_float32(self, signal):
        result = compute_log_mel_spectrogram(signal, sample_rate=SAMPLE_RATE)
        assert result.dtype == np.float32

    def test_output_is_finite(self, signal):
        result = compute_log_mel_spectrogram(signal, sample_rate=SAMPLE_RATE)
        assert np.isfinite(result).all()

    def test_silent_signal_is_finite(self):
        silent = np.zeros(N_SAMPLES, dtype=np.float32)
        result = compute_log_mel_spectrogram(silent, sample_rate=SAMPLE_RATE)
        assert np.isfinite(result).all()

    def test_n_mels_controls_row_count(self, signal):
        for n in (16, 32, 64):
            result = compute_log_mel_spectrogram(signal, sample_rate=SAMPLE_RATE, n_mels=n)
            assert result.shape[0] == n

    def test_deterministic(self, signal):
        r1 = compute_log_mel_spectrogram(signal, sample_rate=SAMPLE_RATE)
        r2 = compute_log_mel_spectrogram(signal, sample_rate=SAMPLE_RATE)
        np.testing.assert_array_equal(r1, r2)
