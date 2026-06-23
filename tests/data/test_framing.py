"""Tests for data/framing.py."""

from __future__ import annotations

import numpy as np
import pytest

from data.framing import count_frames, frame_signal


# ── frame_signal ──────────────────────────────────────────────────────────

class TestFrameSignal:

    def test_output_shape_no_overlap(self):
        # hop == frame_length → no overlap, 2 non-overlapping frames
        signal = np.arange(8, dtype=float)
        frames = frame_signal(signal, frame_length=4, hop_length=4)
        assert frames.shape == (2, 4)

    def test_output_shape_with_overlap(self):
        # hop < frame_length → overlapping frames
        signal = np.arange(10, dtype=float)
        frames = frame_signal(signal, frame_length=4, hop_length=2)
        # frames at 0, 2, 4, 6 → 4 frames
        assert frames.shape == (4, 4)

    def test_frame_values_correct(self):
        signal = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        frames = frame_signal(signal, frame_length=4, hop_length=2)
        expected = np.array([[0.0, 1.0, 2.0, 3.0],
                              [2.0, 3.0, 4.0, 5.0]])
        np.testing.assert_array_equal(frames, expected)

    def test_single_frame_when_signal_equals_frame_length(self):
        signal = np.arange(4, dtype=float)
        frames = frame_signal(signal, frame_length=4, hop_length=1)
        assert frames.shape == (1, 4)

    def test_returns_copy_not_view(self):
        signal = np.arange(8, dtype=float)
        frames = frame_signal(signal, frame_length=4, hop_length=4)
        signal[0] = 999.0
        assert frames[0, 0] != 999.0

    def test_frame_length_exceeds_signal_raises(self):
        signal = np.arange(3, dtype=float)
        with pytest.raises(ValueError):
            frame_signal(signal, frame_length=4, hop_length=1)

    def test_hop_length_zero_raises(self):
        signal = np.arange(8, dtype=float)
        with pytest.raises(ValueError):
            frame_signal(signal, frame_length=4, hop_length=0)

    def test_hop_length_negative_raises(self):
        signal = np.arange(8, dtype=float)
        with pytest.raises(ValueError):
            frame_signal(signal, frame_length=4, hop_length=-2)


# ── count_frames ──────────────────────────────────────────────────────────

class TestCountFrames:

    def test_agrees_with_frame_signal(self):
        signal = np.arange(20, dtype=float)
        frame_length, hop_length = 6, 3
        frames = frame_signal(signal, frame_length, hop_length)
        assert count_frames(len(signal), frame_length, hop_length) == len(frames)

    def test_no_overlap(self):
        # (10 - 4) // 4 + 1 = 2
        assert count_frames(n_samples=10, frame_length=4, hop_length=4) == 2

    def test_with_overlap(self):
        # (10 - 4) // 2 + 1 = 4
        assert count_frames(n_samples=10, frame_length=4, hop_length=2) == 4

    def test_exact_fit(self):
        # 2 non-overlapping frames of length 4 fill 8 samples exactly
        assert count_frames(n_samples=8, frame_length=4, hop_length=4) == 2

    def test_signal_shorter_than_frame_returns_zero(self):
        assert count_frames(n_samples=3, frame_length=4, hop_length=1) == 0

    @pytest.mark.parametrize("n,fl,hl,expected", [
        (16_000, 8_000, 4_000, 3),   # 1 s @ 16 kHz, 0.5 s frames, 0.25 s hop
        (16_000, 4_000, 2_000, 7),   # same file, smaller frames
        (16_000, 16_000, 16_000, 1), # single frame covering the whole signal
    ])
    def test_realistic_audio_parameters(self, n, fl, hl, expected):
        assert count_frames(n, fl, hl) == expected

