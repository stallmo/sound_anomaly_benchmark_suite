"""
Shared pytest fixtures for the audio_processing test suite.

Fixtures
--------
wav_mono          : Path to a synthetic mono 16-bit PCM WAV file (1 s, 16 kHz).
wav_multichannel  : Path to a synthetic 8-channel 16-bit PCM WAV file (1 s, 16 kHz).
mimii_dir         : Minimal MIMII directory tree with 2 normal + 1 abnormal file.
                    Returns the path to the id_00 directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

# ── constants used by both fixtures and tests ──────────────────────────────
SAMPLE_RATE: int = 16_000
DURATION_S: float = 1.0
N_SAMPLES: int = int(SAMPLE_RATE * DURATION_S)  # 16 000


def _make_wav(path: Path, samplerate: int, n_channels: int, duration: float = DURATION_S) -> Path:
    """Write a synthetic sine-wave WAV (440 Hz tone) and return *path*."""
    n_samples = int(samplerate * duration)
    t = np.arange(n_samples) / samplerate
    tone = (np.sin(2 * np.pi * 440 * t) * 32_767).astype(np.int16)
    data = tone if n_channels == 1 else np.column_stack([tone] * n_channels)
    wavfile.write(str(path), samplerate, data)
    return path


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def wav_mono(tmp_path) -> Path:
    """Mono 16-bit PCM WAV, 1 s at 16 kHz."""
    return _make_wav(tmp_path / "mono.wav", SAMPLE_RATE, 1)


@pytest.fixture()
def wav_multichannel(tmp_path) -> Path:
    """8-channel 16-bit PCM WAV, 1 s at 16 kHz (mimics a MIMII recording)."""
    return _make_wav(tmp_path / "multichannel.wav", SAMPLE_RATE, 8)


@pytest.fixture()
def mimii_dir(tmp_path) -> Path:
    """
    Minimal MIMII sub-tree::

        <tmp>/fan/id_00/normal/   00000000.wav  00000001.wav
        <tmp>/fan/id_00/abnormal/ 00000000.wav

    Returns the path to the ``id_00`` directory.
    """
    normal_dir = tmp_path / "fan" / "id_00" / "normal"
    abnormal_dir = tmp_path / "fan" / "id_00" / "abnormal"
    normal_dir.mkdir(parents=True)
    abnormal_dir.mkdir(parents=True)

    _make_wav(normal_dir / "00000000.wav", SAMPLE_RATE, 1)
    _make_wav(normal_dir / "00000001.wav", SAMPLE_RATE, 1)
    _make_wav(abnormal_dir / "00000000.wav", SAMPLE_RATE, 1)

    return tmp_path / "fan" / "id_00"

