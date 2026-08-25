from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torchaudio.transforms as T


def compute_log_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int = 16_000,
    n_fft: int = 1_024,
    hop_length: int = 512,
    n_mels: int = 64,
    top_db: float = 80.0,
    signal_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """
    Compute the log-mel spectrogram of a full audio signal.

    Processes the entire signal in one pass and returns the two-dimensional
    ``(n_mels, T)`` spectrogram array.  This is the file-level feature
    extractor used by :class:`~data.dataset.MelFrameDataset`, which caches
    one spectrogram per file and then slides a context window of *n_frames*
    consecutive columns over the time axis — the DCASE baseline approach.

    :param signal: 1-D float array of audio samples, shape ``(n_samples,)``.
    :type signal: np.ndarray
    :param sample_rate: Sample rate of *signal* in Hz.
    :type sample_rate: int
    :param n_fft: STFT window size in samples.
    :type n_fft: int
    :param hop_length: STFT hop size in samples.  Controls time resolution:
        ``T = n_samples // hop_length + 1`` (with torchaudio ``center=True``).
    :type hop_length: int
    :param n_mels: Number of mel filter banks.
    :type n_mels: int
    :param top_db: Dynamic range cap passed to
        :class:`~torchaudio.transforms.AmplitudeToDB`.
    :type top_db: float
    :param signal_transform: Optional callable applied to the raw signal
        **before** mel feature extraction (e.g. an HPSS component selector).
        Must accept and return a 1-D ``np.ndarray`` of audio samples.
    :type signal_transform: Callable[[np.ndarray], np.ndarray] or None
    :returns: Log-mel spectrogram of shape ``(n_mels, T)``, dtype float32.
    :rtype: np.ndarray
    """
    if signal_transform is not None:
        signal = signal_transform(signal)
    waveform = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # (1, n_samples)
    mel = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )(waveform)                                    # (1, n_mels, T)
    mel_db = T.AmplitudeToDB(stype="power", top_db=top_db)(mel)  # (1, n_mels, T)
    return mel_db.squeeze(0).numpy()               # (n_mels, T)


class MelSpectrogramTransform:
    """
    Converts a 1-D audio frame (np.ndarray) into a flattened log-mel
    spectrogram tensor suitable as autoencoder input.

    Pipeline:
        np.ndarray  →  (1, n_samples) float32 tensor
                    →  MelSpectrogram  →  (1, n_mels, time_bins)
                    →  AmplitudeToDB   →  (1, n_mels, time_bins)  [log scale]
                    →  squeeze + flatten  →  (n_mels * time_bins,)

    Output size
    -----------
    With torchaudio's default center=True padding:
        time_bins  = frame_length // hop_length + 1
        n_features = n_mels * time_bins
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 64,
    ) -> None:
        # Instantiate once — reused across all __getitem__ calls in AudioFrameDataset.
        self._mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self._to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        waveform = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # (1, n_samples)
        mel = self._mel(waveform)           # (1, n_mels, time_bins)
        mel_db = self._to_db(mel)           # (1, n_mels, time_bins)
        return mel_db.squeeze(0).flatten()  # (n_mels * time_bins,)

