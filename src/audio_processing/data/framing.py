from __future__ import annotations

import librosa
import numpy as np


def frame_signal(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    writeable: bool = True,
) -> np.ndarray:
    """
    Split a 1-D audio signal into (potentially overlapping) frames.

    Example with frame_length=4, hop_length=2, signal=[0,1,2,3,4,5]:
        → [[0,1,2,3], [2,3,4,5]]

    :param signal: 1-D float array of audio samples, shape (n_samples,).
    :param frame_length: Number of samples in each frame.
    :param hop_length: Number of samples to advance between consecutive frames.
    :param writeable: Whether the returned array should be writeable.
        Setting to False can save memory when the frames are used read-only.
    :return: 2-D array of shape (n_frames, frame_length).
    :raises ValueError: If frame_length > len(signal) or hop_length <= 0.
    """
    if frame_length > len(signal):
        raise ValueError(f"frame_length ({frame_length}) cannot be greater than signal length ({len(signal)})")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")

    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length, writeable=writeable).T.copy()
    return frames



def count_frames(n_samples: int, frame_length: int, hop_length: int) -> int:
    """
    Calculate the number of frames produced by frame_signal without materialising them.

    Useful for pre-allocating buffers and validating configuration before
    processing large datasets.

    :param n_samples: Total number of audio samples in the signal.
    :param frame_length: Number of samples per frame.
    :param hop_length: Stride between consecutive frames.
    :return: Number of complete frames.
    """
    no_frames = 1 + (n_samples - frame_length) // hop_length
    return max(0, no_frames)

