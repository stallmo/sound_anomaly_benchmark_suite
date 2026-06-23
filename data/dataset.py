from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Callable

import itertools

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from data.framing import count_frames, frame_signal
from data.loader import find_wav_files, load_wav
from features.log_mel import compute_log_mel_spectrogram
from features.hpss import SignalTransform


class AudioFrameDataset(Dataset):
    """
    PyTorch Dataset that turns a directory of WAV files into individual
    fixed-length frames.

    Pipeline per file:
        load_wav  →  frame_signal  →  transform (optional)

    The transform is intended to be the feature-extraction step (e.g.,
    log-mel extraction + flattening from the features/ module), keeping
    this class decoupled from any specific feature type.

    Attributes
    ----------
    root_dir : Path
    frame_length : int
    hop_length : int
    target_samplerate : int | None
    transform : Callable[[np.ndarray], torch.Tensor] | None
    """

    def __init__(
        self,
        root_dir: str | Path,
        frame_length: int,
        hop_length: int,
        target_samplerate: int | None = None,
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
    ) -> None:
        """
        Scan *root_dir* for WAV files and pre-build the frame index.

        :param root_dir: Directory containing WAV files (searched recursively).
        :param frame_length: Number of audio samples per frame.
        :param hop_length: Stride between consecutive frames in samples.
        :param target_samplerate: Resample all files to this rate; None keeps native rate.
        :param transform: Optional callable applied to each raw frame (np.ndarray)
                          before it is returned. When None the frame is cast to a
                          float32 tensor without any further processing.
        """
        self.root_dir = Path(root_dir)
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.target_samplerate = target_samplerate
        self.transform = transform
        
        # Derived attributes
        self._file_paths = find_wav_files(self.root_dir) # sorted list
        
        # Initializing actions
        self._build_index()

    @classmethod
    def from_file_paths(
        cls,
        file_paths: list[Path],
        frame_length: int,
        hop_length: int,
        target_samplerate: int | None = None,
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
    ) -> "AudioFrameDataset":
        """
        Construct a dataset from an explicit list of file paths instead of
        scanning a directory.

        Intended for use with :func:`data.splitting.make_train_test_split`::

            split    = make_train_test_split(root_dir, seed=42)
            train_ds = AudioFrameDataset.from_file_paths(split.train_paths, ...)
            test_ds  = AudioFrameDataset.from_file_paths(split.test_paths,  ...)

        :param file_paths: Explicit list of WAV file paths.
        :param frame_length: Number of audio samples per frame.
        :param hop_length: Stride between consecutive frames in samples.
        :param target_samplerate: Resample all files to this rate; None keeps native rate.
        :param transform: Optional callable applied to each raw frame before returning.
        :return: A fully initialised :class:`AudioFrameDataset`.
        """
        instance = object.__new__(cls)
        instance.root_dir = None
        instance.frame_length = frame_length
        instance.hop_length = hop_length
        instance.target_samplerate = target_samplerate
        instance.transform = transform
        instance._file_paths = list(file_paths)
        instance._build_index()
        return instance

    def __len__(self) -> int:
        """Return the total number of frames across all files in root_dir."""
        return self._total_frames

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Return the frame at global index *idx*.

        Resolves the global index to a (file_path, local_frame_index) pair using
        the pre-built index, loads the file, extracts the frame, and applies
        the transform if one is configured.

        :param idx: Global frame index in [0, len(self)).
        :return: Frame tensor; shape depends on the transform.
        :raises IndexError: If idx is out of range.
        """

        if idx < 0 or idx >= self._total_frames:
            raise IndexError(
                f"Index {idx} out of range for dataset of length {self._total_frames}"
            )

        file_index = bisect.bisect_right(self._cumulative_counts, idx)
        if file_index == 0:
            local_frame_index = idx
        else:
            offset = self._cumulative_counts[file_index - 1]
            local_frame_index = idx - offset

        audio = load_wav(self._file_paths[file_index], self.target_samplerate, mono=True)
        frames = frame_signal(audio.signal, self.frame_length, self.hop_length)
        frame = frames[local_frame_index]

        if self.transform is not None:
            return self.transform(frame)
        return torch.tensor(frame, dtype=torch.float32)

    def _build_index(self) -> None:
        """
        Pre-compute a list of (file_path, local_frame_index) tuples.

        Called once during __init__. Allows __getitem__ to resolve any global
        index to a specific frame within a specific file efficiently, without
        loading all files into memory up front.
        """
        self._frame_counts: list[int] = []

        for path in self._file_paths:
            # Read only the file header — no audio data is loaded into memory.
            info = sf.info(path)
            n_samples = info.frames  # native sample count (samples per channel)

            # Scale to the target sample rate when resampling will be applied.
            # Note: int() truncation can be off by ±1 vs. the actual resampled
            # length produced by librosa, because resampling backends (soxr,
            # resampy) use their own rounding.  This is benign in practice since
            # a ±1 difference only shifts count_frames by 1 in edge cases.
            if self.target_samplerate is not None and self.target_samplerate != info.samplerate:
                n_samples = int(n_samples * self.target_samplerate / info.samplerate)

            self._frame_counts.append(
                count_frames(n_samples, self.frame_length, self.hop_length)
            )

        # Prefix sum: _cumulative_counts[i] = total frames in files 0..i (inclusive).
        # bisect_right on this list resolves any global idx to a file in O(log n).
        self._cumulative_counts: list[int] = list(itertools.accumulate(self._frame_counts))
        self._total_frames: int = self._cumulative_counts[-1] if self._cumulative_counts else 0


class MelFrameDataset(Dataset):
    """
    DCASE baseline dataset.

    Computes the log-mel spectrogram of each WAV file **once** at
    construction time (caching it in memory), then exposes a sliding window
    of *n_frames* consecutive mel columns as individual training samples.

    Each sample ξ_t ∈ R^D is the concatenation of *n_frames* consecutive
    mel spectrogram frames::

        ξ_t = [X_t, X_{t+1}, …, X_{t+n_frames−1}],  D = n_frames × n_mels

    This matches the DCASE baseline feature extraction where F = *n_mels*
    and P = *n_frames*.

    Memory usage is bounded by the cached spectrograms (≈ n_mels × T × 4
    bytes per file; ~80 KB per 10-second MIMII file at the default params).

    :param file_paths: Ordered list of WAV file paths.
    :type file_paths: list[Path]
    :param sample_rate: Target sample rate.  Files are resampled if their
        native rate differs.
    :type sample_rate: int
    :param n_fft: STFT window size in samples.
    :type n_fft: int
    :param mel_hop_length: STFT hop size in samples.  Determines time
        resolution: ``T ≈ n_samples / mel_hop_length``.
    :type mel_hop_length: int
    :param n_mels: Number of mel filter banks (F in the DCASE baseline).
    :type n_mels: int
    :param n_frames: Context window width in mel time steps (P in the DCASE
        baseline).  Each sample consists of *n_frames* consecutive mel columns.
    :type n_frames: int
    :param context_hop: Stride between consecutive context windows in mel
        time steps.  ``1`` (default) gives the densest coverage and matches
        the DCASE baseline.
    :type context_hop: int
    """

    def __init__(
        self,
        file_paths: list[Path],
        sample_rate: int = 16_000,
        n_fft: int = 1_024,
        mel_hop_length: int = 512,
        n_mels: int = 64,
        n_frames: int = 5,
        context_hop: int = 1,
        signal_transform: SignalTransform | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.file_paths = [Path(p) for p in file_paths]
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.mel_hop_length = mel_hop_length
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.context_hop = context_hop
        self.signal_transform = signal_transform
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        self._spectrograms: list[np.ndarray] = []   # each: (n_mels, T_i)
        self._window_counts: list[int] = []

        self._build_cache()

        self._cumulative_counts: list[int] = list(
            itertools.accumulate(self._window_counts)
        )
        self._total_windows: int = (
            self._cumulative_counts[-1] if self._cumulative_counts else 0
        )

    # ── cache helpers ─────────────────────────────────────────────────────

    def _transform_config(self) -> dict:
        """Return a JSON-serialisable dict describing the signal transform."""
        if self.signal_transform is None:
            return {"signal_transform": "none"}
        if hasattr(self.signal_transform, "to_config"):
            return self.signal_transform.to_config()
        return {"signal_transform": repr(self.signal_transform)}

    def _cache_key(self, path: Path) -> str:
        """
        Compute a stable hex digest that uniquely identifies the spectrogram
        for *path* under the current feature-extraction and transform config.

        The digest encodes:
        - Absolute file path + modification time  (detects stale entries)
        - ``sample_rate``, ``n_fft``, ``mel_hop_length``, ``n_mels``
        - Signal-transform config (component, margin, …)
        """
        fingerprint = json.dumps(
            {
                "path":           str(path.resolve()),
                "mtime":          path.stat().st_mtime,
                "sample_rate":    self.sample_rate,
                "n_fft":          self.n_fft,
                "mel_hop_length": self.mel_hop_length,
                "n_mels":         self.n_mels,
                "transform":      self._transform_config(),
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(fingerprint).hexdigest()

    def _load_or_compute(self, path: Path) -> np.ndarray:
        """
        Return the mel spectrogram for *path*, loading from the disk cache
        when available and writing to it otherwise.

        When ``cache_dir`` is ``None`` the spectrogram is always computed
        fresh (no caching).
        """
        if self.cache_dir is None:
            return self._compute_spectrogram(path)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{self._cache_key(path)}.npy"

        if cache_file.exists():
            return np.load(cache_file)

        spec = self._compute_spectrogram(path)
        np.save(cache_file, spec)
        return spec

    def _compute_spectrogram(self, path: Path) -> np.ndarray:
        """Load *path* and compute its (possibly transformed) mel spectrogram."""
        audio = load_wav(path, target_samplerate=self.sample_rate, mono=True)
        return compute_log_mel_spectrogram(
            audio.signal,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.mel_hop_length,
            n_mels=self.n_mels,
            signal_transform=self.signal_transform,
        )

    def _build_cache(self) -> None:
        """Load every file, compute its mel spectrogram, and cache it."""
        for path in self.file_paths:
            spec = self._load_or_compute(path)  # (n_mels, T_i)
            self._spectrograms.append(spec)
            n_windows = max(0, 1 + (spec.shape[1] - self.n_frames) // self.context_hop)
            self._window_counts.append(n_windows)

    def __len__(self) -> int:
        """Total number of context windows across all files."""
        return self._total_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Return the context window at global index *idx*.

        Columns are concatenated in temporal order matching the DCASE
        baseline: ξ_t = [X_t, X_{t+1}, …].  The returned tensor has shape
        ``(n_frames × n_mels,)`` and dtype ``float32``.

        :param idx: Global window index in ``[0, len(self))``.
        :type idx: int
        :raises IndexError: If *idx* is out of range.
        """
        if idx < 0 or idx >= self._total_windows:
            raise IndexError(
                f"Index {idx} out of range for dataset of length {self._total_windows}"
            )

        file_idx = bisect.bisect_right(self._cumulative_counts, idx)
        offset = self._cumulative_counts[file_idx - 1] if file_idx > 0 else 0
        local_idx = idx - offset

        start_col = local_idx * self.context_hop
        # (n_mels, n_frames) slice → transpose → (n_frames, n_mels) → flatten
        window = self._spectrograms[file_idx][:, start_col : start_col + self.n_frames]
        return torch.tensor(window.T.flatten(), dtype=torch.float32)


