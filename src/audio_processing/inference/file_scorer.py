"""
inference/file_scorer.py — file-level anomaly scoring.

``FileScorer`` is the main orchestrator of the inference pipeline.  For
each WAV file it:

1. Invokes the injected *dataset_factory* to build a single-file
   :class:`~data.dataset.MelFrameDataset` that replicates the framing
   configuration used during training.
2. Iterates over context windows in batches via a
   :class:`~torch.utils.data.DataLoader`.
3. Collects per-frame reconstruction errors from
   :class:`~inference.frame_scorer.FrameScorer`.
4. Reduces the frame errors to a single scalar via the injected
   :data:`~inference.aggregation.AggregationFn`.

The file scorer produces float scores only; threshold application is the
responsibility of :class:`~inference.detector.AnomalyDetector`.

Use :func:`make_mel_dataset_factory` to construct the *dataset_factory*
argument without writing lambdas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from audio_processing.data.dataset import MelFrameDataset
from audio_processing.features.hpss import SignalTransform
from audio_processing.inference.aggregation import AggregationFn, mean_score
from audio_processing.inference.frame_scorer import FrameScorer
from audio_processing.utilities.paths import DEFAULT_MEL_CACHE_DIR


# ── dataset factory helper ────────────────────────────────────────────────


def make_mel_dataset_factory(
    n_mels: int = 64,
    n_fft: int = 1_024,
    mel_hop_length: int = 512,
    n_frames: int = 5,
    sample_rate: int = 16_000,
    context_hop: int = 1,
    signal_transform: SignalTransform | None = None,
    cache_dir: Path | None = DEFAULT_MEL_CACHE_DIR,
) -> Callable[[list[Path]], Dataset]:
    """
    Return a factory that builds a :class:`~data.dataset.MelFrameDataset`
    from a list of file paths.

    The factory captures the feature-extraction parameters so that the same
    configuration used during training is replicated exactly at inference
    time.  Pass the returned callable as *dataset_factory* when constructing
    :class:`FileScorer`.

    :param n_mels: Number of mel filter banks (F in the DCASE baseline paper).
    :type n_mels: int
    :param n_fft: STFT window size in samples.
    :type n_fft: int
    :param mel_hop_length: STFT hop size in samples.
    :type mel_hop_length: int
    :param n_frames: Context window width in mel time steps (P in the DCASE
        baseline paper).
    :type n_frames: int
    :param sample_rate: Target sample rate in Hz.
    :type sample_rate: int
    :param context_hop: Stride between consecutive context windows in mel
        time steps.  ``1`` (default) matches the DCASE baseline.
    :type context_hop: int
    :param signal_transform: Optional signal pre-processing callable
        (e.g. an HPSS component selector) applied to each raw waveform
        before mel extraction.  Must match the transform used at training
        time.
    :type signal_transform: Callable[[np.ndarray], np.ndarray] or None
    :param cache_dir: Directory for on-disk spectrogram cache.  Each entry
        is a ``.npy`` file keyed by a hash of the source path, its
        modification time, all feature params, and the transform config.
        Defaults to :data:`~utilities.paths.DEFAULT_MEL_CACHE_DIR`
        (platform cache dir).  Pass ``None`` to disable caching entirely.
    :type cache_dir: Path or None
    :returns: ``Callable[[list[Path]], Dataset]`` suitable for
        :class:`FileScorer`.
    :rtype: Callable
    """
    def factory(paths: list[Path]) -> Dataset:
        return MelFrameDataset(
            file_paths=paths,
            sample_rate=sample_rate,
            n_fft=n_fft,
            mel_hop_length=mel_hop_length,
            n_mels=n_mels,
            n_frames=n_frames,
            context_hop=context_hop,
            signal_transform=signal_transform,
            cache_dir=cache_dir,
        )
    return factory


# ── FileScorer ────────────────────────────────────────────────────────────


class FileScorer:
    """
    Scores WAV files by aggregating per-frame reconstruction errors.

    The dataset used for framing is fully injectable via *dataset_factory*,
    keeping :class:`FileScorer` decoupled from feature-extraction details.
    Use :func:`make_mel_dataset_factory` to build the factory.

    :param frame_scorer: Configured :class:`~inference.frame_scorer.FrameScorer`
        instance holding the trained model.
    :type frame_scorer: FrameScorer
    :param dataset_factory: Callable that accepts a ``list[Path]`` and
        returns a :class:`~torch.utils.data.Dataset` of context-window
        tensors.  Must be configured with the **same** parameters used
        during training.
    :type dataset_factory: Callable[[list[Path]], Dataset]
    :param aggregation_fn: Callable that reduces a 1-D tensor of frame
        errors to a single float file score.  Defaults to
        :func:`~inference.aggregation.mean_score`.
    :type aggregation_fn: AggregationFn
    :param batch_size: Number of context windows per DataLoader batch.
    :type batch_size: int
    """

    def __init__(
        self,
        frame_scorer: FrameScorer,
        dataset_factory: Callable[[list[Path]], Dataset],
        aggregation_fn: AggregationFn = mean_score,
        batch_size: int = 256,
    ) -> None:
        self.frame_scorer = frame_scorer
        self.dataset_factory = dataset_factory
        self.aggregation_fn = aggregation_fn
        self.batch_size = batch_size

    def score_file(
        self,
        path: Path,
        return_frame_scores: bool = False,
    ) -> float | tuple[float, Tensor]:
        """
        Score a single WAV file.

        Builds a single-file :class:`~data.dataset.MelFrameDataset` via
        *dataset_factory*, collects all context-window errors, applies
        ``aggregation_fn``, and returns the file-level score.

        :param path: Path to the WAV file to score.
        :type path: Path
        :param return_frame_scores: When ``True``, also return the raw
            per-frame error tensor alongside the aggregated score, which is
            useful for time-localising anomalies within a file.
            When ``False`` (default), only the aggregated float is returned.
        :type return_frame_scores: bool
        :returns: The aggregated file score as a ``float`` when
            *return_frame_scores* is ``False``, or a
            ``(float, Tensor)`` tuple when ``True``.
        :rtype: float or tuple[float, Tensor]
        """
        dataset = self.dataset_factory([path])
        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        errors_collected = torch.cat(
            [self.frame_scorer.score_batch(batch) for batch in data_loader]
        )
        file_score = self.aggregation_fn(errors_collected)

        if return_frame_scores:
            return file_score, errors_collected
        return file_score

    def score_files(self, paths: list[Path]) -> list[float]:
        """
        Score a list of WAV files, returning one float per file.

        Each file is scored independently via :meth:`score_file`; there is
        no cross-file state.  The returned list preserves the order of
        *paths*.

        :param paths: Ordered list of WAV file paths to score.
        :type paths: list[Path]
        :returns: List of aggregated file scores in the same order as
            *paths*.
        :rtype: list[float]
        """
        return [
            self.score_file(path, return_frame_scores=False)
            for path in tqdm(paths, desc="Scoring files", unit="file")
        ]
