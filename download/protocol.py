"""
download/protocol.py — structural protocols for dataset acquisition and preprocessing.

Two protocols form the acquisition contract for every supported dataset:

* :class:`DatasetDownloader` — fetches raw source files (ZIP archives, tarballs,
  API responses …) into a local raw directory.  No assumptions are made about the
  layout of that directory; that is left to the preprocessor.

* :class:`DatasetPreprocessor` — converts the raw download into the canonical
  ``audio_data/`` layout expected by the rest of the pipeline::

      output_dir/
          {machine_type}/
              {machine_id}/
                  normal/   *.wav
                  abnormal/ *.wav

A :class:`DatasetInfo` record ties both together with dataset-level metadata
(name, machine types, entity IDs).

Concrete implementations live in sibling modules (``mimii.py``, etc.).
Register new datasets in :mod:`download.registry`.

Errors
------
:class:`DownloadError` and :class:`PreprocessingError` are the expected
exception types for failures in each respective phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ── downloader protocol ───────────────────────────────────────────────────


@runtime_checkable
class DatasetDownloader(Protocol):
    """
    Downloads raw source files for a dataset into a local directory.

    Implementations are free to use any transport (HTTP, Zenodo API, torrent,
    Hugging Face Hub …).  No specific directory layout is required for the raw
    download — that is the responsibility of :class:`DatasetPreprocessor`.

    Both methods must be idempotent: calling them when the work is already done
    should be a fast no-op, not an error.
    """

    def download(self, destination: Path) -> None:
        """
        Fetch raw dataset files to *destination*.

        *destination* is created if it does not exist.  Any existing content
        should be left intact to support resumable downloads.

        :param destination: Local directory that will hold the raw files.
        :type destination: Path
        :raises DownloadError: On any network or I/O failure.
        """
        ...

    def is_downloaded(self, destination: Path) -> bool:
        """
        Return ``True`` if all expected raw files are already present.

        Implementations may do a lightweight check (e.g. presence of a
        sentinel file or the expected archives) rather than a full integrity
        verification.

        :param destination: Raw download directory to inspect.
        :type destination: Path
        :returns: ``True`` if a (re-)download can be safely skipped.
        :rtype: bool
        """
        ...


# ── preprocessor protocol ─────────────────────────────────────────────────


@runtime_checkable
class DatasetPreprocessor(Protocol):
    """
    Converts a raw download into the canonical ``audio_data/`` layout.

    Expected output structure after :meth:`preprocess`::

        output_dir/
            {machine_type}/
                {machine_id}/
                    normal/   *.wav   (mono, 16 kHz)
                    abnormal/ *.wav

    Preprocessing may involve: extracting archives, channel down-mixing,
    resampling, renaming files, and writing the final mono WAVs.

    Both methods must be idempotent.
    """

    def preprocess(self, raw_dir: Path, output_dir: Path) -> None:
        """
        Transform the contents of *raw_dir* into the canonical layout under
        *output_dir*.

        :param raw_dir: Directory produced by :meth:`DatasetDownloader.download`.
        :type raw_dir: Path
        :param output_dir: Root of the canonical ``audio_data/`` tree.
            Created if absent.
        :type output_dir: Path
        :raises PreprocessingError: On any extraction or I/O failure.
        """
        ...

    def is_preprocessed(self, output_dir: Path) -> bool:
        """
        Return ``True`` if preprocessing has already been completed.

        A lightweight check (e.g. presence of a known subdirectory) is
        sufficient — a full file-count audit is not required.

        :param output_dir: Root of the canonical ``audio_data/`` tree.
        :type output_dir: Path
        :returns: ``True`` if preprocessing can be skipped.
        :rtype: bool
        """
        ...


# ── metadata container ────────────────────────────────────────────────────


@dataclass
class DatasetInfo:
    """
    Metadata and acquisition components for one dataset.

    Instances are stored in :data:`~download.registry.DATASET_REGISTRY` and
    looked up by the runner via :func:`~download.registry.get_dataset_info`.

    :param name: Short slug used as the registry key and in CLI output
        (e.g. ``"mimii"``).
    :type name: str
    :param description: Human-readable one-liner describing the dataset.
    :type description: str
    :param entity_types: Top-level categories provided by this dataset
        (e.g. ``["fan", "pump", "slider", "valve"]`` for MIMII, or
        ``["engine", "gearbox"]`` for a future dataset).
    :type entity_types: list[str]
    :param entity_ids: Instance IDs available in the dataset
        (e.g. ``["id_00", "id_02", "id_04", "id_06"]``).
    :type entity_ids: list[str]
    :param downloader: Configured :class:`DatasetDownloader` instance.
    :type downloader: DatasetDownloader
    :param preprocessor: Configured :class:`DatasetPreprocessor` instance.
    :type preprocessor: DatasetPreprocessor
    :param metadata: Optional dataset-specific properties that do not belong
        in the fixed schema.  Kept as a plain ``dict`` so that each dataset
        can advertise whatever is relevant without forcing unrelated datasets
        to carry empty fields.

        Common keys (by convention, not enforced):

        * ``"noise_levels_db"`` (:class:`list[int]`) — SNR levels available
          in the dataset, e.g. ``[-6, 0, 6]`` for MIMII.

        Defaults to an empty dict when omitted.
    :type metadata: dict[str, Any]
    """

    name: str
    description: str
    entity_types: list[str]
    entity_ids: list[str]
    downloader: DatasetDownloader
    preprocessor: DatasetPreprocessor
    metadata: dict[str, Any] = field(default_factory=dict)


# ── exceptions ────────────────────────────────────────────────────────────


class DownloadError(RuntimeError):
    """Raised when a dataset download fails (network error, bad status code, etc.)."""


class PreprocessingError(RuntimeError):
    """Raised when dataset preprocessing fails (bad archive, I/O error, etc.)."""

