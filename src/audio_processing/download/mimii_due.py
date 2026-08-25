"""
MIMII DUE: Sound Dataset for Malfunctioning Industrial Machine Investigation
and Inspection with Domain Shifts due to Changes in Operational and
Environmental Conditions.
Zenodo record 4740355: https://zenodo.org/record/4740355

------------------------------------------------
The runner passes a ``tempfile.TemporaryDirectory`` as the destination, so
ZIP archives are stored only for the duration of preprocessing and then
deleted automatically.  The in-memory layout inside that temp dir is::

    <tmp>/
        dev_data_fan.zip
        dev_data_gearbox.zip
        dev_data_pump.zip
        dev_data_slider.zip
        dev_data_valve.zip

Layout after extraction (inside each ZIP)
-----------------------------------------
::

    dev_data_{entity_type}/
        {entity_type}/
            train/
                section_0{N}_{domain}_train_*.wav
            target_test/
                section_0{N}_target_test_{normal|anomaly}*.wav
            source_test/
                section_0{N}_source_test_{normal|anomaly}*.wav
    eval_data_{entity_type}_train/
        {entity_type}/
            train/
                section_0{N}_source_train_normal_*.wav
    …

Note: dev_data_{entity_type}/ contains sections (=machine IDs) 00, 01, 02 including the ground truth in the *_test
subdirectories. The eval_data_* directory contains sections 03, 04, 05, but without any ground truth labels.
Hence, there is no test dataset for sections 03, 04, 05, and these sections are not included in the canonical output layout under test/.

Unlike MIMII (record 3384388) there are no noise-level variants — the
domain shift between ``source_test`` and ``target_test`` is the defining
characteristic of MIMII DUE.

Expected output of :class:`MimiiDuePreprocessor`
-------------------------------------------------
::

    output_dir/
        fan/
            id_00/
                normal/   *.wav   (mono, 16 kHz)
                abnormal/ *.wav
            id_01/ …
        gearbox/ …
        pump/ …
        slider/ …
        valve/ …

The ``train/`` files are all normal; ``source_test/`` and ``target_test/``
contain both normal and abnormal files distinguished by their filename
prefix (``normal_`` / ``anomaly_``).

Neither :meth:`MimiiDueDownloader.download` nor
:meth:`MimiiDuePreprocessor.preprocess` is implemented yet — both raise
:class:`NotImplementedError`.  See the inline TODO comments for implementation
guidance.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict

import requests

from audio_processing.download.protocol import DownloadError, PreprocessingError  # noqa: F401
from audio_processing.download.helpers import download_file                        # noqa: F401

# ── Zenodo source URLs ────────────────────────────────────────────────────

#: Map of entity_type → { zip_prefix → Zenodo download URL }.
#: Each prefix becomes the local filename stem: ``{prefix}_{entity_type}.zip``.
#: Source: https://zenodo.org/record/4740355
#: TODO: verify final URLs once the record is publicly accessible.
MIMII_DUE_ZENODO_URLS: dict[str, dict[str, str]] = {
    "fan": {
        "dev_data":  "https://zenodo.org/records/4740355/files/dev_data_fan.zip?download=1",
        "eval_data": "https://zenodo.org/records/4740355/files/eval_data_fan_train.zip?download=1",
    },
    "gearbox": {
        "dev_data":  "https://zenodo.org/records/4740355/files/dev_data_gearbox.zip?download=1",
        "eval_data": "https://zenodo.org/records/4740355/files/eval_data_gearbox_train.zip?download=1",
    },
    "pump": {
        "dev_data":  "https://zenodo.org/records/4740355/files/dev_data_pump.zip?download=1",
        "eval_data": "https://zenodo.org/records/4740355/files/eval_data_pump_train.zip?download=1",
    },
    "slider": {
        "dev_data":  "https://zenodo.org/records/4740355/files/dev_data_slider.zip?download=1",
        "eval_data": "https://zenodo.org/records/4740355/files/eval_data_slider_train.zip?download=1",
    },
    "valve": {
        "dev_data":  "https://zenodo.org/records/4740355/files/dev_data_valve.zip?download=1",
        "eval_data": "https://zenodo.org/records/4740355/files/eval_data_valve_train.zip?download=1",
    },
}

#: All entity types present in the MIMII DUE dataset.
MIMII_DUE_ENTITY_TYPES: list[str] = list(MIMII_DUE_ZENODO_URLS.keys())

#: All entity IDs present in the MIMII DUE dataset (but only 00, 01, 02 have test sets)
MIMII_DUE_ENTITY_IDS: list[str] = [
    "id_00", "id_01", "id_02", "id_03", "id_04", "id_05",
]

# Subdirectory names used inside each extracted entity archive.
_TRAIN_DIR = "train"
_SOURCE_TEST_DIR = "source_test"
_TARGET_TEST_DIR = "target_test"

# Filename prefixes that distinguish normal from anomalous test clips.
_NORMAL_PREFIX = "normal_"
_ANOMALY_PREFIX = "anomaly_"


# ── downloader ────────────────────────────────────────────────────────────


class MimiiDueDownloader:
    """
    Downloads MIMII DUE ZIP archives from Zenodo record 4740355.

    Unlike MIMII (record 3384388), MIMII DUE has no noise-level variants —
    one ZIP per entity type covers all operating conditions.

    :param entity_types: Which entity types to download.
        Defaults to all five (fan, gearbox, pump, slider, valve).
    :type entity_types: list[str] or None
    """

    def __init__(self, entity_types: list[str] | None = None) -> None:
        self.entity_types = entity_types or list(MIMII_DUE_ENTITY_TYPES)

    def set_entity_types(self, entity_types: list[str]) -> None:
        self.entity_types = entity_types

    def download(self, destination: Path) -> None:
        """
        Download one ZIP archive per entity type to *destination*.

        When called from the runner, *destination* is a
        ``tempfile.TemporaryDirectory`` managed by the caller and deleted
        automatically once preprocessing completes, so only the final mono
        WAVs are kept on disk.

        TODO: implement using ``download_file`` (from ``download.helpers``)
        with a ``tqdm`` progress bar.  Suggested steps:

        1. ``destination.mkdir(parents=True, exist_ok=True)``
        2. For each ``entity_type`` in :attr:`entity_types`, look up
           ``MIMII_DUE_ZENODO_URLS[entity_type]``.
        3. Stream the response in chunks into
           ``destination / f"dev_data_{entity_type}.zip"``.
        4. Wrap network / I/O errors in :class:`~download.protocol.DownloadError`.

        :param destination: Directory to store the raw ZIP files.
            Typically a ``tempfile.TemporaryDirectory`` managed by the caller.
        :raises DownloadError: On HTTP or I/O failure.
        :raises NotImplementedError: Until this method is implemented.
        """

        # ensure directory exists
        destination.mkdir(parents=True, exist_ok=True)

        # Loop through entity types and get urls
        for entity_type in self.entity_types:
            urls_dict = MIMII_DUE_ZENODO_URLS[entity_type]
            for prefix, download_url in urls_dict.items():
                save_path = destination / f"{prefix}_{entity_type}.zip"
                try:
                    download_file(download_url, save_path)
                except requests.RequestException as e:
                    raise DownloadError(f"Failed to download {download_url}") from e

    def is_downloaded(self, destination: Path) -> bool:
        """
        Return ``True`` if all expected ZIP archives exist in *destination*.

        Checks for the presence of ``destination / "dev_data_{entity_type}.zip"``
        for every entity type in :attr:`entity_types`.

        :param destination: Raw download directory to inspect.
        """
        for entity_type in self.entity_types:
            urls_dict = MIMII_DUE_ZENODO_URLS[entity_type]
            for prefix, download_url in urls_dict.items():
                save_path = destination / f"{prefix}_{entity_type}.zip"
                if not save_path.is_file():
                    return False
        return True


# ── preprocessor ─────────────────────────────────────────────────────────


class MimiiDuePreprocessor:
    """
    Converts extracted MIMII DUE archives into the canonical ``audio_data/``
    layout.

    The raw MIMII DUE WAV files are **mono (1-channel), 16 kHz** and are
    organised by split subdirectory inside each ZIP::

        dev_data_{entity_type}/
            {entity_type}/
                train/
                    section_0{N}_{domain}_train_*.wav
                target_test/
                    section_0{N}_target_test_{normal|anomaly}*.wav
                source_test/
                    section_0{N}_source_test_{normal|anomaly}*.wav
        eval_data_{entity_type}_train/
            {entity_type}/
                train/
                    section_0{N}_source_train_normal_*.wav

    Mapping to the canonical layout:

    * ``train/*.wav``                → ``normal/``
    * ``source_test/normal_*.wav``   → ``normal/``
    * ``source_test/anomaly_*.wav``  → ``abnormal/``
    * ``target_test/normal_*.wav``   → ``normal/``
    * ``target_test/anomaly_*.wav``  → ``abnormal/``

    .. note::
        The domain split (source vs. target test) is **not** preserved in
        the canonical output layout.  Both test splits are merged into
        ``normal/`` / ``abnormal/`` subdirectories, consistent with the
        MIMII layout and the rest of the pipeline.
        Downstream evaluation that requires the domain split must implement
        its own label extraction from filenames or directory metadata.
    """

    def _extract_metadata_from_filename(self, file_name: str) -> Dict[str, Any]:
        """
        Parse MIMII DUE WAV metadata from *file_name*.

        Expected filename format::

            section_0{N}_{domain}_{split}_{label}_{index}.wav

        Examples::

            section_00_source_train_normal_0000000.wav
            section_01_target_test_anomaly_0000001.wav

        :param file_name: WAV filename (basename only, not full path).
        :returns: ``{"label": "normal"|"abnormal", "entity_id": "id_0N",
                     "domain": "source"|"target"}``
        :raises PreprocessingError: If the filename does not match the
            expected format.
        """
        parts = file_name.split("_")
        # Minimum expected parts: section / id / domain / split / label / index.wav
        if len(parts) < 6:
            raise PreprocessingError(
                f"Unexpected filename format (too few underscore-separated "
                f"parts): {file_name!r}"
            )

        section_id = parts[1]   # e.g. "00"
        domain     = parts[2]   # e.g. "source" or "target"
        label      = parts[4]   # e.g. "normal" or "anomaly"

        if label not in ("normal", "anomaly"):
            raise PreprocessingError(
                f"Could not parse label from filename {file_name!r}. "
                f"Expected 'normal' or 'anomaly', got {label!r}."
            )
        if domain not in ("source", "target"):
            raise PreprocessingError(
                f"Could not parse domain from filename {file_name!r}. "
                f"Expected 'source' or 'target', got {domain!r}."
            )
        try:
            section_int = int(section_id)
        except ValueError:
            raise PreprocessingError(
                f"Could not parse section id from filename {file_name!r}: "
                f"{section_id!r} is not an integer."
            )
        if section_int not in range(6):
            raise PreprocessingError(
                f"Section id {section_int} out of expected range [0, 5] "
                f"in filename {file_name!r}."
            )

        canonical_label = "abnormal" if label == "anomaly" else "normal"
        return {
            "label":     canonical_label,
            "entity_id": f"id_{section_id}",   # ← fixed: key is always "entity_id"
            "domain":    domain,
        }

    def preprocess(self, raw_dir: Path, output_dir: Path) -> None:
        """
        Extract and write WAVs to *output_dir*.

        Steps:

        1. For each ``dev_data_{entity_type}.zip`` and ``eval_data_{entity_type}.zip`` in *raw_dir*: extract with
           :mod:`zipfile` into a dedicated subdirectory.
        2. For each extracted WAV, determine the destination label:

           * ``train/*.wav``               → ``normal/``
           * ``source_test/normal_*.wav``  → ``normal/``
           * ``source_test/anomaly_*.wav`` → ``abnormal/``
           * ``target_test/normal_*.wav``  → ``normal/``
           * ``target_test/anomaly_*.wav`` → ``abnormal/``

        3. Mirror to
           ``output_dir/{entity_type}/{entity_id}/{label}/{filename}``.
        4. Files are already mono 16 kHz — no down-mixing or resampling needed.
        5. Delete the extracted subdirectory and ZIP after processing.
        6. Wrap I/O errors in :class:`~download.protocol.PreprocessingError`.

        TODO: implement following the pattern in
        :meth:`~download.mimii.MimiiPreprocessor.preprocess`.  Key
        difference: derive the output label from the source subdirectory
        name and the ``normal_`` / ``anomaly_`` filename prefix rather than
        from noise-level subdirectories.

        :param raw_dir: Directory produced by :class:`MimiiDueDownloader`.
        :param output_dir: Root of the canonical ``audio_data/`` tree.
        :raises PreprocessingError: On extraction or I/O failure.
        :raises NotImplementedError: Until this method is implemented.
        """
        # find all zip files in raw dir matching the canonical patterns
        for entity_type in MIMII_DUE_ZENODO_URLS:
            url_dict = MIMII_DUE_ZENODO_URLS[entity_type]
            zip_names = [f"{prefix}_{entity_type}.zip" for prefix in url_dict]
            for zip_name in zip_names:
                zip_path = raw_dir / zip_name
                if not zip_path.is_file():
                    continue
                # ── extract ───────────────────────────────────────────────
                # Use the zip stem (without ".zip") as the extraction directory
                # to avoid treating the filename as a path component.
                extract_root = raw_dir / Path(zip_name).stem / "extracted"
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_root)
                    print(f"Extracted {zip_name} → {extract_root}")
                except Exception as e:
                    raise PreprocessingError(
                        f"Failed to extract ZIP file: {zip_path}"
                    ) from e

                # ── place WAVs into the canonical layout ──────────────────
                for wav_file in extract_root.glob("**/*.wav"):
                    file_metadata = self._extract_metadata_from_filename(wav_file.name)
                    entity_id = file_metadata["entity_id"]
                    label     = file_metadata["label"]
                    target_path = (
                        output_dir / entity_type / entity_id / label / wav_file.name
                    )
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        wav_file.rename(target_path)
                    except Exception as e:
                        raise PreprocessingError(
                            f"Failed to move WAV file to {target_path}"
                        ) from e

                # ── clean up ──────────────────────────────────────────────
                try:
                    zip_path.unlink()
                    shutil.rmtree(extract_root)
                except Exception as e:
                    raise PreprocessingError(
                        f"Failed to clean up after processing {zip_path}"
                    ) from e

    def is_preprocessed(self, output_dir: Path) -> bool:
        """
        Return ``True`` if the canonical output tree already exists.

        Uses ``output_dir/fan/id_00/normal/`` as a lightweight sentinel —
        if that directory is present the full preprocessing is assumed complete.

        TODO: extend to a more comprehensive check covering all five entity
        types and at least one entity ID each.

        :param output_dir: Root of the canonical ``audio_data/`` tree.
        """
        return (output_dir / "fan" / "id_00" / "normal").is_dir()
