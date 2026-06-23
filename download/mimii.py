"""
MIMII: Malfunctioning Industrial Machine Inspection and Investigation
Zenodo record 3384388: https://zenodo.org/record/3384388

------------------------------------------------
The runner passes a ``tempfile.TemporaryDirectory`` as the destination, so
ZIP archives are stored only for the duration of preprocessing and then
deleted automatically.  The in-memory layout inside that temp dir is::

    <tmp>/
        fan_+6dB.zip
        fan_0dB.zip
        fan_-6dB.zip
        pump_+6dB.zip
        …

Layout after extraction (inside each ZIP)
-----------------------------------------
::

    {machine_type}/
        id_00/
            normal/   *.wav   (8-channel, 16 kHz)
            abnormal/ *.wav
        id_02/ …
        id_04/ …
        id_06/ …

Expected output of :class:`MimiiPreprocessor`
---------------------------------------------
::

    output_dir/
        fan/
            id_00/
                normal/   *.wav   (mono, 16 kHz)
                abnormal/ *.wav
            id_02/ …
        pump/ …

"""
from __future__ import annotations

import itertools
import shutil
import zipfile
from pathlib import Path

import requests
import soundfile as sf

from download.protocol import DownloadError, PreprocessingError  # noqa: F401 — used by implementations
from download.helpers import download_file  # noqa: F401 — used by implementations

# ── Zenodo source URLs ────────────────────────────────────────────────────

#: Nested map of entity_type → noise_level_db → Zenodo download URL.
#: Source: https://zenodo.org/record/3384388
MIMII_ZENODO_URLS: dict[str, dict[int, str]] = {
    "fan": {
         6: "https://zenodo.org/records/3384388/files/6_dB_fan.zip?download=1",
         0: "https://zenodo.org/records/3384388/files/0_dB_fan.zip?download=1",
        -6: "https://zenodo.org/records/3384388/files/-6_dB_fan.zip?download=1",
    },
    "pump": {
        6: "https://zenodo.org/records/3384388/files/6_dB_pump.zip?download=1",
        0: "https://zenodo.org/records/3384388/files/0_dB_pump.zip?download=1",
        -6: "https://zenodo.org/records/3384388/files/-6_dB_pump.zip?download=1",
    },
    "slider": {
        6: "https://zenodo.org/records/3384388/files/6_dB_slider.zip?download=1",
        0: "https://zenodo.org/records/3384388/files/0_dB_slider.zip?download=1",
        -6: "https://zenodo.org/records/3384388/files/-6_dB_slider.zip?download=1",
    },
    "valve": {
         6: "https://zenodo.org/records/3384388/files/6_dB_valve.zip?download=1",
         0: "https://zenodo.org/records/3384388/files/0_dB_valve.zip?download=1",
        -6: "https://zenodo.org/records/3384388/files/-6_dB_valve.zip?download=1",
    },
}

#: All noise levels present in the MIMII dataset (SNR in dB).
MIMII_NOISE_LEVELS_DB: list[int] = [-6, 0, 6]


# ── downloader ────────────────────────────────────────────────────────────


class MimiiDownloader:
    """
    Downloads MIMII ZIP archives from Zenodo record 3384388.

    :param entity_types: Which entity types to download.
        Defaults to all four (fan, pump, slider, valve).
    :type entity_types: list[str] or None
    :param noise_levels_db: Which SNR levels to download (dB).
        Defaults to all three (``[-6, 0, 6]``).  Pass a subset
        (e.g. ``[0]``) to save bandwidth when only one level is needed.
    :type noise_levels_db: list[int] or None
    """

    def __init__(
        self,
        entity_types: list[str] | None = None,
        noise_levels_db: list[int] | None = None,
    ) -> None:
        self.entity_types    = entity_types    or list(MIMII_ZENODO_URLS.keys())
        self.noise_levels_db = noise_levels_db or list(MIMII_NOISE_LEVELS_DB)

    def set_noise_levels_db(self, noise_levels_db: list[int]) -> None:
        self.noise_levels_db = noise_levels_db

    def set_entity_types(self, entity_types: list[str]) -> None:
        self.entity_types = entity_types

    def download(self, destination: Path) -> None:
        """
        Download one ZIP archive per *(entity_type, noise_level_db)* pair
        to *destination*.

        When called from the runner, *destination* is a
        ``tempfile.TemporaryDirectory`` managed by the caller and deleted
        automatically once preprocessing completes, so only the final mono
        WAVs are kept on disk.

        TODO: implement using ``requests`` (or ``urllib.request``) with a
        ``tqdm`` progress bar.  Suggested steps:

        1. ``destination.mkdir(parents=True, exist_ok=True)``
        2. For each ``entity_type`` in :attr:`entity_types` and each
           ``level`` in :attr:`noise_levels_db`, look up
           ``MIMII_ZENODO_URLS[entity_type][level]``.
        3. Stream the response in chunks (e.g. 8 KiB) into
           ``destination / f"{entity_type}_{level:+d}dB.zip"``.
        4. Wrap errors in :class:`~download.protocol.DownloadError`.

        :param destination: Directory to store the raw ZIP files.
            Typically a ``tempfile.TemporaryDirectory`` managed by the caller.
        :raises DownloadError: On HTTP or I/O failure.
        """

        # create directory if it not exists
        destination.mkdir(parents=True, exist_ok=True)

        # loop through the Cartesian product of entities × noise levels
        for entity_type, noise_level_db in itertools.product(
            self.entity_types, self.noise_levels_db
        ):
            download_url = MIMII_ZENODO_URLS[entity_type][noise_level_db]
            dest_path = destination / f"{entity_type}_{noise_level_db:+d}dB.zip"
            try:
                download_file(download_url, dest_path)
            except requests.RequestException as e:
                raise DownloadError(f"Failed to download {download_url}") from e


    def is_downloaded(self, destination: Path) -> bool:
        """
        Return ``True`` if all expected ZIP archives exist in *destination*.

        .. note::
            The runner always passes a fresh ``tempfile.TemporaryDirectory``
            to :meth:`download`, so this method is not called in the normal
            runner flow.  It is kept for callers that manage their own
            persistent download cache.

        ``(et, level)`` pair in :attr:`entity_types` × :attr:`noise_levels_db`.

        :param destination: Raw download directory to inspect.
        """
        for entity_type, noise_level_db in itertools.product(
            self.entity_types, self.noise_levels_db
        ):
            expected_path = destination / f"{entity_type}_{noise_level_db:+d}dB.zip"
            if not expected_path.is_file():
                return False
        return True


# ── preprocessor ─────────────────────────────────────────────────────────


class MimiiPreprocessor:
    """
    Converts extracted MIMII archives into the canonical ``audio_data/`` layout.

    The raw MIMII WAV files are **8-channel, 16 kHz** and are organised by
    noise level inside each ZIP::

        raw_dir/
            fan_+6dB/          ← extracted from fan_6dB.zip
                id_00/
                    normal/    *.wav  (8-channel, 16 kHz)
                    abnormal/  *.wav
                id_02/ …
            fan_0dB/ …
            fan_-6dB/ …
            pump_+6dB/ …

    Preprocessing steps:

    1. Extract each ``*.zip`` found in *raw_dir*.
    2. Walk the ``{entity_type}/{entity_id}/{label}/`` subtrees.
    3. Write the resulting mono WAV to the mirrored path under *output_dir*.

    .. note::
        Down-mixing could be done here **as a storage optimisation only**.
        :func:`data.loader.load_wav` already mixes down to mono at load
        time (``librosa.load(..., mono=True)``), so correctness is not
        affected by channel count.  Converting 8 → 1 channel during
        preprocessing reduces the cached data footprint by ~87.5 %.

    The label subdirectory names (``normal/`` / ``abnormal/``) and entity
    naming (``id_XX``) already match the expected canonical format.
    """

    def preprocess(self, raw_dir: Path, output_dir: Path, down_mix_signal: bool = False) -> None:
        """
        Extract and write WAVs to *output_dir*, optionally mixing down to mono.

        Steps:

        1. For each ``*.zip`` in *raw_dir*: extract with :mod:`zipfile`.
        2. For each extracted WAV:

           a. ``data, sr = soundfile.read(src_path, always_2d=True)``  →  shape ``(T, C)``
           b. If *down_mix_signal* is ``True``, average across the channel
              dimension: ``data = data.mean(axis=1, keepdims=True)``
              This is equivalent to ``librosa.load(..., mono=True)`` and
              gives ≈ √C array gain against uncorrelated microphone noise.
           c. Mirror the relative path under *output_dir* and save with
              ``soundfile.write``.

        3. I/O errors are wrapped in :class:`~download.protocol.PreprocessingError`.

        :param raw_dir: Directory produced by :class:`MimiiDownloader`.
        :param output_dir: Root of the canonical ``audio_data/`` tree.
        :param down_mix_signal: If ``True``, average all channels to mono
            before saving.  Reduces storage by ~87.5 % for 8-channel MIMII
            files.  Defaults to ``False`` because
            :func:`data.loader.load_wav` already handles mono conversion at
            load time.
        :raises PreprocessingError: On extraction or I/O failure.
        """
        
        # find the zip files for each machine type in raw_dir
        all_machine_types = MIMII_ZENODO_URLS.keys()
        for machine_type in all_machine_types:
            for noise_level_db in MIMII_NOISE_LEVELS_DB:
                zip_path = raw_dir / f"{machine_type}_{noise_level_db:+d}dB.zip"
                if not zip_path.is_file():
                    continue

                # Extract into a dedicated subdirectory so cleanup never
                # touches pre-existing content in raw_dir — safe whether
                # raw_dir is a tempfile.TemporaryDirectory or a persistent path.
                extract_root = raw_dir / f"{machine_type}_{noise_level_db:+d}dB_extracted"
                try:
                    extract_root.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_root)
                    print("Extracted", zip_path)
                except Exception as e:
                    raise PreprocessingError(f"Failed to extract {zip_path}") from e

                # mirror every extracted WAV into output_dir
                wav_files = list(extract_root.glob("**/*.wav"))
                print(f"Starting to process {len(wav_files)} files in {extract_root}.")
                for src_path in wav_files:
                    try:
                        # soundfile returns (T, C) and sample rate
                        data, sr = sf.read(src_path, always_2d=True)

                        if down_mix_signal:
                            # mean across microphone channels → (T, 1)
                            # consistent with librosa.load(..., mono=True) and
                            # provides ~√C array gain against uncorrelated noise
                            data = data.mean(axis=1, keepdims=True)

                        # replicate the relative path under output_dir,
                        # rooted at extract_root (not raw_dir) so the path
                        # stays as {machine_type}/{entity_id}/{label}/file.wav
                        rel_path = src_path.relative_to(extract_root)
                        dst_path = output_dir / rel_path
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(dst_path), data, sr)
                    except Exception as e:
                        raise PreprocessingError(f"Failed to process {src_path}") from e

                # Delete only what we created: the ZIP and its dedicated
                # extraction subdirectory.  raw_dir itself is never touched.
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
        # TODO: Make this a more comprehensive check

        :param output_dir: Root of the canonical ``audio_data/`` tree.
        """
        return (output_dir / "fan" / "id_00" / "normal").is_dir()



