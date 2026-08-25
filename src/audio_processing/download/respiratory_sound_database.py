"""
Respiratory Sound Database
Kaggle dataset: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database

------------------------------------------------
Unlike MIMII/MIMII DUE (Zenodo, anonymous HTTP download), this dataset is
hosted on Kaggle and requires authenticated access via the ``kaggle`` package.

The runner passes a ``tempfile.TemporaryDirectory`` as the destination, so the
downloaded archive is stored only for the duration of preprocessing and then
deleted automatically.  The in-memory layout inside that temp dir is::

    <tmp>/
        archive.zip

Layout after extraction (inside the ZIP)
-----------------------------------------
::

    Respiratory_Sound_Database/Respiratory_Sound_Database/
        demographic_info.txt
        patient_diagnosis.csv
        filename_format.txt
        audio_and_txt_files/
            {recording}.wav   # multi-cycle, e.g. 148_1b1_Al_sc_Meditron.wav
            {recording}.txt   # tab-separated: cycle_start_s, cycle_end_s, crackle(0/1), wheeze(0/1)

The equipment used for a recording (the entity ID, equivalent to ``id_00`` in
MIMII) is the token after the last ``_`` in the filename, e.g. ``Meditron``.

Expected output of :class:`RespiratorySoundDatabasePreprocessor`
-------------------------------------------------------------------
::

    output_dir/
        respiratory_sounds/
            AKGC417L/
                normal/   *.wav   (mono, one file per cycle)
                abnormal/ *.wav
            LittC2SE/ …
            Litt3200/ …
            Meditron/ …

"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import soundfile as sf

from audio_processing.download.protocol import DownloadError, PreprocessingError  # noqa: F401 — used by implementations

# ── Kaggle source ─────────────────────────────────────────────────────────

#: Kaggle dataset reference, as used by ``kaggle datasets download -d <ref>``.
KAGGLE_DATASET_REF = "vbookshelf/respiratory-sound-database"

#: Fixed filename the downloaded archive is normalised to, regardless of
#: whatever name the Kaggle API itself gives the file.
ARCHIVE_FILENAME = "archive.zip"

#: Path Kaggle's newer CLI/API versions check for a bearer-style API token,
#: consulted in preference to explicitly-provided credentials.
_KAGGLE_ACCESS_TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
_KAGGLE_JSON_PATH = Path.home() / ".kaggle" / "kaggle.json"

# ── canonical mapping ────────────────────────────────────────────────────

#: Fixed, single entity type for this dataset — there is only one "machine"
#: category (respiratory sounds); the recording equipment plays the role
#: that machine_id/entity_id plays for MIMII (e.g. "id_00").
RESPIRATORY_SOUNDS_ENTITY_TYPE = "respiratory_sounds"

#: Recording equipment types found in the dataset's filenames (the token
#: after the last "_"), used as entity IDs.
RESPIRATORY_SOUNDS_ENTITY_IDS: list[str] = [
    "AKGC417L", "LittC2SE", "Litt3200", "Meditron",
]


# ── downloader ────────────────────────────────────────────────────────────


class RespiratorySoundDatabaseDownloader:
    """
    Downloads the Respiratory Sound Database archive from Kaggle.

    Credential resolution order (checked in :meth:`download`):

    1. Native Kaggle discovery — ``KAGGLE_API_TOKEN`` env var,
       ``~/.kaggle/access_token``, ``KAGGLE_USERNAME``/``KAGGLE_KEY`` env
       vars, or ``~/.kaggle/kaggle.json``. These are recognised directly by
       :meth:`kaggle.api.kaggle_api_extended.KaggleApi.authenticate`.
    2. Only if none of the above are present: explicit credentials passed to
       the constructor or :meth:`set_credentials` (``kaggle_token`` takes
       priority over the ``kaggle_username``/``kaggle_key`` pair), applied as
       environment variables just before authenticating.

    :param kaggle_username: Fallback Kaggle username, used only if no native
        credentials are found.
    :param kaggle_key: Fallback Kaggle API key, paired with *kaggle_username*.
    :param kaggle_token: Fallback Kaggle API token (newer single-token auth),
        used only if no native credentials are found. Takes priority over
        *kaggle_username*/*kaggle_key* when both are supplied.
    """

    def __init__(
        self,
        kaggle_username: str | None = None,
        kaggle_key: str | None = None,
        kaggle_token: str | None = None,
    ) -> None:
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.kaggle_token = kaggle_token

    def set_credentials(
        self,
        *,
        kaggle_username: str | None = None,
        kaggle_key: str | None = None,
        kaggle_token: str | None = None,
    ) -> None:
        """Set fallback Kaggle credentials, used only if native discovery finds nothing."""
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.kaggle_token = kaggle_token

    @staticmethod
    def _has_native_credentials() -> bool:
        """Return ``True`` if Kaggle's own credential discovery has something to find."""
        return (
            bool(os.environ.get("KAGGLE_API_TOKEN"))
            or _KAGGLE_ACCESS_TOKEN_PATH.is_file()
            or bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
            or _KAGGLE_JSON_PATH.is_file()
        )

    def _authenticate(self):
        """
        Authenticate against the Kaggle API, returning a ready-to-use client.

        ``kaggle`` is imported lazily here (rather than at module level)
        because ``kaggle/__init__.py`` authenticates as a side effect of
        the import itself (``api = KaggleApi(); api.authenticate()`` at
        module scope) — which would crash at import time (e.g. when
        ``download/registry.py`` eagerly instantiates this class) if no
        credentials are configured yet. On missing/invalid credentials that
        module-level call does not raise a normal exception — it prints a
        message and calls ``sys.exit(1)`` — so the import itself must be
        wrapped to convert that into a catchable :class:`DownloadError`.

        Credentials must be resolved into environment variables *before*
        this first import, since that import is what triggers Kaggle's own
        authentication check.

        :raises DownloadError: If no credentials are available anywhere, or
            if authentication is rejected by Kaggle.
        """
        if not self._has_native_credentials():
            if self.kaggle_token:
                print(
                    "[RespiratorySoundDatabaseDownloader] No KAGGLE_API_TOKEN / "
                    "~/.kaggle/access_token / KAGGLE_USERNAME+KAGGLE_KEY / "
                    "~/.kaggle/kaggle.json found — falling back to the explicitly "
                    "provided Kaggle API token."
                )
                os.environ["KAGGLE_API_TOKEN"] = self.kaggle_token
            elif self.kaggle_username and self.kaggle_key:
                print(
                    "[RespiratorySoundDatabaseDownloader] No native Kaggle credentials "
                    "found — falling back to the explicitly provided "
                    "kaggle_username/kaggle_key."
                )
                os.environ["KAGGLE_USERNAME"] = self.kaggle_username
                os.environ["KAGGLE_KEY"] = self.kaggle_key
            else:
                raise DownloadError(
                    "No Kaggle credentials found. Set one of: the KAGGLE_API_TOKEN "
                    "env var, ~/.kaggle/access_token, the KAGGLE_USERNAME+KAGGLE_KEY "
                    "env vars, or ~/.kaggle/kaggle.json — or pass kaggle_token / "
                    "kaggle_username+kaggle_key explicitly to "
                    "RespiratorySoundDatabaseDownloader (or its set_credentials method)."
                )

        try:
            import kaggle  # noqa: PLC0415 — see docstring: import must happen after env vars are set
        except SystemExit as e:
            raise DownloadError(
                "Kaggle authentication failed. Check that your KAGGLE_API_TOKEN / "
                "~/.kaggle/access_token / KAGGLE_USERNAME+KAGGLE_KEY / "
                "~/.kaggle/kaggle.json credentials are present and valid."
            ) from e
        except Exception as e:
            raise DownloadError("Failed to import the 'kaggle' package") from e

        return kaggle.api

    def download(self, destination: Path) -> None:
        """
        Download the Respiratory Sound Database archive to *destination*.

        The Kaggle API names the downloaded file after the dataset slug, not
        ``archive.zip`` — the resulting file is renamed to :data:`ARCHIVE_FILENAME`
        so :class:`RespiratorySoundDatabasePreprocessor` has a fixed, predictable
        filename to look for.

        :param destination: Directory to store the raw archive.
            Typically a ``tempfile.TemporaryDirectory`` managed by the caller.
        :raises DownloadError: If authentication or the download itself fails.
        """
        destination.mkdir(parents=True, exist_ok=True)
        api = self._authenticate()

        try:
            api.dataset_download_files(
                KAGGLE_DATASET_REF, path=str(destination), unzip=False
            )
        except Exception as e:
            raise DownloadError(
                f"Failed to download Kaggle dataset '{KAGGLE_DATASET_REF}'"
            ) from e

        downloaded = [
            p for p in destination.glob("*.zip") if p.name != ARCHIVE_FILENAME
        ]
        if not downloaded:
            if (destination / ARCHIVE_FILENAME).is_file():
                return
            raise DownloadError(
                f"Kaggle download for '{KAGGLE_DATASET_REF}' did not produce a "
                f"ZIP archive in {destination}."
            )
        downloaded[0].rename(destination / ARCHIVE_FILENAME)

    def is_downloaded(self, destination: Path) -> bool:
        """
        Return ``True`` if the archive has already been downloaded to *destination*.

        :param destination: Raw download directory to inspect.
        """
        return (destination / ARCHIVE_FILENAME).is_file()


# ── preprocessor ─────────────────────────────────────────────────────────


class RespiratorySoundDatabasePreprocessor:
    """
    Converts the extracted Respiratory Sound Database archive into the
    canonical ``audio_data/`` layout.

    Unlike MIMII/MIMII DUE, each raw WAV file contains multiple breathing
    cycles; the companion ``.txt`` annotation file (one row per cycle:
    ``cycle_start_s``, ``cycle_end_s``, ``crackle_presence``,
    ``wheeze_presence``) is used to slice each raw WAV into one output WAV
    per cycle. A cycle is labelled ``abnormal`` if it contains a wheeze,
    with or without an accompanying crackle; every other combination
    (crackle-only, neither) is ``normal``.

    The recording equipment (the token after the last ``_`` in the filename,
    e.g. ``Meditron``) becomes the entity ID; the entity type is the fixed
    :data:`RESPIRATORY_SOUNDS_ENTITY_TYPE` for every file.
    """

    def preprocess(self, raw_dir: Path, output_dir: Path) -> None:
        """
        Extract ``archive.zip`` and write one sliced, mono WAV per cycle.

        Steps:

        1. Extract :data:`ARCHIVE_FILENAME` from *raw_dir* into a dedicated
           subdirectory.
        2. For each ``*.txt`` annotation file under the extracted
           ``audio_and_txt_files/`` directory, parse its cycle rows and load
           the companion ``.wav`` once.
        3. Slice out each cycle, down-mix to mono, and write it to
           ``output_dir/respiratory_sounds/{equipment}/{normal|abnormal}/{stem}_{cycle_idx:03d}.wav``.
        4. Delete the extracted subdirectory and the ZIP afterward.

        :param raw_dir: Directory produced by :class:`RespiratorySoundDatabaseDownloader`.
        :param output_dir: Root of the canonical ``audio_data/`` tree.
        :raises PreprocessingError: On extraction, parsing, or I/O failure.
        """
        zip_path = raw_dir / ARCHIVE_FILENAME
        if not zip_path.is_file():
            return

        extract_root = raw_dir / "archive_extracted"
        try:
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_root)
        except Exception as e:
            raise PreprocessingError(f"Failed to extract {zip_path}") from e

        try:
            audio_and_txt_dir = next(extract_root.rglob("audio_and_txt_files"))
        except StopIteration as e:
            raise PreprocessingError(
                f"Could not find an 'audio_and_txt_files' directory under {extract_root}"
            ) from e

        txt_files = sorted(audio_and_txt_dir.glob("*.txt"))
        print(f"Starting to process {len(txt_files)} recordings in {audio_and_txt_dir}.")
        for txt_path in txt_files:
            wav_path = txt_path.with_suffix(".wav")
            if not wav_path.is_file():
                raise PreprocessingError(
                    f"Missing companion WAV file for annotation {txt_path}"
                )

            stem, equipment = self._parse_filename(txt_path.stem)
            cycles = self._parse_cycles(txt_path)

            try:
                data, sr = sf.read(wav_path, always_2d=True)
            except Exception as e:
                raise PreprocessingError(f"Failed to read {wav_path}") from e

            if data.shape[1] > 1:
                # mean across channels → mono, consistent with MimiiPreprocessor
                data = data.mean(axis=1, keepdims=True)

            for cycle_idx, (start_s, end_s, _crackle, wheeze) in enumerate(cycles):
                start_sample = round(start_s * sr)
                end_sample = round(end_s * sr)
                cycle_data = data[start_sample:end_sample]

                label = "abnormal" if wheeze == 1 else "normal"
                dst_path = (
                    output_dir
                    / RESPIRATORY_SOUNDS_ENTITY_TYPE
                    / equipment
                    / label
                    / f"{stem}_{cycle_idx:03d}.wav"
                )
                try:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(dst_path), cycle_data, sr)
                except Exception as e:
                    raise PreprocessingError(f"Failed to write {dst_path}") from e

        try:
            zip_path.unlink()
            shutil.rmtree(extract_root)
        except Exception as e:
            raise PreprocessingError(
                f"Failed to clean up after processing {zip_path}"
            ) from e

    @staticmethod
    def _parse_filename(stem_with_equipment: str) -> tuple[str, str]:
        """
        Split a recording's filename stem into ``(stem, equipment)``.

        :param stem_with_equipment: Filename without extension, e.g.
            ``"148_1b1_Al_sc_Meditron"``.
        :returns: ``("148_1b1_Al_sc", "Meditron")``.
        :raises PreprocessingError: If the equipment suffix is not one of
            :data:`RESPIRATORY_SOUNDS_ENTITY_IDS`.
        """
        parts = stem_with_equipment.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in RESPIRATORY_SOUNDS_ENTITY_IDS:
            raise PreprocessingError(
                f"Could not parse recording equipment from filename "
                f"{stem_with_equipment!r}. Expected it to end with one of "
                f"{RESPIRATORY_SOUNDS_ENTITY_IDS}."
            )
        stem, equipment = parts
        return stem, equipment

    @staticmethod
    def _parse_cycles(txt_path: Path) -> list[tuple[float, float, int, int]]:
        """
        Parse a cycle-annotation file into ``(start_s, end_s, crackle, wheeze)`` rows.

        Expected format: tab-separated, one cycle per line, no header::

            0.022   0.364   0   0
            0.364   2.436   0   0

        :param txt_path: Path to the ``.txt`` annotation file.
        :raises PreprocessingError: If a row cannot be parsed.
        """
        cycles = []
        for line_no, line in enumerate(txt_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 4:
                raise PreprocessingError(
                    f"{txt_path}:{line_no}: expected 4 tab-separated fields, "
                    f"got {len(fields)}: {line!r}"
                )
            try:
                start_s, end_s = float(fields[0]), float(fields[1])
                crackle, wheeze = int(fields[2]), int(fields[3])
            except ValueError as e:
                raise PreprocessingError(
                    f"{txt_path}:{line_no}: could not parse cycle row {line!r}"
                ) from e
            cycles.append((start_s, end_s, crackle, wheeze))
        return cycles

    def is_preprocessed(self, output_dir: Path) -> bool:
        """
        Return ``True`` if the canonical output tree already exists.

        Uses ``output_dir/respiratory_sounds/AKGC417L/normal/`` as a
        lightweight sentinel — if that directory is present the full
        preprocessing is assumed complete.

        :param output_dir: Root of the canonical ``audio_data/`` tree.
        """
        return (
            output_dir / RESPIRATORY_SOUNDS_ENTITY_TYPE / "AKGC417L" / "normal"
        ).is_dir()
