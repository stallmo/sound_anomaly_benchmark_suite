"""
utilities/data.py — data-availability checking and download orchestration.

Provides :func:`ensure_data`, which verifies that every requested entity
directory is present under the canonical ``audio_data/`` tree and, if any
are missing, delegates to the registered downloader and preprocessor.

Usage::

    from utilities.data import ensure_data
    from utilities.paths import DEFAULT_DATA_DIR

    ensure_data(
        dataset_name="mimii",
        entity_type="fan",
        entity_ids=["id_00", "id_02"],
        data_dir=DEFAULT_DATA_DIR,
        noise_level_db=-6,
    )
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

from download.registry import DATASET_REGISTRY
from utilities.paths import effective_data_dir, entity_dir


def ensure_data(
    dataset_name: str,
    entity_type: str,
    entity_ids: list[str],
    data_dir: Path,
    noise_level_db: int | None = None,
) -> Path:
    """
    Check that every requested entity directory is present under *data_dir*.

    *data_dir* should be the **base** directory (without any noise-level
    suffix).  The noise-level subdirectory (e.g. ``"-6db"``) is derived
    automatically from *noise_level_db* via :func:`~utilities.paths.effective_data_dir`
    and appended exactly once, so there is no risk of double-application.

    If any directories are missing the function delegates to the dataset's
    registered :class:`~download.protocol.DatasetDownloader` and
    :class:`~download.protocol.DatasetPreprocessor`.  Raw archives are
    downloaded into a :class:`~tempfile.TemporaryDirectory` and are deleted
    automatically once preprocessing is done, keeping disk usage minimal.

    When *noise_level_db* is provided but the dataset does not support noise
    levels (i.e. ``metadata["noise_levels_db"]`` is absent), a
    :class:`UserWarning` is issued and the value is ignored so the runner can
    continue with the base directory.  When the dataset does support noise
    levels but *noise_level_db* is not among them, a :class:`ValueError` is
    raised immediately — this is likely a misconfiguration.

    :param dataset_name: Key in :data:`~download.registry.DATASET_REGISTRY`
        (e.g. ``"mimii"``).
    :param entity_type: Entity type category (e.g. ``"fan"``).
    :param entity_ids: Entity IDs to verify (e.g. ``["id_00", "id_02"]``).
    :param data_dir: Base data directory **without** any noise-level suffix.
    :param noise_level_db: Optional SNR level in dB.  When provided and
        supported by the dataset, a subdirectory named ``"{noise_level_db}db"``
        is appended to *data_dir*.  Ignored with a warning when the dataset
        does not support noise levels.
    :raises KeyError: If *dataset_name* is not in :data:`~download.registry.DATASET_REGISTRY`.
    :raises ValueError: If the dataset supports noise levels but *noise_level_db*
        is not among the supported values.
    :returns: The resolved effective data directory (with noise-level suffix
        applied when appropriate).  Use this as the canonical ``data_dir``
        for all downstream path lookups.
    :rtype: Path
    """
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Registered: {sorted(DATASET_REGISTRY)}"
        )

    info = DATASET_REGISTRY[dataset_name]

    # Determine the effective noise level to apply to the path.
    # Rules:
    #   1. Dataset has no noise_levels_db metadata → noise levels not applicable;
    #      issue a UserWarning and ignore noise_level_db (no suffix appended).
    #   2. Dataset supports noise levels but requested level is not listed →
    #      raise ValueError — this is a misconfiguration.
    #   3. Dataset supports noise levels and the level is listed → apply suffix.
    available_levels: list[int] | None = info.metadata.get("noise_levels_db")
    effective_noise_level = noise_level_db

    if noise_level_db is not None:
        if available_levels is None:
            warnings.warn(
                f"noise_level_db={noise_level_db} was provided but "
                f"'{dataset_name}' does not support noise levels. "
                f"The noise level will be ignored and the base data directory "
                f"'{data_dir}' will be used as-is.",
                UserWarning,
                stacklevel=2,
            )
            effective_noise_level = None
        elif noise_level_db not in available_levels:
            raise ValueError(
                f"noise_level_db={noise_level_db} is not available for "
                f"'{dataset_name}'. Supported levels: {available_levels}."
            )

    # Resolve the effective directory (appends noise-level suffix if applicable).
    resolved_dir = effective_data_dir(data_dir, effective_noise_level)

    missing = [
        eid for eid in entity_ids
        if not entity_dir(resolved_dir, entity_type, eid).exists()
    ]
    if not missing:
        return resolved_dir

    print(
        f"\nMissing data for {missing} — fetching '{dataset_name}' "
        f"with noise level {effective_noise_level} and entity type {entity_type} …"
    )

    # only set noise level if it is available for the dataset
    if effective_noise_level is not None:
        info.downloader.set_noise_levels_db([effective_noise_level])
        info.downloader.set_entity_types([entity_type])

    with tempfile.TemporaryDirectory() as _tmp:
        raw_dir = Path(_tmp)
        print("  Downloading to temporary directory …")
        info.downloader.download(raw_dir)
        print(f"  Preprocessing → {resolved_dir} …")
        info.preprocessor.preprocess(raw_dir, resolved_dir)
    # raw_dir and all ZIP archives are deleted automatically here

    return resolved_dir
