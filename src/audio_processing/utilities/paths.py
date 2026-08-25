"""
utilities/paths.py — canonical data-directory constants and path helpers.

Exports
-------
DEFAULT_DATA_DIR
    Platform-appropriate cache root for preprocessed audio data.

    * macOS:   ``~/Library/Caches/audio-processing/audio_data``
    * Linux:   ``~/.cache/audio-processing/audio_data``
    * Windows: ``%LOCALAPPDATA%\\audio-processing\\Cache\\audio_data``

    .. warning::
        On macOS ``~/Library/Caches`` is **not** cleared on reboot — only
        ``/tmp`` (``/private/tmp``) is.  Files written here persist
        indefinitely and are only evicted by the OS under disk pressure if
        registered via the proper cache APIs (which manual writes are not).
        Pass an explicit ``--data-dir`` to keep data in a known location,
        or delete the directory manually when it is no longer needed.

effective_data_dir
    Derive the actual data root by appending a noise-level subdirectory
    (e.g. ``"-6db"``, ``"0db"``, ``"6db"``) when ``noise_level_db`` is
    provided.  Returns the base directory unchanged when it is ``None``,
    which is the correct behaviour for datasets without noise-level variants.

entity_dir
    Resolve the canonical path for one entity under the *effective* data dir.

Usage::

    from audio_processing.utilities.paths import DEFAULT_DATA_DIR, effective_data_dir, entity_dir

    data_dir = effective_data_dir(Path("audio_data/downmixed"), -6)
    # → Path("audio_data/downmixed/-6db")

    data_dir = effective_data_dir(Path("audio_data"), None)
    # → Path("audio_data")   (no suffix appended)

    path = entity_dir(data_dir, "fan", "id_00")
    # → Path("audio_data/downmixed/-6db/fan/id_00")
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_dir


#: Platform-appropriate cache directory used when ``--data-dir`` is not supplied.
DEFAULT_DATA_DIR: Path = Path(user_cache_dir("audio-processing")) / "audio_data"

#: Disk cache for pre-computed mel spectrograms (post signal-transform).
#: Each entry is a ``.npy`` file keyed by a hash of the source file path,
#: its modification time, all feature-extraction params, and the transform
#: config — so stale or mis-matched entries are never served.
DEFAULT_MEL_CACHE_DIR: Path = Path(user_cache_dir("audio-processing")) / "mel_cache"


def effective_data_dir(base: Path, noise_level_db: int | None) -> Path:
    """
    Return the effective data root for the given noise level.

    When *noise_level_db* is ``None`` (e.g. for datasets without noise-level
    variants) the *base* path is returned unchanged.  Otherwise a subdirectory
    named ``"{noise_level_db}db"`` is appended, e.g.::

        effective_data_dir(Path("audio_data/downmixed"), -6)
        # → Path("audio_data/downmixed/-6db")

        effective_data_dir(Path("audio_data/downmixed"), 0)
        # → Path("audio_data/downmixed/0db")

        effective_data_dir(Path("audio_data/downmixed"), 6)
        # → Path("audio_data/downmixed/6db")

        effective_data_dir(Path("audio_data"), None)
        # → Path("audio_data")

    :param base: Base data directory (without any noise-level suffix).
    :param noise_level_db: SNR level in dB, or ``None``.
    :returns: Effective :class:`~pathlib.Path` to use for all data lookups.
    """
    if noise_level_db is None:
        return base
    return base / f"{noise_level_db}db"


def entity_dir(
    data_dir: Path,
    entity_type: str,
    entity_id: str,
) -> Path:
    """
    Resolve the canonical directory for one entity.

    The layout is always::

        data_dir / entity_type / entity_id

    Pass the result of :func:`effective_data_dir` as *data_dir* to ensure
    the noise-level subdirectory is included.

    :param data_dir: Root of the canonical ``audio_data/`` tree
        (typically the result of :func:`effective_data_dir`).
    :param entity_type: Entity type category (e.g. ``"fan"``).
    :param entity_id: Entity instance identifier (e.g. ``"id_00"``).
    :returns: Resolved :class:`~pathlib.Path` for this entity.
    """
    return data_dir / entity_type / entity_id
