"""
download/registry.py — central registry mapping dataset names to DatasetInfo.

Adding a new dataset
--------------------
1. Create a sibling module (e.g. ``download/my_dataset.py``) with concrete
   ``MyDownloader`` and ``MyPreprocessor`` classes that satisfy the
   :class:`~download.protocol.DatasetDownloader` and
   :class:`~download.protocol.DatasetPreprocessor` protocols.
2. Instantiate a :class:`~download.protocol.DatasetInfo` record below.
3. Add it to :data:`DATASET_REGISTRY` with a short slug as the key.

Usage::

    from download.registry import DATASET_REGISTRY, get_dataset_info

    info = get_dataset_info("mimii")
    if not info.downloader.is_downloaded(raw_dir):
        info.downloader.download(raw_dir)
    info.preprocessor.preprocess(raw_dir, output_dir)
"""

from __future__ import annotations

from download.mimii import MimiiDownloader, MimiiPreprocessor
from download.mimii_due import MimiiDueDownloader, MimiiDuePreprocessor
from download.protocol import DatasetInfo
from download.respiratory_sound_database import (
    RESPIRATORY_SOUNDS_ENTITY_IDS,
    RESPIRATORY_SOUNDS_ENTITY_TYPE,
    RespiratorySoundDatabaseDownloader,
    RespiratorySoundDatabasePreprocessor,
)


#: All registered datasets, keyed by their CLI slug.

DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "mimii": DatasetInfo(
        name="mimii",
        description=(
            "MIMII: Malfunctioning Industrial Machine Investigation and Inspection. "
            "Fan, pump, slider and valve recordings at 16 kHz (Zenodo record 3384388)."
        ),
        entity_types=["fan", "pump", "slider", "valve"],
        entity_ids=["id_00", "id_02", "id_04", "id_06"],
        downloader=MimiiDownloader(),
        preprocessor=MimiiPreprocessor(),
        metadata={"noise_levels_db": [-6, 0, 6]},
    ),
    "mimii_due": DatasetInfo(
        name="mimii_due",
        description=("MIMII DUE: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection with "
                     "Domain Shifts due to Changes in Operational and Environmental Conditions. "
                     "Fan, gearbox, pump, slider and valve recordings at 16 kHz (Zenodo record 4740355)."),
        entity_types=["fan", "gearbox","pump", "slider", "valve"],
        entity_ids=["id_00", "id_01", "id_02", "id_03", "id_04", "id_05"],
        downloader=MimiiDueDownloader(),
        preprocessor=MimiiDuePreprocessor(),
    ),
    "respiratory_sound_database": DatasetInfo(
        name="respiratory_sound_database",
        description=(
            "Respiratory Sound Database: breathing-cycle recordings labelled by "
            "crackle/wheeze presence (Kaggle, vbookshelf/respiratory-sound-database). "
            "A cycle is abnormal if it contains a wheeze, with or without a crackle."
        ),
        entity_types=[RESPIRATORY_SOUNDS_ENTITY_TYPE],
        entity_ids=RESPIRATORY_SOUNDS_ENTITY_IDS,
        downloader=RespiratorySoundDatabaseDownloader(),
        preprocessor=RespiratorySoundDatabasePreprocessor(),
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    """
    Look up a dataset by its registry slug.

    :param name: Dataset slug, e.g. ``"mimii"``.
    :type name: str
    :returns: The corresponding :class:`~download.protocol.DatasetInfo`.
    :rtype: DatasetInfo
    :raises KeyError: If *name* is not registered.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Registered datasets: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name]

