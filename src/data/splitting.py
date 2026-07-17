"""
data/splitting.py — train / test split utilities for the MIMII dataset.

The split follows three rules:
    1. All abnormal files  →  test set.
    2. A random sample of normal files equal in size to the abnormal count
       →  test set  (keeps the test set balanced).
    3. Remaining normal files  →  train set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from data.loader import find_wav_files, parse_label_from_path


@dataclass
class DataSplit:
    """
    Result of :func:`make_train_test_split`.

    Attributes
    ----------
    train_paths : list[Path]
        Normal files reserved for unsupervised training.
    test_normal_paths : list[Path]
        Normal files included in the test set (count == len(test_abnormal_paths)).
    test_abnormal_paths : list[Path]
        All abnormal files.
    """

    train_paths: list[Path]
    test_normal_paths: list[Path]
    test_abnormal_paths: list[Path]

    @property
    def test_paths(self) -> list[Path]:
        """All test files (normal + abnormal), sorted."""
        return sorted(self.test_normal_paths + self.test_abnormal_paths)


def make_train_test_split(
    root_dir: str | Path,
    seed: int | None = None,
) -> DataSplit:
    """
    Split WAV files under *root_dir* into train and test sets.

    Rules
    -----
    * All abnormal files go into the test set.
    * An equal number of normal files are randomly sampled into the test set.
    * The remaining normal files form the train set.

    If there are no abnormal files the test set is empty and every normal
    file goes to the train set (useful for pure training runs).

    :param root_dir: Root directory to search for WAV files (recursive).
    :param seed: Random seed for reproducible sampling. None means no fixed seed.
    :return: A :class:`DataSplit` with train / test path lists.
    :raises ValueError: If abnormal files outnumber normal files, making a
                        balanced test set impossible.
    """
    all_files = find_wav_files(root_dir)

    normal_files = sorted(p for p in all_files if parse_label_from_path(p) == "normal")
    abnormal_files = sorted(p for p in all_files if parse_label_from_path(p) == "abnormal")

    n_abnormal = len(abnormal_files)

    if n_abnormal > len(normal_files):
        raise ValueError(
            f"Cannot build a balanced test set: {n_abnormal} abnormal file(s) "
            f"but only {len(normal_files)} normal file(s) available."
        )

    rng = random.Random(seed)
    test_normal = sorted(rng.sample(normal_files, k=n_abnormal))
    test_normal_set = set(test_normal)
    train_normal = sorted(p for p in normal_files if p not in test_normal_set)

    return DataSplit(
        train_paths=train_normal,
        test_normal_paths=test_normal,
        test_abnormal_paths=abnormal_files,
    )


def make_combined_split(
    root_dirs: list[str | Path],
    seed: int | None = None,
) -> DataSplit:
    """
    Build a single :class:`DataSplit` from multiple entity directories.

    Each directory is split independently (preserving per-entity balance),
    then the three path lists are merged and returned as one ``DataSplit``.
    Using a per-entity balance strategy prevents a dominant entity from
    crowding out another entity's abnormal files in the test set.

    :param root_dirs: Sequence of root directories, one per entity.
    :param seed: Random seed forwarded to each :func:`make_train_test_split`
                 call so every entity's split is reproducible.
    :return: A merged :class:`DataSplit`.
    :raises ValueError: If any individual directory raises during splitting.
    """
    train_paths: list[Path] = []
    test_normal_paths: list[Path] = []
    test_abnormal_paths: list[Path] = []

    for root_dir in root_dirs:
        split = make_train_test_split(root_dir, seed=seed)
        train_paths.extend(split.train_paths)
        test_normal_paths.extend(split.test_normal_paths)
        test_abnormal_paths.extend(split.test_abnormal_paths)

    return DataSplit(
        train_paths=sorted(train_paths),
        test_normal_paths=sorted(test_normal_paths),
        test_abnormal_paths=sorted(test_abnormal_paths),
    )

