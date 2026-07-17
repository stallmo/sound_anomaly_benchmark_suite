"""
evaluation/labels.py — ground-truth label extraction from MIMII file paths.

Labels are encoded in the directory structure:

    audio_data/fan/id_00/normal/xxxxx.wav    →  0
    audio_data/fan/id_00/abnormal/xxxxx.wav  →  1

:func:`extract_labels` is a thin wrapper around
:func:`~data.loader.parse_label_from_path` that converts the ``"normal"`` /
``"abnormal"`` strings to the binary integers expected by scikit-learn metrics.

It is its own module so that a label vector can be built from any path list
(e.g. ``split.test_paths``) without touching scoring or metrics logic.
"""

from __future__ import annotations

from pathlib import Path

from data.loader import parse_label_from_path

# Canonical mapping from directory-name label to integer
LABEL_MAP: dict[str, int] = {"normal": 0, "abnormal": 1}


def extract_labels(paths: list[Path]) -> list[int]:
    """
    Convert a list of WAV file paths to binary integer labels.

    Labels are inferred from the parent directory name following the MIMII
    convention: ``normal/`` → ``0``, ``abnormal/`` → ``1``.

    :param paths: Ordered list of WAV file paths.  Order is preserved in the
        returned label list, so it must match the order of the score list
        passed to :func:`~evaluation.metrics.compute_metrics`.
    :type paths: list[Path]
    :returns: List of integers, one per path: ``0`` for normal,
        ``1`` for abnormal.
    :rtype: list[int]
    :raises ValueError: If any path has an unrecognised parent directory name
        (i.e. neither ``"normal"`` nor ``"abnormal"``).
    """
    labels: list[int] = []
    for path in paths:
        raw = parse_label_from_path(path)
        if raw not in LABEL_MAP:
            raise ValueError(
                f"Cannot determine label for path '{path}'. "
                f"Expected parent directory to be 'normal' or 'abnormal', "
                f"got '{Path(path).parent.name}'."
            )
        labels.append(LABEL_MAP[raw])
    return labels

