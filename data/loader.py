from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import librosa


@dataclass
class AudioFile1D:
    """Container for a loaded audio file and its metadata."""

    signal: np.ndarray  # shape: (n_samples,) — always 1-D (mono)
    samplerate: int
    path: Path
    label: str | None  # "normal" or "abnormal", parsed from directory structure
    machine_id: str | None  # e.g. "id_00", parsed from directory structure


def load_wav(
        path: str | Path,
        target_samplerate: int | None = None,
        mono: bool = True,
) -> AudioFile1D:
    """
    Load a WAV file from disk and return an AudioFile1D.

    Multi-channel files (e.g., MIMII 8-channel recordings) are mixed down
    to mono when mono=True. If target_samplerate is given the signal is
    resampled to that rate; otherwise the native sample rate is kept.

    :param path: Path to the WAV file.
    :param target_samplerate: Resample to this rate if given; None keeps the native rate.
    :param mono: If True, mix all channels down to a single channel.
    :return: AudioFile1D with signal, samplerate, path, label, and machine_id populated.
    """
    audio_signal, samplerate = librosa.load(path, sr=target_samplerate, mono=mono)
    machine_id = parse_machine_id_from_path(path)
    label = parse_label_from_path(path)

    audio_file = AudioFile1D(signal=audio_signal, samplerate=samplerate, path=path, machine_id=machine_id, label=label)
    return audio_file


def find_wav_files(directory: str | Path) -> list[Path]:
    """
    Recursively find all WAV files under *directory*.

    :param directory: Root directory to search.
    :return: Sorted list of absolute paths to all discovered WAV files.
    :raises FileNotFoundError: If *directory* does not exist.
    """
    path_dir = Path(directory)
    if not path_dir.exists():
        raise FileNotFoundError(f"Directory not found: {path_dir}")
    files = path_dir.glob("**/*.wav")
    files_absolute_list = sorted([f.absolute() for f in files])
    return files_absolute_list

def parse_label_from_path(path: str | Path) -> str | None:
    """
    Infer the class label from the file path by inspecting parent directory names.

    Expected MIMII structure:  .../fan/id_00/normal/00000010.wav

    :param path: Path to a WAV file.
    :return: "normal", "abnormal", or None if the structure is unrecognised.
    """

    label = Path(path).parent.name
    if label in ["abnormal", "normal"]:
        return label
    else:
        return None

def parse_machine_id_from_path(path: str | Path) -> str | None:
    """
    Infer the machine ID (e.g. "id_00") from the file path.

    Expected MIMII structure:  .../fan/id_00/normal/00000010.wav

    :param path: Path to a WAV file.
    :return: Machine ID string (e.g. "id_00") or None if unrecognised.
    """
    machine_id = Path(path).parent.parent.name
    if machine_id.startswith("id_"):
        return machine_id
    else:
        return None
