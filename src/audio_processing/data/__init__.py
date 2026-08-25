"""
data package — audio loading, framing, dataset, and splitting utilities.

Public API
----------
AudioFile1D               : dataclass holding a loaded audio signal and its metadata
load_wav                  : load a single WAV file from disk
find_wav_files            : recursively discover WAV files under a directory
parse_label_from_path     : infer "normal" / "abnormal" from path structure
parse_machine_id_from_path: infer machine id (e.g. "id_00") from path structure
frame_signal              : split a 1-D signal into overlapping frames
count_frames              : calculate number of frames without materialising them
MelFrameDataset           : PyTorch Dataset — DCASE baseline mel context-window framing
AudioFrameDataset         : PyTorch Dataset — raw-audio framing with per-frame transform
DataSplit                 : dataclass holding the result of a train/test split
make_train_test_split     : split a directory of WAV files into train and test sets
make_combined_split       : merge per-entity splits across multiple directories
"""

from audio_processing.data.loader import (
    AudioFile1D,
    find_wav_files,
    load_wav,
    parse_label_from_path,
    parse_machine_id_from_path,
)
from audio_processing.data.framing import count_frames, frame_signal
from audio_processing.data.dataset import AudioFrameDataset, MelFrameDataset
from audio_processing.data.splitting import DataSplit, make_combined_split, make_train_test_split

__all__ = [
    "AudioFile1D",
    "load_wav",
    "find_wav_files",
    "parse_label_from_path",
    "parse_machine_id_from_path",
    "frame_signal",
    "count_frames",
    "MelFrameDataset",
    "AudioFrameDataset",
    "DataSplit",
    "make_train_test_split",
    "make_combined_split",
]
