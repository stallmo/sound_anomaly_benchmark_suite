# Unsupervised Sound Anomaly Detection Benchmark Suite

A benchmark suite for unsupervised sound anomaly detection consisting of:
* Data mapped into a common format
* Common evaluation metrics and principled train-test splits.
* Common baseline model.

---

## Table of Contents

1. [Project structure](#project-structure)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Dataset](#dataset)
5. [Package reference](#package-reference)
6. [Usage](#usage)
7. [Running the tests](#running-the-tests)
8. [Exploration notebooks](#exploration-notebooks)

---

## Project structure

```
audio_processing/
│
├── data/                        # Audio loading, framing, splitting, dataset
│   ├── loader.py                # load_wav, find_wav_files, path parsers
│   ├── framing.py               # frame_signal, count_frames
│   ├── dataset.py               # AudioFrameDataset (PyTorch Dataset)
│   └── splitting.py             # make_train_test_split, DataSplit
│
├── features/                    # Feature extraction
│   ├── log_mel.py               # MelSpectrogramTransform
│   └── hpss.py                  # HpssTransform, hpss_harmonic, hpss_percussive
│
├── models/                      # Neural network architectures + checkpoint I/O
│   ├── autoencoder.py           # Autoencoder (DCASE baseline)
│   └── checkpoint.py            # load_from_checkpoint
│
├── training/                    # Training loop, config, experiment tracking
│   ├── trainer.py               # PtTrainer — loop, checkpointing, early stopping
│   ├── config.py                # TrainingConfig dataclass
│   └── tracking.py              # ExperimentTracker protocol, NullTracker, WandbTracker
│
├── tracking/                    # Canonical tracker package (supersedes training/tracking.py)
│   ├── base.py                  # ExperimentTracker protocol + NullTracker
│   └── wandb.py                 # WandbTracker (optional wandb dependency)
│
├── inference/                   # Anomaly scoring pipeline
│   ├── frame_scorer.py          # ReconstructionFrameScorer, MahalanobisFrameScorer
│   ├── aggregation.py           # mean_score, max_score, percentile_score, AggregationFn
│   ├── file_scorer.py           # FileScorer — frames → aggregated file score
│   ├── detector.py              # AnomalyDetector + AnomalyResult
│   └── threshold.py             # calibrate_threshold — derive threshold from train scores
│
├── audio_data/                  # datasets (not committed to git)
│   └── {machine_type}/          # fan, pump, slider, valve (+ gearbox for MIMII DUE)
│       └── {id}/                # id_00, id_02, … (MIMII) / id_00 … id_05 (MIMII DUE)
│           ├── normal/          # WAV files used for training
│           └── abnormal/        # WAV files used for evaluation only
│
├── exploration/                 # Jupyter notebooks
│   ├── eda.ipynb                # Exploratory data analysis
│   ├── load_data.ipynb          # Data pipeline walkthrough
│   └── sound_anomaly_model.ipynb
│
├── tests/                       # pytest test suite
│   ├── conftest.py              # Shared fixtures (wav_mono, wav_multichannel, mimii_dir)
│   ├── data/
│   │   ├── test_loader.py
│   │   ├── test_framing.py
│   │   ├── test_dataset.py
│   │   └── test_splitting.py
│   ├── features/
│   │   └── test_log_mel.py
│   ├── models/
│   │   └── test_autoencoder.py
│   ├── training/
│   │   ├── test_trainer.py
│   │   └── test_tracking.py
│   └── inference/
│       ├── test_frame_scorer.py
│       ├── test_aggregation.py
│       ├── test_file_scorer.py
│       └── test_detector.py
│
├── main.py                      # redirect stub — use dcase_ae_baseline_runner.py instead
└── pyproject.toml
```

---

## Requirements

- **Python ≥ 3.14**
- **[uv](https://docs.astral.sh/uv/)** — for environment and dependency management

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

**1. Clone the repository**

```bash
git clone <repo-url>
cd audio_processing
```

**2. Create the virtual environment and install all dependencies**

```bash
uv sync
```

This reads `pyproject.toml`, creates `.venv/`, and installs all runtime and dev dependencies in one step.

**3. Activate the environment** *(optional — `uv run` handles this automatically)*

```bash
source .venv/bin/activate
```

**4. Verify the installation**

```bash
uv run pytest --tb=short -q
```

All tests should pass.

---

## Dataset

This project supports two MIMII variants. Run the download script to fetch and preprocess both:

```bash
python download_dataset.py
```

### MIMII

Original MIMII dataset (Malfunctioning Industrial Machine Investigation and Inspection).

**Zenodo**: https://zenodo.org/record/3384388

| Property | Value |
|---|---|
| Machine types | fan, pump, slider, valve |
| Machine IDs | id_00, id_02, id_04, id_06 |
| Noise levels | −6 dB, 0 dB, +6 dB |
| Channels | 8 (mixed down to mono on load) |
| Sample rate | 16 000 Hz |
| Duration per file | ~10 seconds |

### MIMII DUE

MIMII with domain shifts due to changes in operational and environmental conditions.

**Zenodo**: https://zenodo.org/record/4740355

| Property | Value |
|---|---|
| Machine types | fan, gearbox, pump, slider, valve |
| Machine IDs | id_00 – id_05 (test labels only for id_00 – id_02) |
| Channels | 1 (already mono) |
| Sample rate | 16 000 Hz |

Both datasets are placed in the canonical layout after preprocessing:

```
audio_data/
└── {machine_type}/
    └── {id}/
        ├── normal/      ← WAV files used for training
        └── abnormal/    ← WAV files used for evaluation only
```

---

## Package reference

### `data`

#### `load_wav(path, target_samplerate=None, mono=True) → AudioFile1D`

Loads a WAV file and returns an `AudioFile1D` dataclass containing the signal, sample rate, file path, and labels parsed from the MIMII directory structure.

```python
from data import load_wav

audio = load_wav("audio_data/fan/id_00/normal/00000000.wav", target_samplerate=16_000)
# audio.signal      → np.ndarray, shape (n_samples,)
# audio.samplerate  → 16000
# audio.label       → "normal"
# audio.machine_id  → "id_00"
```

#### `find_wav_files(directory) → list[Path]`

Recursively discovers all WAV files under a directory, returned as a sorted list.

```python
from data import find_wav_files

paths = find_wav_files("audio_data/fan/id_00/normal")
```

#### `frame_signal(signal, frame_length, hop_length) → np.ndarray`

Splits a 1-D signal into overlapping frames of shape `(n_frames, frame_length)`.

```python
from data import frame_signal

frames = frame_signal(audio.signal, frame_length=16_000, hop_length=8_000)
# frames.shape → (n_frames, 16000)
```

#### `AudioFrameDataset`

A `torch.utils.data.Dataset` that lazily loads and frames WAV files. Accepts an optional `transform` for feature extraction.

```python
from data import AudioFrameDataset
from features import MelSpectrogramTransform

transform = MelSpectrogramTransform(sample_rate=16_000, n_fft=1024, hop_length=320, n_mels=64)

ds = AudioFrameDataset(
    root_dir="audio_data/fan/id_00/normal",
    frame_length=16_000,
    hop_length=8_000,
    target_samplerate=16_000,
    transform=transform,
)
print(len(ds))      # total frames across all files
print(ds[0].shape)  # (n_mels * time_bins,) = (64 * 51,) = (3264,)
```

It can also be constructed directly from a list of paths (used together with `make_train_test_split`):

```python
ds = AudioFrameDataset.from_file_paths(file_paths, frame_length=16_000, hop_length=8_000)
```

#### `make_train_test_split(root_dir, seed=None) → DataSplit`

Splits a machine ID directory into train and test sets following the MIMII evaluation protocol:

- All **abnormal** files → test set
- An equal number of randomly sampled **normal** files → test set
- Remaining **normal** files → train set

```python
from data import make_train_test_split

split = make_train_test_split("audio_data/fan/id_00", seed=42)

# split.train_paths          → normal files for training
# split.test_paths           → balanced mix of normal + abnormal for evaluation
# split.test_normal_paths    → normal subset of the test set  (label = 0)
# split.test_abnormal_paths  → all abnormal files             (label = 1)
```

---

### `features`

#### `MelSpectrogramTransform`

Converts a 1-D audio frame (`np.ndarray`) into a flattened log-mel spectrogram tensor (`torch.Tensor`) ready for autoencoder input.

```python
from features import MelSpectrogramTransform

transform = MelSpectrogramTransform(
    sample_rate=16_000,
    n_fft=1024,       # STFT window size (64 ms at 16 kHz)
    hop_length=320,   # STFT hop (20 ms)
    n_mels=64,        # number of mel filterbank bins
)

import numpy as np
frame = np.random.randn(16_000).astype(np.float32)
features = transform(frame)
# features.shape → (3264,)  i.e. 64 mels × 51 time bins
```

**Output size formula** (with `center=True`):

```
time_bins  = frame_length // hop_length + 1
n_features = n_mels × time_bins
```

#### `HpssTransform`

Optional pre-processing step that applies Harmonic-Percussive Source Separation (HPSS) via `librosa.effects.hpss` before feature extraction. Keeps either the harmonic or percussive component of the signal.

```python
from features.hpss import HpssTransform, hpss_harmonic, hpss_percussive, make_hpss_transform

# Pre-built convenience callables
clean = hpss_harmonic(signal)    # keep harmonic component
clean = hpss_percussive(signal)  # keep percussive component

# Or construct explicitly
transform = HpssTransform(component="harmonic", margin=1.0)
clean = transform(signal)        # np.ndarray → np.ndarray, same shape
```

`make_hpss_transform(component, margin=1.0)` is a factory that constructs a named `HpssTransform`. `resolve_signal_transform(hpss_component, entity_type)` resolves a CLI/TOML string to a transform, falling back to per-machine-type defaults from `ENTITY_TYPE_COMPONENT`.

---

### `models`

#### `load_from_checkpoint(path, model, device="cpu") → nn.Module`

Restores weights from a checkpoint file written by `PtTrainer` and returns the model in `eval()` mode on the target device.

```python
from models import Autoencoder, load_from_checkpoint

model = Autoencoder(input_dim=3264)
model = load_from_checkpoint("checkpoints/best.pt", model, device="cpu")
# model is now in eval() mode with restored weights
```

---

### `inference`

#### `FrameScorer`

Wraps a trained model and computes a scalar MSE reconstruction error for every frame in a batch. The model is set to `eval()` on construction and never switched back.

```python
from inference import FrameScorer

scorer = FrameScorer(model, device="cpu")
errors = scorer.score_batch(batch)   # Tensor, shape (batch_size,)
```

#### Aggregation functions

Reduce a 1-D tensor of per-frame errors to a single file-level float. Any `Callable[[Tensor], float]` satisfies the `AggregationFn` type alias.

```python
from inference import mean_score, max_score, percentile_score

mean_score(errors)           # arithmetic mean
max_score(errors)            # single most anomalous frame
percentile_score(95)(errors) # 95th-percentile (factory pattern)
```

#### `FileScorer`

Orchestrates the full per-file pipeline: constructs an `AudioFrameDataset` from the file path (using the same framing parameters as training), collects frame errors from `FrameScorer`, and reduces them via an `AggregationFn`.

```python
from inference import FileScorer, mean_score

file_scorer = FileScorer(
    frame_scorer=scorer,
    aggregation_fn=mean_score,
    transform=transform,          # same MelSpectrogramTransform used during training
    frame_length=16_000,
    hop_length=8_000,
    target_samplerate=16_000,
)

score = file_scorer.score_file(path)                     # float
score, frame_errors = file_scorer.score_file(            # (float, Tensor)
    path, return_frame_scores=True
)
scores = file_scorer.score_files(split.test_paths)       # list[float]
```

#### `AnomalyDetector` / `AnomalyResult`

Applies a fixed threshold to file-level scores. A score **strictly greater than** the threshold is flagged as anomalous.

```python
from inference import AnomalyDetector

detector = AnomalyDetector(threshold=0.015)

result  = detector.detect(score, path=path)
# result.score      → float
# result.is_anomaly → bool
# result.path       → Path | None

results = detector.detect_batch(scores, paths=split.test_paths)
```

### End-to-end: build train and test DataLoaders

```python
from torch.utils.data import DataLoader

from data import AudioFrameDataset, make_train_test_split
from features import MelSpectrogramTransform

# 1. Split files
split = make_train_test_split("audio_data/fan/id_00", seed=42)

# 2. Define feature transform
transform = MelSpectrogramTransform(
    sample_rate=16_000, n_fft=1024, hop_length=320, n_mels=64
)

# 3. Build datasets
train_ds = AudioFrameDataset.from_file_paths(
    split.train_paths,
    frame_length=16_000,
    hop_length=8_000,
    target_samplerate=16_000,
    transform=transform,
)
test_ds = AudioFrameDataset.from_file_paths(
    split.test_paths,
    frame_length=16_000,
    hop_length=8_000,
    target_samplerate=16_000,
    transform=transform,
)

# 4. Wrap in DataLoaders
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# 5. Inspect a batch
batch = next(iter(train_loader))
print(batch.shape)   # (64, 3264)
```

### Ground-truth labels for evaluation

`split.test_normal_paths` and `split.test_abnormal_paths` give you the two halves of the test set. Use them to build the label vector for AUC-ROC:

```python
import numpy as np

labels = np.array(
    [0] * len(split.test_normal_paths) +
    [1] * len(split.test_abnormal_paths)
)
```

---

### End-to-end: score test files and detect anomalies

```python
from models import Autoencoder, load_from_checkpoint
from features import MelSpectrogramTransform
from inference import (
    FrameScorer, FileScorer, AnomalyDetector,
    mean_score, percentile_score,
)
from data import make_train_test_split

# 1. Restore the trained model
model = Autoencoder(input_dim=3264)
model = load_from_checkpoint("checkpoints/best.pt", model, device="cpu")

# 2. Build the inference pipeline with the same parameters used during training
transform = MelSpectrogramTransform(
    sample_rate=16_000, n_fft=1024, hop_length=320, n_mels=64
)
frame_scorer = FrameScorer(model, device="cpu")
file_scorer  = FileScorer(
    frame_scorer=frame_scorer,
    aggregation_fn=mean_score,
    transform=transform,
    frame_length=16_000,
    hop_length=8_000,
    target_samplerate=16_000,
)

# 3. Score every test file
split  = make_train_test_split("audio_data/fan/id_00", seed=42)
scores = file_scorer.score_files(split.test_paths)

# 4. Apply a threshold (e.g. 95th percentile of training scores)
detector = AnomalyDetector(threshold=0.015)
results  = detector.detect_batch(scores, paths=split.test_paths)

for r in results:
    print(r.path.name, r.score, "ANOMALY" if r.is_anomaly else "normal")
```

---

## Usage

`dcase_ae_baseline_runner.py` is the primary CLI entry point. It trains and evaluates the AE baseline on one or more machine entity IDs, downloading data automatically if it is not already present.

```bash
# Single entity, all defaults (10 epochs, DCASE baseline config):
uv run python dcase_ae_baseline_runner.py mimii id_00

# Multiple entities, fan machine type:
uv run python dcase_ae_baseline_runner.py mimii id_00,id_02,id_04 --entity-type fan

# Custom hyperparameters, named W&B run:
uv run python dcase_ae_baseline_runner.py mimii id_00 \
    --experiment-name my-sweep \
    --epochs 50 --lr 5e-4 --bottleneck-dim 32

# See all options:
uv run python dcase_ae_baseline_runner.py --help
```

---

## Running the tests

```bash
# All tests
uv run pytest

# Single module
uv run pytest tests/data/test_loader.py

# With detailed output
uv run pytest -v
```

The test suite uses only synthetic in-memory WAV files — the `audio_data/` directory is never required to run tests.

| Test file | Covers |
|---|---|
| `tests/data/test_loader.py` | `load_wav`, `find_wav_files`, path parsers |
| `tests/data/test_framing.py` | `frame_signal`, `count_frames` |
| `tests/data/test_dataset.py` | `AudioFrameDataset.__getitem__`, `DataLoader` compatibility |
| `tests/data/test_splitting.py` | `make_train_test_split`, `AudioFrameDataset.from_file_paths` |
| `tests/features/test_log_mel.py` | `MelSpectrogramTransform` output shape, dtype, value range |
| `tests/models/test_autoencoder.py` | `Autoencoder` forward pass, output shapes, configurability |
| `tests/training/test_trainer.py` | `PtTrainer` loop, checkpointing, early stopping, tracker integration |
| `tests/training/test_tracking.py` | `ExperimentTracker` protocol, `NullTracker` |
| `tests/inference/test_frame_scorer.py` | `ReconstructionFrameScorer`, `MahalanobisFrameScorer` — output shape, eval mode, no-grad |
| `tests/inference/test_aggregation.py` | `mean_score`, `max_score`, `percentile_score` correctness |
| `tests/inference/test_file_scorer.py` | `FileScorer.score_file` (incl. `return_frame_scores`), `score_files` |
| `tests/inference/test_detector.py` | `AnomalyDetector` threshold logic, `detect_batch`, `AnomalyResult` |

---

## Exploration notebooks

Located in `exploration/`. Connect them to the project environment by registering the kernel:

```bash
uv run python -m ipykernel install --user --name audio-processing
```

Then select **audio-processing** as the kernel inside JupyterLab / VS Code.

| Notebook | Contents |
|---|---|
| `eda.ipynb` | Waveform visualisation, Fourier transforms, mel spectrograms |
| `load_data.ipynb` | Full data pipeline walkthrough using `AudioFrameDataset` |
| `sound_anomaly_model.ipynb` | Autoencoder training and anomaly scoring (in progress) |

