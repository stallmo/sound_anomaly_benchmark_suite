"""Tests for training/tracking.py — ExperimentTracker protocol and implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from audio_processing.training.tracking import ExperimentTracker, NullTracker


# ── spy tracker — used across both test modules ───────────────────────────

class SpyTracker:
    """Records every call so tests can assert on what the trainer logged."""

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.metrics: list[tuple[str, float, int | None]] = []
        self.artifacts: list[tuple[str, str | None]] = []

    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.metrics.append((key, value, step))

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        self.artifacts.append((str(path), name))


# ── NullTracker ───────────────────────────────────────────────────────────

class TestNullTracker:

    def test_satisfies_protocol(self):
        assert isinstance(NullTracker(), ExperimentTracker)

    def test_log_param_does_not_raise(self):
        NullTracker().log_param("lr", 1e-3)

    def test_log_metric_does_not_raise(self):
        NullTracker().log_metric("loss", 0.42, step=1)

    def test_log_artifact_does_not_raise(self, tmp_path):
        p = tmp_path / "ckpt.pt"
        p.write_bytes(b"")
        NullTracker().log_artifact(p, name="checkpoint")

    def test_log_metric_step_is_optional(self):
        NullTracker().log_metric("loss", 0.1)   # no step — must not raise


# ── SpyTracker (validates the protocol itself) ────────────────────────────

class TestSpyTrackerSatisfiesProtocol:

    def test_spy_satisfies_protocol(self):
        assert isinstance(SpyTracker(), ExperimentTracker)

    def test_log_param_recorded(self):
        t = SpyTracker()
        t.log_param("epochs", 50)
        assert t.params["epochs"] == 50

    def test_log_metric_recorded(self):
        t = SpyTracker()
        t.log_metric("val_loss", 0.25, step=3)
        assert ("val_loss", 0.25, 3) in t.metrics

    def test_log_artifact_recorded(self, tmp_path):
        t = SpyTracker()
        p = tmp_path / "model.pt"
        t.log_artifact(p, name="best")
        assert t.artifacts[0] == (str(p), "best")


# ── WandbTracker — import-error path only (no live W&B connection) ────────

class TestWandbTrackerImportError:

    def test_raises_import_error_when_wandb_missing(self, monkeypatch):
        """Simulate wandb not being installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "wandb":
                raise ImportError("No module named 'wandb'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from audio_processing.training.tracking import WandbTracker
        with pytest.raises(ImportError, match="wandb is not installed"):
            WandbTracker(project="test")

