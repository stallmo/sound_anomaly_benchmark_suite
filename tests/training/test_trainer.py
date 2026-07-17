"""Tests for training/trainer.py — PtTrainer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from audio_processing.models.autoencoder import Autoencoder
from audio_processing.training.config import TrainingConfig
from audio_processing.training.trainer import PtTrainer
from tests.training.test_tracking import SpyTracker

# ── test-local constants ──────────────────────────────────────────────────
INPUT_DIM = 32
N_SAMPLES = 128
BATCH_SIZE = 16


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def model() -> Autoencoder:
    """Tiny autoencoder — fast enough for unit tests."""
    return Autoencoder(input_dim=INPUT_DIM, hidden_dims=(16,), bottleneck_dim=4)


@pytest.fixture()
def loader() -> DataLoader:
    """Small synthetic DataLoader."""
    data = torch.randn(N_SAMPLES, INPUT_DIM)
    return DataLoader(TensorDataset(data), batch_size=BATCH_SIZE, shuffle=False)


@pytest.fixture()
def config(tmp_path) -> TrainingConfig:
    return TrainingConfig(epochs=3, checkpoint_dir=tmp_path / "ckpt")


@pytest.fixture()
def trainer(model, config) -> PtTrainer:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    return PtTrainer(model, optimizer, config)


# ── train_epoch ───────────────────────────────────────────────────────────

class TestTrainEpoch:

    def test_returns_float(self, trainer, loader):
        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float)

    def test_loss_is_positive(self, trainer, loader):
        assert trainer.train_epoch(loader) > 0.0

    def test_loss_is_finite(self, trainer, loader):
        assert torch.isfinite(torch.tensor(trainer.train_epoch(loader)))

    def test_model_is_in_train_mode_after_call(self, trainer, loader):
        trainer.train_epoch(loader)
        assert trainer.model.training


# ── evaluate ──────────────────────────────────────────────────────────────

class TestEvaluate:

    def test_returns_dict(self, trainer, loader):
        assert isinstance(trainer.evaluate(loader), dict)

    def test_contains_val_loss_key(self, trainer, loader):
        # Run one training step first to initialise BatchNorm running stats
        trainer.train_epoch(loader)
        assert "val_loss" in trainer.evaluate(loader)

    def test_val_loss_is_positive(self, trainer, loader):
        trainer.train_epoch(loader)
        assert trainer.evaluate(loader)["val_loss"] > 0.0

    def test_model_is_in_eval_mode_after_call(self, trainer, loader):
        trainer.train_epoch(loader)
        trainer.evaluate(loader)
        assert not trainer.model.training


# ── train ─────────────────────────────────────────────────────────────────

class TestTrain:

    def test_runs_without_error(self, trainer, loader):
        trainer.train(loader)

    def test_epoch_counter_matches_config_epochs(self, trainer, loader, config):
        trainer.train(loader)
        assert trainer._epoch == config.epochs

    def test_checkpoint_saved(self, trainer, loader, config):
        trainer.train(loader)
        assert (Path(config.checkpoint_dir) / "best.pt").exists()

    def test_train_with_val_loader(self, trainer, loader):
        trainer.train(loader, val_loader=loader)


# ── experiment tracker integration ───────────────────────────────────────

class TestTrackerIntegration:

    def test_log_param_called_for_each_config_field(self, model, config, loader):
        import dataclasses
        spy = SpyTracker()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = PtTrainer(model, optimizer, config, tracker=spy)
        trainer.train(loader)

        for key in dataclasses.asdict(config):
            assert key in spy.params

    def test_train_loss_logged_each_epoch(self, model, config, loader):
        spy = SpyTracker()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = PtTrainer(model, optimizer, config, tracker=spy)
        trainer.train(loader)

        train_loss_steps = [m for m in spy.metrics if m[0] == "train_loss"]
        assert len(train_loss_steps) == config.epochs

    def test_val_loss_logged_when_val_loader_provided(self, model, config, loader):
        spy = SpyTracker()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = PtTrainer(model, optimizer, config, tracker=spy)
        trainer.train(loader, val_loader=loader)

        val_loss_steps = [m for m in spy.metrics if m[0] == "val_loss"]
        assert len(val_loss_steps) == config.epochs

    def test_artifact_logged_on_improvement(self, model, config, loader):
        spy = SpyTracker()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = PtTrainer(model, optimizer, config, tracker=spy)
        trainer.train(loader)

        assert len(spy.artifacts) >= 1


# ── early stopping ────────────────────────────────────────────────────────

class TestEarlyStopping:

    def test_stops_before_max_epochs(self, model, tmp_path, loader, monkeypatch):
        config = TrainingConfig(
            epochs=100,
            early_stopping_patience=3,
            checkpoint_dir=tmp_path / "ckpt",
        )
        optimizer = torch.optim.Adam(model.parameters())
        trainer = PtTrainer(model, optimizer, config)

        # Patch evaluate to return a fixed loss so it never improves after epoch 1
        monkeypatch.setattr(trainer, "evaluate", lambda _: {"val_loss": 999.0})

        trainer.train(loader, val_loader=loader)

        # Epoch 1: 999.0 < inf  → best updated.  Epochs 2-4: no improvement
        # → stopped at epoch 4  (patience = 3)
        assert trainer._epoch == 4
        assert trainer._epoch < config.epochs


# ── checkpoint save / load ────────────────────────────────────────────────

class TestCheckpoint:

    def test_save_creates_file(self, trainer, loader, tmp_path):
        trainer.train_epoch(loader)
        path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(path)
        assert path.exists()

    def test_load_restores_epoch(self, model, config, loader, tmp_path):
        optimizer = torch.optim.Adam(model.parameters())
        trainer = PtTrainer(model, optimizer, config)
        trainer.train(loader)
        epoch_after_train = trainer._epoch

        path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(path)

        # Reset and reload
        trainer._epoch = 0
        trainer.load_checkpoint(path)
        assert trainer._epoch == epoch_after_train

    def test_load_restores_model_weights(self, model, config, loader, tmp_path):
        optimizer = torch.optim.Adam(model.parameters())
        trainer = PtTrainer(model, optimizer, config)
        trainer.train(loader)

        path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(path)

        original_params = {k: v.clone() for k, v in model.state_dict().items()}

        # Scramble weights then restore
        for p in model.parameters():
            p.data.fill_(0.0)

        trainer.load_checkpoint(path)
        for k, v in model.state_dict().items():
            assert torch.allclose(v, original_params[k])

