"""Tests for inference/frame_scorer.py — ReconstructionFrameScorer and MahalanobisFrameScorer."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoder import Autoencoder
from inference.frame_scorer import FrameScorer, ReconstructionFrameScorer, MahalanobisFrameScorer

# ── test-local constants ──────────────────────────────────────────────────
INPUT_DIM = 32
BATCH_SIZE = 8
BOTTLENECK_DIM = 4


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def model() -> Autoencoder:
    """Tiny autoencoder — fast enough for unit tests."""
    return Autoencoder(input_dim=INPUT_DIM, hidden_dims=(16,), bottleneck_dim=BOTTLENECK_DIM)


@pytest.fixture()
def scorer(model) -> ReconstructionFrameScorer:
    return ReconstructionFrameScorer(model, device="cpu")


@pytest.fixture()
def batch() -> Tensor:
    return torch.randn(BATCH_SIZE, INPUT_DIM)


@pytest.fixture()
def normal_loader() -> DataLoader:
    """Small DataLoader of normal frames for Mahalanobis fitting."""
    data = torch.randn(64, INPUT_DIM)
    return DataLoader(TensorDataset(data), batch_size=16)


# ── protocol conformance ───────────────────────────────────────────────────

class TestFrameScorerProtocol:

    def test_reconstruction_scorer_satisfies_protocol(self, scorer):
        assert isinstance(scorer, FrameScorer)

    def test_mahalanobis_scorer_satisfies_protocol(self, model, normal_loader):
        ms = MahalanobisFrameScorer.fit(model.encoder, normal_loader)
        assert isinstance(ms, FrameScorer)


# ── ReconstructionFrameScorer construction ────────────────────────────────

class TestFrameScorerConstruction:

    def test_model_set_to_eval_on_init(self, scorer):
        assert not scorer.model.training

    def test_model_moved_to_device(self, scorer):
        assert all(p.device.type == "cpu" for p in scorer.model.parameters())


# ── ReconstructionFrameScorer.score_batch ─────────────────────────────────

class TestScoreBatch:

    def test_returns_tensor(self, scorer, batch):
        assert isinstance(scorer.score_batch(batch), Tensor)

    def test_output_shape_is_batch_size(self, scorer, batch):
        assert scorer.score_batch(batch).shape == (BATCH_SIZE,)

    def test_errors_are_non_negative(self, scorer, batch):
        assert (scorer.score_batch(batch) >= 0).all()

    def test_errors_are_finite(self, scorer, batch):
        assert torch.isfinite(scorer.score_batch(batch)).all()

    def test_does_not_update_model_weights(self, scorer, batch):
        params_before = [p.clone() for p in scorer.model.parameters()]
        scorer.score_batch(batch)
        for before, after in zip(params_before, scorer.model.parameters()):
            torch.testing.assert_close(before, after)

    def test_model_remains_in_eval_mode_after_call(self, scorer, batch):
        scorer.score_batch(batch)
        assert not scorer.model.training


# ── MahalanobisFrameScorer.fit ────────────────────────────────────────────

class TestMahalanobisFrameScorerFit:

    def test_fit_returns_mahalanobis_scorer(self, model, normal_loader):
        ms = MahalanobisFrameScorer.fit(model.encoder, normal_loader)
        assert isinstance(ms, MahalanobisFrameScorer)

    def test_mean_shape(self, model, normal_loader):
        ms = MahalanobisFrameScorer.fit(model.encoder, normal_loader)
        assert ms.mean.shape == (BOTTLENECK_DIM,)

    def test_precision_shape(self, model, normal_loader):
        ms = MahalanobisFrameScorer.fit(model.encoder, normal_loader)
        assert ms.precision.shape == (BOTTLENECK_DIM, BOTTLENECK_DIM)


# ── MahalanobisFrameScorer.score_batch ────────────────────────────────────

class TestMahalanobisScoreBatch:

    def test_output_shape_is_batch_size(self, model, normal_loader, batch):
        ms = MahalanobisFrameScorer.fit(model.encoder, normal_loader)
        assert ms.score_batch(batch).shape == (BATCH_SIZE,)

    def test_distances_are_non_negative(self, model, normal_loader, batch):
        ms = MahalanobisFrameScorer.fit(model.encoder, normal_loader)
        assert (ms.score_batch(batch) >= 0).all()

    def test_distances_are_finite(self, model, normal_loader, batch):
        ms = MahalanobisFrameScorer.fit(model.encoder, normal_loader)
        assert torch.isfinite(ms.score_batch(batch)).all()
