"""Tests for models/autoencoder.py — Autoencoder."""

from __future__ import annotations

import pytest
import torch

from models.autoencoder import Autoencoder

# ── test-local constants ──────────────────────────────────────────────────
INPUT_DIM = 128           # arbitrary feature vector size
BATCH_SIZE = 32
DEFAULT_HIDDEN = (128, 128, 128, 128)
DEFAULT_BOTTLENECK = 8


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def model() -> Autoencoder:
    """Autoencoder with default (DCASE baseline) parameters."""
    return Autoencoder(input_dim=INPUT_DIM)


@pytest.fixture()
def batch() -> torch.Tensor:
    """Random input batch of shape (BATCH_SIZE, INPUT_DIM)."""
    return torch.randn(BATCH_SIZE, INPUT_DIM)


# ── basic contract ────────────────────────────────────────────────────────

class TestAutoencoderContract:

    def test_is_nn_module(self, model):
        assert isinstance(model, torch.nn.Module)

    def test_forward_returns_tuple_of_two_tensors(self, model, batch):
        output = model(batch)
        assert isinstance(output, tuple) and len(output) == 2

    def test_reconstruction_shape(self, model, batch):
        reconstruction, _ = model(batch)
        assert reconstruction.shape == (BATCH_SIZE, INPUT_DIM)

    def test_latent_shape(self, model, batch):
        _, latent = model(batch)
        assert latent.shape == (BATCH_SIZE, DEFAULT_BOTTLENECK)

    def test_reconstruction_dtype_matches_input(self, model, batch):
        reconstruction, _ = model(batch)
        assert reconstruction.dtype == batch.dtype

    def test_has_trainable_parameters(self, model):
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) > 0


# ── default parameters reproduce the DCASE baseline ─────────────────

class TestDefaultParameters:

    def test_default_hidden_dims(self, model):
        assert model.hidden_dims == DEFAULT_HIDDEN

    def test_default_bottleneck_dim(self, model):
        assert model.bottleneck_dim == DEFAULT_BOTTLENECK

    def test_default_input_dim_stored(self, model):
        assert model.input_dim == INPUT_DIM


# ── configurability ───────────────────────────────────────────────────────

class TestConfigurability:

    @pytest.mark.parametrize("bottleneck_dim", [2, 8, 32, 64])
    def test_configurable_bottleneck_dim(self, bottleneck_dim):
        model = Autoencoder(input_dim=INPUT_DIM, bottleneck_dim=bottleneck_dim)
        _, latent = model(torch.randn(BATCH_SIZE, INPUT_DIM))
        assert latent.shape == (BATCH_SIZE, bottleneck_dim)

    @pytest.mark.parametrize("hidden_dims", [
        (64,),
        (256, 128),
        (512, 256, 128, 64),
    ])
    def test_configurable_hidden_dims(self, hidden_dims):
        model = Autoencoder(input_dim=INPUT_DIM, hidden_dims=hidden_dims)
        reconstruction, _ = model(torch.randn(BATCH_SIZE, INPUT_DIM))
        assert reconstruction.shape == (BATCH_SIZE, INPUT_DIM)

    def test_configurable_bn_momentum(self):
        # Should not raise — just verifies construction succeeds
        model = Autoencoder(input_dim=INPUT_DIM, bn_momentum=0.1)
        model(torch.randn(BATCH_SIZE, INPUT_DIM))

    def test_configurable_bn_eps(self):
        model = Autoencoder(input_dim=INPUT_DIM, bn_eps=1e-5)
        model(torch.randn(BATCH_SIZE, INPUT_DIM))

    def test_decoder_mirrors_encoder_hidden_dims(self):
        """For any hidden_dims, reconstruction shape must equal input shape."""
        for hidden_dims in [(32,), (64, 32), (256, 128, 64)]:
            model = Autoencoder(input_dim=INPUT_DIM, hidden_dims=hidden_dims)
            reconstruction, _ = model(torch.randn(BATCH_SIZE, INPUT_DIM))
            assert reconstruction.shape == (BATCH_SIZE, INPUT_DIM)


# ── encoder / decoder structure ───────────────────────────────────────────

class TestArchitectureStructure:

    def test_encoder_is_sequential(self, model):
        assert isinstance(model.encoder, torch.nn.Sequential)

    def test_decoder_is_sequential(self, model):
        assert isinstance(model.decoder, torch.nn.Sequential)

    def test_encoder_block_count(self, model):
        # One block per hidden layer + one block for the bottleneck
        assert len(model.encoder) == len(DEFAULT_HIDDEN) + 1

    def test_decoder_block_count(self, model):
        # One block per hidden layer + one final Linear (no activation)
        assert len(model.decoder) == len(DEFAULT_HIDDEN) + 1

    def test_final_decoder_layer_is_linear(self, model):
        assert isinstance(model.decoder[-1], torch.nn.Linear)

    def test_final_decoder_layer_has_no_batchnorm(self, model):
        # The last element must be a plain Linear, not a Sequential block
        assert not isinstance(model.decoder[-1], torch.nn.Sequential)


# ── eval-mode inference (batch size = 1) ─────────────────────────────────

class TestEvalMode:

    def test_batch_size_1_in_eval_mode(self, model):
        """BatchNorm1d works with a single sample when running stats are available."""
        # Run one training-mode forward pass to initialise running statistics
        model(torch.randn(BATCH_SIZE, INPUT_DIM))
        model.eval()
        reconstruction, latent = model(torch.randn(1, INPUT_DIM))
        assert reconstruction.shape == (1, INPUT_DIM)
        assert latent.shape == (1, DEFAULT_BOTTLENECK)

    def test_batch_size_1_raises_in_train_mode(self, model):
        """BatchNorm1d requires ≥ 2 samples during training."""
        with pytest.raises(ValueError):
            model(torch.randn(1, INPUT_DIM))

