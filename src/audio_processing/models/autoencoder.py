"""
models/autoencoder.py — Dense autoencoder baseline for unsupervised
sound anomaly detection.

:Reference:
    Architecture adapted from the DCASE baseline:

    Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi,
    "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,"
    arXiv preprint arXiv:1909.09347, 2019.


:Differences from the original:
    - Hidden layer dimensions are configurable via ``hidden_dims``
      (default reproduces the original four 128-unit layers).
    - Bottleneck dimension is configurable via ``bottleneck_dim``
      (default reproduces the original 8 units).
    - BatchNorm ``momentum`` and ``eps`` are exposed as constructor arguments.
    - The unused ``cov_source`` / ``cov_target`` buffer parameters are omitted.
"""

from __future__ import annotations

import torch
from torch import nn


def _dense_block(
    in_features: int,
    out_features: int,
    bn_momentum: float,
    bn_eps: float,
) -> nn.Sequential:
    """Linear → BatchNorm1d → ReLU building block shared by encoder and decoder."""
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=bn_momentum, eps=bn_eps),
        nn.ReLU(),
    )


class Autoencoder(nn.Module):
    """
    Fully-connected autoencoder for unsupervised sound anomaly detection.

    The encoder compresses a log-mel spectrogram feature vector to a
    low-dimensional bottleneck; the decoder reconstructs the original vector.
    Anomaly scores at inference time are derived from the per-frame
    reconstruction error (e.g. MSE between input and reconstruction).

    **Architecture** (default parameters reproduce the DCASE baseline)::

        Encoder:  input_dim → [128, 128, 128, 128] → bottleneck_dim
        Decoder:  bottleneck_dim → [128, 128, 128, 128] → input_dim

    Every hidden layer uses the pattern ``Linear → BatchNorm1d → ReLU``.
    The final decoder layer is a plain ``Linear`` with no activation so that
    the output range is unconstrained.

    :param input_dim: Size of the input feature vector
        (= ``n_mels × time_bins`` from
        :class:`~features.log_mel.MelSpectrogramTransform`).
    :type input_dim: int
    :param hidden_dims: Width of each hidden layer in the encoder. The decoder
        mirrors these in reverse order.
        Default: ``(128, 128, 128, 128)`` — matches the DCASE baseline.
    :type hidden_dims: tuple[int, ...] or list[int]
    :param bottleneck_dim: Dimensionality of the latent representation.
        Default: ``8`` — matches the DCASE baseline.
    :type bottleneck_dim: int
    :param bn_momentum: ``momentum`` argument passed to every
        :class:`~torch.nn.BatchNorm1d` layer.
        Default: ``0.01`` — matches the DCASE baseline.
    :type bn_momentum: float
    :param bn_eps: ``eps`` argument passed to every
        :class:`~torch.nn.BatchNorm1d` layer.
        Default: ``1e-3`` — matches the DCASE baseline.
    :type bn_eps: float

    .. note::
        :class:`~torch.nn.BatchNorm1d` requires a batch size ≥ 2
        **in training mode**. Call ``model.eval()`` before scoring individual
        samples at inference time; this also freezes the running statistics.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128, 128, 128),
        bottleneck_dim: int = 8,
        bn_momentum: float = 0.01,
        bn_eps: float = 1e-3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.bottleneck_dim = bottleneck_dim

        # ── encoder ──────────────────────────────────────────────────────
        enc_layers: list[nn.Module] = []
        in_features = input_dim
        for out_features in self.hidden_dims:
            enc_layers.append(_dense_block(in_features, out_features, bn_momentum, bn_eps))
            in_features = out_features
        enc_layers.append(_dense_block(in_features, bottleneck_dim, bn_momentum, bn_eps))
        self.encoder = nn.Sequential(*enc_layers)

        # ── decoder ──────────────────────────────────────────────────────
        dec_layers: list[nn.Module] = []
        in_features = bottleneck_dim
        for out_features in reversed(self.hidden_dims):
            dec_layers.append(_dense_block(in_features, out_features, bn_momentum, bn_eps))
            in_features = out_features
        # Final layer: no BatchNorm or activation — unconstrained output
        dec_layers.append(nn.Linear(in_features, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode then decode *x*.

        :param x: Input tensor of shape ``(batch_size, input_dim)``.
        :type x: torch.Tensor
        :returns: A tuple ``(reconstruction, latent)`` where:

            - ``reconstruction`` — shape ``(batch_size, input_dim)``
            - ``latent``         — shape ``(batch_size, bottleneck_dim)``
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        latent = self.encoder(x.view(-1, self.input_dim))
        reconstruction = self.decoder(latent)
        return reconstruction, latent

