"""
inference/frame_scorer.py — per-frame anomaly scoring.

Two concrete scorers are provided:

* :class:`ReconstructionFrameScorer` — MSE between input and reconstruction
  (DCASE baseline).
* :class:`MahalanobisFrameScorer` — Mahalanobis distance in encoder latent
  space.  Requires a fitting step on normal training frames before inference.

Both satisfy the :class:`FrameScorer` structural protocol so they are
interchangeable wherever a ``FrameScorer`` is expected — no inheritance
needed.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader


# ── protocol ──────────────────────────────────────────────────────────────


@runtime_checkable
class FrameScorer(Protocol):
    """
    Structural protocol for per-frame anomaly scorers.

    Any class that implements :meth:`score_batch` satisfies this protocol —
    no inheritance required.  :class:`ReconstructionFrameScorer` and
    :class:`MahalanobisFrameScorer` both conform.
    """

    def score_batch(self, batch: Tensor) -> Tensor:
        """
        Compute a scalar anomaly score for every sample in *batch*.

        :param batch: Input tensor of shape ``(batch_size, input_dim)``.
        :type batch: Tensor
        :returns: 1-D tensor of shape ``(batch_size,)`` with one non-negative
            score per frame.
        :rtype: Tensor
        """
        ...


# ── MSE reconstruction scorer ─────────────────────────────────────────────


class ReconstructionFrameScorer:
    """
    Computes per-frame MSE reconstruction errors using a trained autoencoder.

    The model is set to ``eval()`` mode on construction and never returned
    to training mode by this class.

    :param model: Trained autoencoder.  ``forward(x)`` must return a tuple
        ``(reconstruction, latent)`` — matching
        :class:`~models.autoencoder.Autoencoder`.
    :type model: nn.Module
    :param device: PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    :type device: str
    :param loss_fn: Element-wise reconstruction loss called as
        ``loss_fn(reconstruction, input, reduction="none")``.
        Defaults to :func:`~torch.nn.functional.mse_loss`.
    :type loss_fn: Callable
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        loss_fn: Callable[..., Tensor] = F.mse_loss,
    ) -> None:
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.model.eval()
        self.model.to(self.device)

    def score_batch(self, batch: Tensor) -> Tensor:
        """
        Compute per-sample MSE for every sample in *batch*.

        :param batch: Input tensor ``(batch_size, input_dim)``.
        :type batch: Tensor
        :returns: 1-D tensor ``(batch_size,)`` of MSE values ≥ 0.
        :rtype: Tensor
        """
        batch = batch.to(self.device)
        with torch.no_grad():
            reconstruction, _latent = self.model(batch)
            # reduction="none" → (B, D); mean over feature dim → (B,)
            errors = self.loss_fn(reconstruction, batch, reduction="none").mean(dim=1)
        return errors


# ── Mahalanobis scorer ────────────────────────────────────────────────────


class MahalanobisFrameScorer:
    """
    Scores frames by their Mahalanobis distance in encoder latent space.

    After training, the encoder is run over all normal training frames to
    estimate the latent distribution (sample mean **μ** and precision matrix
    **Σ⁻¹**).  At inference time each frame's Mahalanobis distance from
    that distribution is returned as the anomaly score:

    .. math::

        d(\\mathbf{z}) = \\sqrt{(\\mathbf{z} - \\boldsymbol{\\mu})^{\\top}
                         \\boldsymbol{\\Sigma}^{-1}
                         (\\mathbf{z} - \\boldsymbol{\\mu})}

    Use the :meth:`fit` classmethod to produce a ready-to-use instance.

    :param encoder: Trained encoder sub-network (e.g. ``autoencoder.encoder``).
    :type encoder: nn.Module
    :param mean: Fitted latent mean **μ**, shape ``(bottleneck_dim,)``.
    :type mean: Tensor
    :param precision: Precision matrix **Σ⁻¹**, shape
        ``(bottleneck_dim, bottleneck_dim)``.
    :type precision: Tensor
    :param device: Target compute device.
    :type device: str
    """

    def __init__(
        self,
        encoder: nn.Module,
        mean: Tensor,
        precision: Tensor,
        device: str = "cpu",
    ) -> None:
        self.encoder = encoder
        self.mean = mean.to(device)
        self.precision = precision.to(device)
        self.device = device
        self.encoder.eval()
        self.encoder.to(device)

    @classmethod
    def fit(
        cls,
        encoder: nn.Module,
        normal_loader: DataLoader,
        device: str = "cpu",
    ) -> MahalanobisFrameScorer:
        """
        Estimate the latent distribution from normal training frames.

        Runs the encoder over every batch in *normal_loader*, computes the
        sample mean and inverse covariance (precision matrix), and returns
        a fitted scorer ready for inference.

        :param encoder: Trained encoder sub-network (``autoencoder.encoder``).
        :type encoder: nn.Module
        :param normal_loader: DataLoader of normal training frames,
            shape ``(batch_size, input_dim)`` per batch.
        :type normal_loader: DataLoader
        :param device: Target compute device.
        :type device: str
        :returns: Fitted :class:`MahalanobisFrameScorer`.
        :rtype: MahalanobisFrameScorer
        :raises torch.linalg.LinAlgError: If the sample covariance is
            singular (too few samples or a constant latent dimension).
        """
        encoder = encoder.to(device)
        encoder.eval()
        latents: list[Tensor] = []
        with torch.no_grad():
            for batch in normal_loader:
                # DataLoader with TensorDataset yields (tensor,) tuples;
                # MelFrameDataset yields plain tensors — handle both.
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                latents.append(encoder(x.to(device)))
        z = torch.cat(latents, dim=0)                       # (N, D)
        mean = z.mean(dim=0)                                # (D,)
        z_c = z - mean                                      # (N, D)
        cov = (z_c.T @ z_c) / (z.shape[0] - 1)             # (D, D)
        # Ridge regularisation: keeps the matrix invertible when one or more
        # latent dimensions have (near-)zero variance (e.g. small bottleneck,
        # fresh BatchNorm running stats, or highly correlated features).
        eps = 1e-6
        cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        precision = torch.linalg.inv(cov)                   # (D, D)
        return cls(encoder=encoder, mean=mean, precision=precision, device=device)

    def score_batch(self, batch: Tensor) -> Tensor:
        """
        Compute the Mahalanobis distance for every frame in *batch*.

        :param batch: Input tensor ``(batch_size, input_dim)``.
        :type batch: Tensor
        :returns: 1-D tensor ``(batch_size,)`` of distances ≥ 0.
        :rtype: Tensor
        """
        batch = batch.to(self.device)
        with torch.no_grad():
            z = self.encoder(batch)                         # (B, D)
        z_c = z - self.mean                                 # (B, D)
        distances = torch.sqrt(
            torch.einsum("bi,ij,bj->b", z_c, self.precision, z_c)
        )
        return distances
