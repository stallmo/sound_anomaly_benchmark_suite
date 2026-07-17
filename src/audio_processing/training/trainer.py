"""
training/trainer.py — lightweight PyTorch training loop.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from audio_processing.training.config import TrainingConfig
from audio_processing.training.tracking import ExperimentTracker, NullTracker


class PtTrainer:
    """
    Lightweight, framework-free PyTorch trainer.

    Encapsulates the training loop, validation, checkpointing, early stopping,
    and experiment tracking behind a small, readable API.  All heavy lifting
    remains visible inside :meth:`train_epoch` and :meth:`evaluate` — there
    is no magic lifecycle or hidden hook system.

    **Injectable dependencies** — ``model``, ``optimizer``, and ``tracker``
    are all passed in rather than constructed internally.  This makes the
    trainer straightforward to test (swap in a tiny model and a
    :class:`~torch.utils.data.TensorDataset`) and straightforward to extend
    (swap in a different tracker without touching this class).

    :param model: The model to train.  ``forward(x)`` must return either a
        plain :class:`~torch.Tensor` or a tuple whose first element is the
        reconstruction tensor.
    :type model: nn.Module
    :param optimizer: Pre-constructed optimiser.  Use
        :attr:`~training.config.TrainingConfig.learning_rate` and
        :attr:`~training.config.TrainingConfig.weight_decay` when building it.
    :type optimizer: torch.optim.Optimizer
    :param config: Training hyperparameters.
    :type config: TrainingConfig
    :param loss_fn: Reconstruction loss function called as
        ``loss_fn(reconstruction, target)``.  Defaults to
        :func:`~torch.nn.functional.mse_loss`.
    :type loss_fn: Callable[[Tensor, Tensor], Tensor]
    :param tracker: Experiment tracking back-end.  Defaults to
        :class:`~training.tracking.NullTracker` (silent no-op).
    :type tracker: ExperimentTracker or None

    .. note::
        The model is moved to ``config.device`` during ``__init__``.
        Construct the optimiser *after* moving the model, or pass it in
        already built for the correct device.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_fn = loss_fn
        self.tracker: ExperimentTracker = tracker if tracker is not None else NullTracker()

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Mutable training state — inspectable from tests and callbacks
        self._epoch: int = 0
        self._best_val_loss: float = float("inf")
        self._epochs_without_improvement: int = 0

    # ── public API ────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        """
        Run the full training loop for up to ``config.epochs`` epochs.

        Logs all config fields as params once, then logs ``train_loss``
        (and ``val_loss`` when a validation loader is provided) after every
        epoch.  Saves a ``best.pt`` checkpoint whenever the monitored metric
        improves and uploads it as a tracker artifact.

        Early stopping is applied when
        :attr:`~training.config.TrainingConfig.early_stopping_patience` is set
        and the monitored metric fails to improve for that many epochs.

        :param train_loader: DataLoader for the training set.
        :type train_loader: DataLoader
        :param val_loader: Optional DataLoader for the validation set.
            When ``None``, training loss is used for checkpointing and early
            stopping.
        :type val_loader: DataLoader or None
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Log every config field once at the start of the run
        for key, value in dataclasses.asdict(self.config).items():
            self.tracker.log_param(key, str(value))

        for epoch in range(1, self.config.epochs + 1):
            self._epoch = epoch

            train_loss = self.train_epoch(train_loader)
            self.tracker.log_metric("train_loss", train_loss, step=epoch)

            val_metrics: dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                for key, value in val_metrics.items():
                    self.tracker.log_metric(key, value, step=epoch)

            # Checkpoint whenever the monitored metric improves
            monitored = val_metrics.get("val_loss", train_loss)
            if monitored < self._best_val_loss:
                self._best_val_loss = monitored
                self._epochs_without_improvement = 0
                best_path = checkpoint_dir / "best.pt"
                self.save_checkpoint(best_path)
                # self.tracker.log_artifact(best_path, name="best_checkpoint")
            else:
                self._epochs_without_improvement += 1

            if (
                self.config.early_stopping_patience is not None
                and self._epochs_without_improvement
                >= self.config.early_stopping_patience
            ):
                break

    def train_epoch(self, loader: DataLoader) -> float:
        """
        Run one full pass over *loader* in training mode.

        :param loader: DataLoader for the training set.
        :type loader: DataLoader
        :returns: Mean batch loss over the epoch.
        :rtype: float
        """
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            # DataLoader wraps TensorDataset output in a list — unwrap it
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)

            output = self.model(batch)
            reconstruction = output[0] if isinstance(output, tuple) else output

            loss = self.loss_fn(reconstruction, batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """
        Evaluate the model on *loader* without updating parameters.

        :param loader: DataLoader for the validation or test set.
        :type loader: DataLoader
        :returns: Dictionary containing ``"val_loss"`` (mean MSE over all
            batches).
        :rtype: dict[str, float]
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device)

                output = self.model(batch)
                reconstruction = output[0] if isinstance(output, tuple) else output
                total_loss += self.loss_fn(reconstruction, batch).item()

        return {"val_loss": total_loss / len(loader)}

    def save_checkpoint(self, path: str | Path) -> None:
        """
        Persist model weights, optimiser state, and trainer state to *path*.

        :param path: Destination file path (typically ``*.pt``).
        :type path: str or Path
        """
        torch.save(
            {
                "epoch": self._epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self._best_val_loss,
            },
            Path(path),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """
        Restore model weights, optimiser state, and trainer state from *path*.

        :param path: Path to a checkpoint saved by :meth:`save_checkpoint`.
        :type path: str or Path
        """
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epoch = checkpoint.get("epoch", 0)
        self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))

