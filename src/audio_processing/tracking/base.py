"""
tracking/base.py — experiment tracker protocol and no-op implementation.

Contains no optional dependencies; safe to import anywhere in the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExperimentTracker(Protocol):
    """
    Structural protocol for experiment tracking back-ends.

    Any object that implements the three methods below satisfies this
    protocol without inheriting from it.  Pass a concrete implementation
    to :class:`~training.trainer.PtTrainer` at construction time.

    :Example:

    .. code-block:: python

        tracker = WandbTracker(project="sound-anomaly")
        trainer = PtTrainer(model, optimizer, config, tracker=tracker)
    """

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a single scalar or string hyperparameter.

        Called once per run, typically at the start of training, to record
        configuration values such as learning rate, batch size, or model
        architecture details.

        :param key: Parameter name.
        :type key: str
        :param value: Parameter value.  Must be serialisable by the backend.
        :type value: Any
        """
        ...

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """
        Log a scalar metric value, optionally associated with a training step.

        Called after each epoch to record losses, accuracy, AUC, etc.

        :param key: Metric name (e.g. ``"train_loss"``, ``"val_loss"``).
        :type key: str
        :param value: Scalar metric value.
        :type value: float
        :param step: Global step or epoch number used as the x-axis in plots.
            ``None`` lets the back-end assign a step automatically.
        :type step: int or None
        """
        ...

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        """
        Upload or register a file artifact (e.g. a model checkpoint).

        Called whenever a new best checkpoint is saved.

        :param path: Path to the file to upload.
        :type path: str or Path
        :param name: Human-readable artifact name.  Defaults to the file stem.
        :type name: str or None
        """
        ...


class NullTracker:
    """
    Silent no-op tracker.

    Used as the default inside :class:`~training.trainer.PtTrainer` so the
    trainer never has to branch on ``tracker is None``.  All calls are
    accepted and silently discarded.
    """

    def log_param(self, key: str, value: Any) -> None:  # noqa: D102
        pass

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:  # noqa: D102
        pass

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:  # noqa: D102
        pass

