"""
training/tracking.py — experiment tracker protocol and built-in implementations.

The :class:`ExperimentTracker` is a structural protocol (``typing.Protocol``).
Any object that exposes ``log_param``, ``log_metric``, and ``log_artifact``
satisfies it — no inheritance required.  This keeps third-party tracking
libraries (W&B, MLflow, TensorBoard …) entirely optional and swappable.

Built-in implementations
------------------------
:class:`NullTracker`
    Silent no-op.  Used as the default when the caller does not supply a
    tracker, so :class:`~training.trainer.PtTrainer` never has to guard
    against ``None``.

:class:`WandbTracker`
    Thin wrapper around the ``wandb`` Python SDK.  ``wandb`` is an optional
    dependency; importing this module does **not** require it to be installed.
    The ``ImportError`` is raised only when :class:`WandbTracker` is
    *instantiated*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ── protocol ──────────────────────────────────────────────────────────────

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


# ── built-in implementations ──────────────────────────────────────────────

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


class WandbTracker:
    """
    Experiment tracker backed by `Weights & Biases <https://wandb.ai>`_.

    ``wandb`` is imported lazily so that the rest of the training package
    remains usable without it installed.  Install it with::

        uv add wandb

    :param project: W&B project name.
    :type project: str
    :param run_name: Optional display name for this run.
    :type run_name: str or None
    :param config: Initial config dict logged to the W&B run.  Typically
        obtained from ``dataclasses.asdict(training_config)``.
    :type config: dict or None

    :raises ImportError: If ``wandb`` is not installed.
    """

    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        config: dict | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed.  Run:  uv add wandb"
            ) from exc

        self._wandb = wandb
        wandb.init(project=project, name=run_name, config=config or {})

    def log_param(self, key: str, value: Any) -> None:
        """
        Update the W&B run config with a single key-value pair.

        :param key: Parameter name.
        :type key: str
        :param value: Parameter value.
        :type value: Any
        """
        self._wandb.config.update({key: value})

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """
        Log a scalar metric to the W&B run.

        :param key: Metric name.
        :type key: str
        :param value: Scalar value.
        :type value: float
        :param step: Training step used as the x-axis.
        :type step: int or None
        """
        self._wandb.log({key: value}, step=step)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        """
        Upload a file as a W&B artifact of type ``"model"``.

        :param path: Local path to the file.
        :type path: str or Path
        :param name: Artifact name.  Defaults to the file stem.
        :type name: str or None
        """
        artifact = self._wandb.Artifact(
            name=name or Path(path).stem, type="model"
        )
        artifact.add_file(str(path))
        self._wandb.log_artifact(artifact)

