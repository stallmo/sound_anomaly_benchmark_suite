"""
tracking/wandb.py — Weights & Biases experiment tracker.

``wandb`` is an optional dependency.  This module can be imported freely;
the ``ImportError`` is only raised when :class:`WandbTracker` is instantiated
without ``wandb`` installed::

    uv add wandb
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class WandbTracker:
    """
    Experiment tracker backed by `Weights & Biases <https://wandb.ai>`_.

    ``wandb`` is imported lazily so that the rest of the project remains
    usable without it installed.

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
        self._wandb.config.update({key: value}, allow_val_change=True)

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

    def finish(self) -> None:
        """
        Mark the W&B run as finished.

        Call once at the end of each experiment when running multiple
        sequential runs in the same process.
        """
        self._wandb.finish()

