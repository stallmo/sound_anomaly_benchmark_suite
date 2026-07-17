"""
inference/aggregation.py — frame-score → file-score aggregation strategies.

All functions operate on a 1-D :class:`~torch.Tensor` of per-frame
reconstruction errors and return a single ``float`` file-level score.

The ``AggregationFn`` type alias defines the expected signature so that
:class:`~inference.file_scorer.FileScorer` can accept any of the built-in
functions or a custom callable without inheritance.

Built-in strategies
-------------------
mean_score          : arithmetic mean of all frame errors
max_score           : maximum frame error (most anomalous frame)
percentile_score(p) : factory that returns the *p*-th percentile aggregator
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

# Any callable with this signature can be used as an aggregation strategy.
AggregationFn = Callable[[Tensor], float]


def mean_score(errors: Tensor) -> float:
    """
    Aggregate frame errors by arithmetic mean.

    :param errors: 1-D tensor of per-frame reconstruction errors, shape
        ``(n_frames,)``.
    :type errors: Tensor
    :returns: Mean reconstruction error across all frames.
    :rtype: float
    """

    return torch.mean(errors).item()


def max_score(errors: Tensor) -> float:
    """
    Aggregate frame errors by taking the maximum value.

    Sensitive to the single most anomalous frame; useful when anomalies
    are brief and would be diluted by a mean.

    :param errors: 1-D tensor of per-frame reconstruction errors, shape
        ``(n_frames,)``.
    :type errors: Tensor
    :returns: Maximum reconstruction error across all frames.
    :rtype: float
    """
    return torch.max(errors).item()

def percentile_score(p: float) -> AggregationFn:
    """
    Return an aggregation function that computes the *p*-th percentile.

    A factory so that ``percentile_score(95)`` can be passed wherever an
    :data:`AggregationFn` is expected, for example::

        scorer = FileScorer(..., aggregation_fn=percentile_score(95))

    :param p: Percentile in the range ``[0, 100]``.
    :type p: float
    :returns: A callable ``(errors: Tensor) -> float`` that returns the
        *p*-th percentile of *errors*.
    :rtype: AggregationFn
    :raises ValueError: If *p* is outside ``[0, 100]``.
    """

    if not (0 <= p <= 100):
        raise ValueError(f"Percentile p must be in [0, 100], got {p}")

    def aggregator(errors: Tensor) -> float:
        return torch.quantile(errors, p / 100).item()

    return aggregator

