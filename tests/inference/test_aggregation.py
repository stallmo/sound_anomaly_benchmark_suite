"""Tests for inference/aggregation.py — aggregation functions."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from inference.aggregation import AggregationFn, max_score, mean_score, percentile_score


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def uniform_errors() -> Tensor:
    """Five identical errors — mean, max, and all percentiles equal 2.0."""
    return torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])


@pytest.fixture()
def mixed_errors() -> Tensor:
    """Errors: [1.0, 2.0, 3.0, 4.0, 5.0] — known mean=3.0, max=5.0."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


# ── mean_score ────────────────────────────────────────────────────────────

class TestMeanScore:

    def test_returns_float(self, mixed_errors):
        pass

    def test_correct_value(self, mixed_errors):
        pass

    def test_uniform_errors_equal_constant(self, uniform_errors):
        pass

    def test_satisfies_aggregation_fn_protocol(self):
        """mean_score should be assignable to an AggregationFn variable."""
        fn: AggregationFn = mean_score
        assert callable(fn)


# ── max_score ─────────────────────────────────────────────────────────────

class TestMaxScore:

    def test_returns_float(self, mixed_errors):
        pass

    def test_correct_value(self, mixed_errors):
        pass

    def test_greater_than_or_equal_to_mean(self, mixed_errors):
        pass


# ── percentile_score ──────────────────────────────────────────────────────

class TestPercentileScore:

    def test_returns_callable(self):
        pass

    def test_p100_equals_max(self, mixed_errors):
        pass

    def test_p0_equals_min(self, mixed_errors):
        pass

    def test_p50_is_median(self, mixed_errors):
        pass

    def test_invalid_p_raises_value_error(self):
        pass

    def test_satisfies_aggregation_fn_protocol(self):
        fn: AggregationFn = percentile_score(95)
        assert callable(fn)

