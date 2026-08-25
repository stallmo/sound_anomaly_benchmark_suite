"""tests/features/test_hpss.py — unit tests for features.hpss."""

from __future__ import annotations

import numpy as np
import pytest

from audio_processing.features.hpss import (
    ENTITY_TYPE_COMPONENT,
    HpssTransform,
    SignalTransform,
    hpss_harmonic,
    hpss_percussive,
    make_hpss_transform,
    resolve_signal_transform,
)

# ── helpers ───────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
DURATION_S  = 0.5
N_SAMPLES   = int(SAMPLE_RATE * DURATION_S)


def _sine_signal(freq: float = 440.0) -> np.ndarray:
    """Return a short float32 sine wave."""
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# ── HpssTransform ─────────────────────────────────────────────────────────

class TestHpssTransform:
    def test_harmonic_output_shape(self):
        signal = _sine_signal()
        transform = HpssTransform(component="harmonic")
        out = transform(signal)
        assert out.shape == signal.shape

    def test_percussive_output_shape(self):
        signal = _sine_signal()
        transform = HpssTransform(component="percussive")
        out = transform(signal)
        assert out.shape == signal.shape

    def test_invalid_component_raises(self):
        with pytest.raises(ValueError, match="component must be one of"):
            HpssTransform(component="invalid")

    def test_harmonic_and_percussive_differ(self):
        """Harmonic and percussive outputs should not be identical."""
        signal = _sine_signal()
        h = HpssTransform(component="harmonic")(signal)
        p = HpssTransform(component="percussive")(signal)
        assert not np.allclose(h, p)

    def test_to_config_keys(self):
        cfg = HpssTransform(component="percussive", margin=2.0).to_config()
        assert cfg["signal_transform"] == "hpss"
        assert cfg["component"] == "percussive"
        assert cfg["margin"] == 2.0

    def test_to_config_harmonic(self):
        cfg = HpssTransform(component="harmonic").to_config()
        assert cfg["component"] == "harmonic"

    def test_margin_stored(self):
        t = HpssTransform(component="harmonic", margin=3.0)
        assert t.margin == 3.0

    def test_output_dtype_float(self):
        signal = _sine_signal()
        out = HpssTransform(component="harmonic")(signal)
        assert np.issubdtype(out.dtype, np.floating)


# ── module-level callables ────────────────────────────────────────────────

class TestModuleLevelCallables:
    def test_hpss_harmonic_shape(self):
        signal = _sine_signal()
        out = hpss_harmonic(signal)
        assert out.shape == signal.shape

    def test_hpss_percussive_shape(self):
        signal = _sine_signal()
        out = hpss_percussive(signal)
        assert out.shape == signal.shape

    def test_hpss_harmonic_is_harmonic_component(self):
        signal = _sine_signal()
        expected = HpssTransform(component="harmonic")(signal)
        np.testing.assert_array_equal(hpss_harmonic(signal), expected)

    def test_hpss_percussive_is_percussive_component(self):
        signal = _sine_signal()
        expected = HpssTransform(component="percussive")(signal)
        np.testing.assert_array_equal(hpss_percussive(signal), expected)


# ── make_hpss_transform ───────────────────────────────────────────────────

class TestMakeHpssTransform:
    def test_returns_hpss_transform(self):
        t = make_hpss_transform("harmonic")
        assert isinstance(t, HpssTransform)
        assert t.component == "harmonic"

    def test_invalid_component(self):
        with pytest.raises(ValueError):
            make_hpss_transform("noise")


# ── ENTITY_TYPE_COMPONENT ─────────────────────────────────────────────────

class TestEntityTypeComponent:
    def test_known_entity_types_present(self):
        for et in ("fan", "pump", "valve", "slider"):
            assert et in ENTITY_TYPE_COMPONENT

    def test_values_are_valid_components(self):
        for component in ENTITY_TYPE_COMPONENT.values():
            assert component in ("harmonic", "percussive")


# ── resolve_signal_transform ──────────────────────────────────────────────

class TestResolveSignalTransform:
    def test_explicit_harmonic(self):
        t = resolve_signal_transform("harmonic", entity_type="pump")
        assert isinstance(t, HpssTransform)
        assert t.component == "harmonic"

    def test_explicit_percussive(self):
        t = resolve_signal_transform("percussive", entity_type="fan")
        assert isinstance(t, HpssTransform)
        assert t.component == "percussive"

    def test_none_falls_back_to_entity_type(self):
        t = resolve_signal_transform(None, entity_type="fan")
        assert isinstance(t, HpssTransform)
        assert t.component == ENTITY_TYPE_COMPONENT["fan"]

    def test_empty_string_falls_back_to_entity_type(self):
        t = resolve_signal_transform("", entity_type="pump")
        assert isinstance(t, HpssTransform)
        assert t.component == ENTITY_TYPE_COMPONENT["pump"]

    def test_none_entity_type_returns_none(self):
        t = resolve_signal_transform(None, entity_type=None)
        assert t is None

    def test_none_with_unknown_entity_type_returns_none(self):
        t = resolve_signal_transform(None, entity_type="unknown_machine")
        assert t is None

    def test_explicit_none_string_returns_none(self):
        t = resolve_signal_transform("none", entity_type="fan")
        assert t is None

