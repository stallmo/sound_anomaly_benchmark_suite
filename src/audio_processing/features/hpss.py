"""
features/hpss.py — Harmonic-Percussive Source Separation (HPSS) transforms.

Provides a :class:`HpssTransform` callable that separates a raw audio signal
into its harmonic and percussive components using
:func:`librosa.effects.hpss`, then returns the requested component.

Typical use
-----------
>>> from audio_processing.features.hpss import HpssTransform, hpss_harmonic, ENTITY_TYPE_COMPONENT
>>> transform = HpssTransform(component="harmonic")
>>> clean_signal = transform(signal)           # np.ndarray → np.ndarray

Entity-type defaults
--------------------
:data:`ENTITY_TYPE_COMPONENT` maps MIMII entity types to the component that
has empirically proven most useful for anomaly detection.  Use
:func:`make_hpss_transform` to construct the appropriate transform from an
entity type name, with an optional override.
"""

from __future__ import annotations

from typing import Callable

import librosa
import numpy as np

# ── type alias ───────────────────────────────────────────────────────────

SignalTransform = Callable[[np.ndarray], np.ndarray]

# ── entity-type → component defaults ────────────────────────────────────

ENTITY_TYPE_COMPONENT: dict[str, str] = {
    "fan":    "harmonic", # Assumption (from listening): might work well
    "pump":   "percussive", # Assumption (from listening): might work well
    "valve":  "percussive", # Assumption (from listening): might work well
    "slider": "percussive", # Assumption (from listening): might not work well
}
"""
Default HPSS component per MIMII entity type.

Override at the CLI with ``--hpss-component`` or in ``pyproject.toml`` via
the ``hpss_component`` key.

Rationale
~~~~~~~~~
* Harmonic sounds makes us hear melodies and chords
* Percussive sounds make us hear clicks, claps, knocks, etc.
* **Fan** — rotating-blade hum sits in stable harmonic partials; the
  harmonic component isolates the tonal signature that changes when a fan
  is faulty.
* **Pump / Valve / Slider** — operation is dominated by impact / click
  events (plunger strokes, gear contacts) whose energy is broadband and
  transient — i.e., percussive.
"""

_VALID_COMPONENTS = frozenset({"harmonic", "percussive"})


# ── transform class ───────────────────────────────────────────────────────


class HpssTransform:
    """
    Callable signal transform that applies HPSS and returns one component.

    :param component: Which component to keep — ``"harmonic"`` or
        ``"percussive"``.
    :param margin: HPSS margin parameter passed to
        :func:`librosa.effects.hpss`.  Larger values yield harder
        separation at the cost of residual artefacts.  Defaults to ``1``
        (the librosa default, equal-weighting).
    :raises ValueError: If *component* is not ``"harmonic"`` or
        ``"percussive"``.
    """

    def __init__(self, component: str = "harmonic", margin: float = 1.0) -> None:
        if component not in _VALID_COMPONENTS:
            raise ValueError(
                f"component must be one of {sorted(_VALID_COMPONENTS)!r}, "
                f"got {component!r}"
            )
        self.component = component
        self.margin = margin

    # ------------------------------------------------------------------
    def __call__(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """
        Separate *signal* and return the selected component.

        :param signal: 1-D float32 audio samples.
        :type signal: np.ndarray
        :param kwargs: Keyword arguments passed to :func:`librosa.effects.hpss` (e.g. *kernel_size*).  See the librosa docs for details.
        :returns: The harmonic or percussive component, same shape and dtype
            as *signal*.
        :rtype: np.ndarray
        """
        harmonic, percussive = librosa.effects.hpss(signal, margin=self.margin, **kwargs)
        return harmonic if self.component == "harmonic" else percussive

    # ------------------------------------------------------------------
    def to_config(self) -> dict:
        """
        Return a JSON-serialisable dict suitable for W&B / logging.

        :returns: ``{"signal_transform": "hpss", "component": ..., "margin": ...}``
        :rtype: dict
        """
        return {
            "signal_transform": "hpss",
            "component": self.component,
            "margin": self.margin,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"HpssTransform(component={self.component!r}, margin={self.margin})"


# ── module-level convenience callables ───────────────────────────────────

hpss_harmonic: SignalTransform = HpssTransform(component="harmonic")
"""Pre-built :class:`HpssTransform` that keeps the harmonic component."""

hpss_percussive: SignalTransform = HpssTransform(component="percussive")
"""Pre-built :class:`HpssTransform` that keeps the percussive component."""


# ── factory ──────────────────────────────────────────────────────────────

def make_hpss_transform(component: str, margin: float = 1.0) -> HpssTransform:
    """
    Construct a :class:`HpssTransform` for the given component name.

    :param component: ``"harmonic"`` or ``"percussive"``.
    :param margin: Passed to :class:`HpssTransform`.
    :returns: A configured :class:`HpssTransform` instance.
    :raises ValueError: If *component* is invalid.
    """
    return HpssTransform(component=component, margin=margin)


def resolve_signal_transform(
    hpss_component: str | None,
    entity_type: str | None = None,
) -> HpssTransform | None:
    """
    Resolve a ``hpss_component`` string (from CLI / TOML) to a
    :class:`HpssTransform`, using the entity-type default as a fallback.

    Resolution order:
    1. If *hpss_component* is a non-empty string (``"harmonic"`` /
       ``"percussive"``), use it directly — explicit override.
    2. If *hpss_component* is ``None`` or empty **and** *entity_type* is in
       :data:`ENTITY_TYPE_COMPONENT`, use the mapped default.
    3. Otherwise return ``None`` (no transform).

    :param hpss_component: Value from ``--hpss-component`` CLI / TOML.
        Pass ``None`` or ``""`` to trigger the entity-type fallback.
    :param entity_type: Entity type string (e.g. ``"fan"``).
    :returns: A :class:`HpssTransform` or ``None``.
    """
    component: str | None = None

    if hpss_component == "none":
        return None
    elif hpss_component and hpss_component not in ("",):
        component = hpss_component
    elif entity_type and entity_type in ENTITY_TYPE_COMPONENT:
        component = ENTITY_TYPE_COMPONENT[entity_type]

    if component is None:
        return None
    return HpssTransform(component=component)


