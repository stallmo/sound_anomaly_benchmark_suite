"""
features package — audio feature extraction.

Public API
----------
MelSpectrogramTransform : converts a 1-D audio frame into a flattened
                          log-mel spectrogram tensor for autoencoder input.
"""

from features.log_mel import MelSpectrogramTransform

__all__ = ["MelSpectrogramTransform"]

