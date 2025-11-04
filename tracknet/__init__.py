"""Top-level package for TrackNet.

This package provides modules for datasets, models, training, utilities,
and scripts to train and use a tennis ball detection model that predicts
heatmaps from images using a ViT backbone and an upsampling decoder.

The default pretrained model name is exposed from ``tracknet.constants``.
"""

from .constants import DEFAULT_PRETRAINED_MODEL

__all__ = [
    "DEFAULT_PRETRAINED_MODEL",
]
