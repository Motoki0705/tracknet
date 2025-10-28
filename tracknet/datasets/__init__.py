"""Datasets for TrackNet.

Modules:
- ``base.image_dataset``: Base class for frame datasets.
- ``base.sequence_dataset``: Base class for sequence datasets.
- ``utils.augmentations``: Augmentation helpers.
- ``utils.collate``: Batch collation and heatmap generation.
- ``tracknet_frame``: Concrete TrackNet frame dataset.
- ``tracknet_sequence``: Concrete TrackNet sequence dataset.
"""

from .base.image_dataset import BaseImageDataset, PreprocessConfig
from .base.sequence_dataset import BaseSequenceDataset
from .utils.augmentations import IMAGENET_MEAN, IMAGENET_STD
from .utils.collate import (
    gaussian_2d,
    collate_frames,
    collate_sequences,
)
from .tracknet_frame import TrackNetFrameDataset, TrackNetFrameDatasetConfig
from .tracknet_sequence import TrackNetSequenceDataset, TrackNetSequenceDatasetConfig

__all__ = [
    "BaseImageDataset",
    "BaseSequenceDataset",
    "PreprocessConfig",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "gaussian_2d",
    "collate_frames",
    "collate_sequences",
    "TrackNetFrameDataset",
    "TrackNetFrameDatasetConfig",
    "TrackNetSequenceDataset",
    "TrackNetSequenceDatasetConfig",
]
