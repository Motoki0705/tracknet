"""Augmentation utilities for datasets.

Includes geometric transforms that preserve coordinate consistency, and
helpers to convert images to tensors with optional normalization.
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torchvision.transforms.functional as F
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _horizontal_flip(
    img: Image.Image, coord: tuple[float, float]
) -> tuple[Image.Image, tuple[float, float]]:
    """Flip image horizontally and update coordinate accordingly.

    Args:
        img: Input PIL image.
        coord: Point ``(x, y)`` in pixel coordinates.

    Returns:
        A tuple of (flipped_image, adjusted_coord).
    """

    w, _ = img.size
    flipped = F.hflip(img)
    x, y = coord
    # New x is mirrored across center line. Pixel indices are 0..w-1.
    new_x = (w - 1) - x
    return flipped, (float(new_x), float(y))


def _resize(
    img: Image.Image, coord: tuple[float, float], size: tuple[int, int]
) -> tuple[Image.Image, tuple[float, float]]:
    """Resize image to ``size=(W, H)`` and scale the coordinate accordingly."""

    target_w, target_h = size
    src_w, src_h = img.size
    sx = target_w / float(src_w)
    sy = target_h / float(src_h)
    resized = img.resize((target_w, target_h), Image.BILINEAR)
    x, y = coord
    return resized, (float(x) * sx, float(y) * sy)


def apply_augmentations_single(
    img: Image.Image,
    coord: tuple[float, float],
    cfg: Any,
) -> tuple[Image.Image, tuple[float, float]]:
    """Apply geometric augmentations to a single image/coord pair.

    Applies optional resize and probabilistic horizontal flip, keeping the
    coordinate consistent. Color transforms can be added as needed.

    Args:
        img: PIL image in RGB.
        coord: ``(x, y)`` in pixel coordinates.
        cfg: Preprocess settings.

    Returns:
        Tuple of (possibly augmented image, adjusted coord).
    """

    if cfg.resize is not None:
        img, coord = _resize(img, coord, cfg.resize)

    if cfg.flip_prob > 0.0 and random.random() < cfg.flip_prob:
        img, coord = _horizontal_flip(img, coord)

    return img, coord


def to_tensor_and_normalize(img: Image.Image, normalize: bool = True) -> torch.Tensor:
    """Convert PIL image to tensor and optionally normalize with ImageNet stats.

    Args:
        img: PIL image in RGB.
        normalize: If True, apply ImageNet mean/std normalization.

    Returns:
        Tensor of shape ``[C, H, W]`` in float32.
    """

    t = F.to_tensor(img)  # [0,1]
    if normalize:
        t = F.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return t
