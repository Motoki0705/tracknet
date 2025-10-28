"""Collate utilities for building batches with heatmaps and masks.

This module provides collate functions that:
- stack images from dataset samples
- generate Gaussian heatmaps centered at the provided coordinates
- produce visibility masks to exclude missing targets from loss

Heatmap generation:
- Uses an isotropic Gaussian with standard deviation ``sigma`` (in heatmap
  pixels). The target heatmap size is configurable via ``(width, height)``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import math
import torch


def gaussian_2d(width: int, height: int, cx: float, cy: float, sigma: float) -> torch.Tensor:
    """Create a 2D Gaussian heatmap.

    Args:
        width: Heatmap width (W).
        height: Heatmap height (H).
        cx: Center x in heatmap pixel coordinates.
        cy: Center y in heatmap pixel coordinates.
        sigma: Standard deviation of the Gaussian (in pixels).

    Returns:
        Tensor of shape ``[H, W]``.
    """

    xs = torch.arange(width, dtype=torch.float32)
    ys = torch.arange(height, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    return g


def _scale_coord_to_heatmap(
    coord: Tuple[float, float],
    image_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
) -> Tuple[float, float]:
    """Scale a coordinate from image space to heatmap space."""

    img_w, img_h = image_size
    hm_w, hm_h = heatmap_size
    sx = hm_w / float(img_w)
    sy = hm_h / float(img_h)
    x, y = coord
    return (x * sx, y * sy)


def collate_frames(
    batch: Sequence[Dict],
    heatmap_size: Tuple[int, int],
    sigma: float,
) -> Dict[str, torch.Tensor]:
    """Collate a batch of frame samples and build heatmaps/masks.

    Args:
        batch: List of samples from a frame dataset (keys: image, coord, visibility, meta).
        heatmap_size: Target heatmap size as ``(W, H)``.
        sigma: Gaussian sigma in heatmap pixels.

    Returns:
        Dict with keys:
        - ``images``: ``[B, C, H, W]``
        - ``heatmaps``: ``[B, 1, Hh, Wh]``
        - ``masks``: ``[B, 1, Hh, Wh]`` visibility masks
    """

    images = torch.stack([s["image"] for s in batch], dim=0)
    hm_w, hm_h = heatmap_size

    heatmaps: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for s in batch:
        vis = int(s.get("visibility", 1))
        size = tuple(s["meta"]["size"])  # (W, H)
        cx, cy = _scale_coord_to_heatmap(tuple(s["coord"]), size, heatmap_size)
        hm = gaussian_2d(hm_w, hm_h, cx, cy, sigma)
        if vis == 0:
            # If not visible, zero-out and zero mask
            mask = torch.zeros_like(hm)
            hm = hm * 0.0
        else:
            mask = torch.ones_like(hm)
        heatmaps.append(hm.unsqueeze(0))  # [1, Hh, Wh]
        masks.append(mask.unsqueeze(0))

    heatmaps_t = torch.stack(heatmaps, dim=0)
    masks_t = torch.stack(masks, dim=0)
    return {"images": images, "heatmaps": heatmaps_t, "masks": masks_t}


def collate_sequences(
    batch: Sequence[Dict],
    heatmap_size: Tuple[int, int],
    sigma: float,
) -> Dict[str, torch.Tensor]:
    """Collate a batch of sequence samples and build heatmaps/masks.

    Args:
        batch: List of sequence samples (keys: images [T,C,H,W], coords [T], visibility [T], meta.sizes [T]).
        heatmap_size: Target heatmap size as ``(W, H)``.
        sigma: Gaussian sigma in heatmap pixels.

    Returns:
        Dict with keys:
        - ``images``: ``[B, T, C, H, W]``
        - ``heatmaps``: ``[B, T, 1, Hh, Wh]``
        - ``masks``: ``[B, T, 1, Hh, Wh]``
    """

    images = torch.stack([s["images"] for s in batch], dim=0)  # [B, T, C, H, W]
    hm_w, hm_h = heatmap_size

    heatmaps: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for s in batch:
        T = s["images"].shape[0]
        sizes = [tuple(x) for x in s["meta"]["sizes"]]
        hms_t: List[torch.Tensor] = []
        mks_t: List[torch.Tensor] = []
        for t in range(T):
            cx, cy = _scale_coord_to_heatmap(tuple(s["coords"][t]), sizes[t], heatmap_size)
            hm = gaussian_2d(hm_w, hm_h, cx, cy, sigma)
            vis = int(s["visibility"][t])
            if vis == 0:
                mk = torch.zeros_like(hm)
                hm = hm * 0.0
            else:
                mk = torch.ones_like(hm)
            hms_t.append(hm.unsqueeze(0))
            mks_t.append(mk.unsqueeze(0))
        heatmaps.append(torch.stack(hms_t, dim=0))  # [T, 1, Hh, Wh]
        masks.append(torch.stack(mks_t, dim=0))

    return {
        "images": images,
        "heatmaps": torch.stack(heatmaps, dim=0),
        "masks": torch.stack(masks, dim=0),
    }

