"""Metrics for heatmap-based localization.

Includes coordinate extraction via argmax and soft-argmax, and error metrics
such as L2 distance and PCK@r. All metrics can leverage visibility masks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def heatmap_argmax_coords(hm: torch.Tensor) -> torch.Tensor:
    """Compute integer argmax coordinates from heatmaps.

    Args:
        hm: Heatmaps of shape ``[B, 1, H, W]``.

    Returns:
        Tensor of shape ``[B, 2]`` containing ``(x, y)`` in heatmap pixels.
    """

    b, _, h, w = hm.shape
    flat = hm.view(b, -1)
    idx = flat.argmax(dim=1)
    y = idx // w
    x = idx % w
    coords = torch.stack([x.to(torch.float32), y.to(torch.float32)], dim=1)
    return coords


def heatmap_soft_argmax_coords(hm: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
    """Compute soft-argmax coordinates from heatmaps.

    Args:
        hm: Heatmaps of shape ``[B, 1, H, W]``.
        beta: Softmax temperature scaling factor.

    Returns:
        Tensor of shape ``[B, 2]`` with continuous coordinates ``(x, y)``.
    """

    b, _, h, w = hm.shape
    flat = hm.view(b, -1) * beta
    prob = F.softmax(flat, dim=1)
    ys = torch.arange(h, dtype=torch.float32, device=hm.device)
    xs = torch.arange(w, dtype=torch.float32, device=hm.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [H*W, 2]
    coords = prob @ grid  # [B,2]
    return coords


def l2_error(
    pred_xy: torch.Tensor, target_xy: torch.Tensor, visible: torch.Tensor
) -> torch.Tensor:
    """Compute mean L2 distance between predicted and target coordinates.

    Args:
        pred_xy: Predicted coordinates ``[B, 2]`` in ``(x, y)`` heatmap pixels.
        target_xy: Target coordinates ``[B, 2]`` in ``(x, y)`` heatmap pixels.
        visible: Visibility mask per sample ``[B]`` where ``1`` includes.

    Returns:
        Scalar tensor of mean L2 error over visible samples.
    """

    diff = pred_xy - target_xy
    dist = torch.linalg.norm(diff, dim=1)
    vis = visible.to(dist.dtype)
    denom = vis.sum().clamp_min(1.0)
    return (dist * vis).sum() / denom


def pck_at_r(
    pred_xy: torch.Tensor, target_xy: torch.Tensor, visible: torch.Tensor, r: float
) -> torch.Tensor:
    """Compute PCK@r in heatmap pixels.

    Args:
        pred_xy: Predicted coordinates ``[B, 2]``.
        target_xy: Target coordinates ``[B, 2]``.
        visible: Visibility mask per sample ``[B]``.
        r: Threshold radius in pixels.

    Returns:
        Scalar tensor with PCK in ``[0, 1]`` over visible samples.
    """

    diff = pred_xy - target_xy
    dist = torch.linalg.norm(diff, dim=1)
    vis = visible.to(torch.bool)
    if vis.sum() == 0:
        return torch.tensor(0.0, device=pred_xy.device)
    correct = (dist <= float(r)) & vis
    return correct.sum().to(torch.float32) / vis.sum().to(torch.float32)


def visible_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Derive per-sample visibility from a visibility mask map.

    Args:
        mask: Visibility mask ``[B, 1, H, W]`` with zeros for missing targets.

    Returns:
        Tensor ``[B]`` with 1 for any visible pixel and 0 otherwise.
    """

    b = mask.shape[0]
    vis = (mask.view(b, -1).max(dim=1).values > 0).to(torch.int64)
    return vis
