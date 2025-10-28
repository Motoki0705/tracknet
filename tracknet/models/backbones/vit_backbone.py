"""ViT backbone wrapper for TrackNet.

This module provides a wrapper that can use a Hugging Face ViT (via
``AutoImageProcessor`` and ``AutoModel``) when available, and a lightweight
fallback patch embedding when offline. The forward returns patch tokens
reshaped to a spatial grid ``[B, H_p, W_p, C]``.

Usage:
    - Online/with cache: initialize with ``use_pretrained=True`` and a
      ``pretrained_model_name``.
    - Offline fallback: initialize with ``use_pretrained=False`` to use a
      simple Conv2d patch embedding with stride 16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ViTBackboneConfig:
    """Configuration for the ViT backbone wrapper.

    Attributes:
        pretrained_model_name: Hugging Face model id (e.g.,
            ``facebook/dinov3-vitb16-pretrain-lvd1689m``).
        use_pretrained: If True, try to load HF model with
            ``local_files_only=True``. If loading fails, an exception is raised.
            If False, use a lightweight Conv2d patch embedding fallback.
        fallback_dim: Output channel dimension for the fallback embedding.
        patch_size: Patch size for fallback embedding (default 16).
    """

    pretrained_model_name: str
    use_pretrained: bool = True
    fallback_dim: int = 384
    patch_size: int = 16


class ViTBackbone(nn.Module):
    """Backbone that outputs patch tokens as a spatial grid.

    Forward input shape is ``[B, C, H, W]`` and output is ``[B, H_p, W_p, C]``.
    """

    def __init__(self, cfg: ViTBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._mode = "pretrained" if cfg.use_pretrained else "fallback"

        if self._mode == "pretrained":
            try:  # Lazy import to avoid hard dependency when offline
                from transformers import AutoImageProcessor, AutoModel
            except Exception as e:  # pragma: no cover - import environment specific
                raise RuntimeError(
                    "transformers not available; set use_pretrained=False to use fallback"
                ) from e

            # Hold references in submodules for proper .to(device)
            self.processor = AutoImageProcessor.from_pretrained(
                cfg.pretrained_model_name, local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                cfg.pretrained_model_name, local_files_only=True
            )
            # Determine token dims
            hidden_size = int(self.model.config.hidden_size)
            self.out_dim = hidden_size
        else:
            # Fallback simple patch embedding: Conv2d with stride=patch_size
            self.processor = None  # type: ignore[assignment]
            self.model = None  # type: ignore[assignment]
            self.out_dim = int(cfg.fallback_dim)
            ps = int(cfg.patch_size)
            self.embed = nn.Conv2d(3, self.out_dim, kernel_size=ps, stride=ps)

    @torch.no_grad()
    def _forward_pretrained(self, images: torch.Tensor) -> torch.Tensor:
        """Forward through HF model and return patch grid.

        Args:
            images: Float tensor ``[B, C, H, W]`` in 0..1. Converted to PIL and
                processed by the associated ``AutoImageProcessor``.

        Returns:
            Tensor ``[B, H_p, W_p, C]`` of patch tokens.
        """

        from torchvision.transforms.functional import to_pil_image  # lazy import

        B = images.shape[0]
        pil_list = [to_pil_image(images[i].cpu()) for i in range(B)]
        inputs = self.processor(images=pil_list, return_tensors="pt")
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        last = outputs.last_hidden_state  # [B, 1+reg+patch, C]
        num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
        patch = last[:, 1 + num_reg :, :]  # drop cls and registers

        # Infer patch grid size (H_p x W_p)
        tokens = patch.shape[1]
        # Heuristic: assume square grid if possible
        w = int(torch.tensor(tokens).sqrt().item())
        h = tokens // w if w > 0 else tokens
        grid = patch.unflatten(1, (h, w))  # [B, H_p, W_p, C]
        return grid

    def _forward_fallback(self, images: torch.Tensor) -> torch.Tensor:
        """Fallback conv patch embedding to approximate ViT tokens.

        Args:
            images: Float tensor ``[B, C, H, W]``.

        Returns:
            Tensor ``[B, H_p, W_p, C]``.
        """

        x = self.embed(images)  # [B, C, H_p, W_p]
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute patch token grid from input images.

        Args:
            images: Float tensor ``[B, C, H, W]``.

        Returns:
            Tensor of shape ``[B, H_p, W_p, C]``.
        """

        if self._mode == "pretrained":
            return self._forward_pretrained(images)
        return self._forward_fallback(images)

