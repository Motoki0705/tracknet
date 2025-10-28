"""ConvNeXt backbone wrapper providing multi-scale features for FPN.

Supports two modes:
- Hugging Face (HF) mode: loads a ConvNeXt-based backbone via AutoModel
  (or AutoBackbone) with ``local_files_only=True`` and returns ``hidden_states``
  as feature maps. Requires a locally cached pretrained model.
- Fallback torchvision mode: uses ``torchvision.models.convnext_tiny`` and
  ``create_feature_extractor`` to obtain intermediate feature maps.

Output format:
- A list of tensors ``[C3, C4, C5]`` (optionally C2), each with shape
  ``[B, C_i, H_i, W_i]`` ordered from higher to lower resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class ConvNeXtBackboneConfig:
    """Configuration for ConvNeXt backbone.

    Attributes:
        pretrained_model_name: HF model identifier.
        use_pretrained: If True, try HF (local only). If False, use torchvision fallback.
        return_stages: Indices of hidden_states to return (HF mode). Defaults to [2,3,4].
        tv_model: Torchvision variant name for fallback (e.g., 'convnext_tiny').
    """

    pretrained_model_name: str = "facebook/dinov3-convnext-base-pretrain-lvd1689m"
    use_pretrained: bool = True
    return_stages: Sequence[int] = (2, 3, 4)
    tv_model: str = "convnext_tiny"


class ConvNeXtBackbone(nn.Module):
    """Return multi-scale ConvNeXt features suitable for FPN decoders."""

    def __init__(self, cfg: ConvNeXtBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._mode = "pretrained" if cfg.use_pretrained else "fallback"

        if self._mode == "pretrained":
            try:
                from transformers import AutoModel
            except Exception as e:  # pragma: no cover
                raise RuntimeError("transformers not available; use_pretrained=False for fallback") from e
            self.model = AutoModel.from_pretrained(
                cfg.pretrained_model_name, local_files_only=True
            )
        else:
            # Torchvision fallback
            from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large
            from torchvision.models.feature_extraction import create_feature_extractor

            tv_name = cfg.tv_model
            if tv_name == "convnext_tiny":
                base = convnext_tiny(weights=None)
            elif tv_name == "convnext_small":
                base = convnext_small(weights=None)
            elif tv_name == "convnext_base":
                base = convnext_base(weights=None)
            elif tv_name == "convnext_large":
                base = convnext_large(weights=None)
            else:
                raise ValueError(f"Unsupported tv_model: {tv_name}")

            # Return nodes for multi-scale features; empirically valid for torchvision ConvNeXt
            return_nodes = {
                "features.1": "C3",  # ~1/8 (depending on variant)
                "features.2": "C4",  # ~1/16
                "features.3": "C5",  # ~1/32
            }
            self.extractor = create_feature_extractor(base, return_nodes=return_nodes)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute multi-scale feature maps.

        Args:
            x: Input image tensor ``[B,3,H,W]``.

        Returns:
            List of feature maps ``[C3, C4, C5]`` with decreasing spatial sizes.
        """

        if self._mode == "pretrained":
            out = self.model(x, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states  # type: ignore[attr-defined]
            if hs is None:
                raise RuntimeError("HF ConvNeXt did not return hidden_states. Set output_hidden_states=True.")
            feats: List[torch.Tensor] = []
            for idx in self.cfg.return_stages:
                f = hs[idx]
                # Ensure NCHW (HF returns [B, C, H, W] already for ConvNeXt)
                feats.append(f)
            return feats
        else:
            fx = self.extractor(x)
            return [fx["C3"], fx["C4"], fx["C5"]]

