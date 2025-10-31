"""ConvNeXt backbone wrapper providing multi-scale features for FPN.

Always uses a Hugging Face (HF) pretrained ConvNeXt via AutoModel and returns
``hidden_states`` as feature maps. It expects the model to be cached locally
(``local_files_only=True`` by default).

Output format:
- A list of tensors ``[C1, C2, C3, C4, C5]`` (by default), each with shape
  ``[B, C_i, H_i, W_i]`` ordered from higher to lower resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class ConvNeXtBackboneConfig:
    """Configuration for ConvNeXt backbone (HF only).

    Attributes:
        pretrained_model_name: HF model identifier.
        return_stages: Indices of hidden_states to return (default: (0,1,2,3,4)).
        device_map: Passed to HF .from_pretrained (e.g., "auto" or None).
        local_files_only: If True, load only from local cache.
    """
    pretrained_model_name: str = "facebook/dinov3-convnext-base-pretrain-lvd1689m"
    return_stages: Sequence[int] = (0, 1, 2, 3, 4)  # <-- C1..C5 をデフォルトで返す
    device_map: Optional[str] = "auto"
    local_files_only: bool = True


class ConvNeXtBackbone(nn.Module):
    """Return multi-scale ConvNeXt features suitable for FPN decoders (HF only)."""

    def __init__(self, cfg: ConvNeXtBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        print("Using Hugging Face ConvNeXt pretrained backbone.")

        try:
            from transformers import AutoModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "transformers is not available. Please install `transformers` to use this backbone."
            ) from e

        try:
            self.model = AutoModel.from_pretrained(
                cfg.pretrained_model_name,
                device_map=cfg.device_map,          # fixed: device_map
                local_files_only=cfg.local_files_only,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load the pretrained model from local cache. "
                "Ensure the model is cached locally or set `local_files_only=False` in the config."
            ) from e

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute multi-scale feature maps.

        Args:
            x: Input image tensor ``[B, 3, H, W]``.

        Returns:
            List of feature maps (default ``[C1, C2, C3, C4, C5]``) with decreasing spatial sizes.
        """
        out = self.model(x, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states  # type: ignore[attr-defined]
        feats: List[torch.Tensor] = []
        for idx in self.cfg.return_stages:
            f = hs[idx]  # expected [B, C, H, W]
            feats.append(f)
        return feats
