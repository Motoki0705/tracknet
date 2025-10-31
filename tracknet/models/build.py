"""Model factory utilities for TrackNet.

This module centralizes the construction of heatmap prediction models based on
an OmegaConf model configuration. It exposes a single ``build_model`` function
that is used by the trainer and inference scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig

from tracknet.models.backbones.convnext_backbone import (
    ConvNeXtBackbone,
    ConvNeXtBackboneConfig,
)
from tracknet.models.backbones.vit_backbone import ViTBackbone, ViTBackboneConfig
from tracknet.models.decoders.fpn_decoder import FPNDecoder, FPNDecoderConfig
from tracknet.models.decoders.upsampling_decoder import UpsamplingDecoder
from tracknet.models.heads.heatmap_head import HeatmapHead


class HeatmapModel(nn.Module):
    """Wrapper module that assembles backbone, decoder, and head.

    The concrete architecture is chosen based on the presence of ``decoder`` or
    ``fpn`` entries inside ``model_cfg``.
    """

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__()
        hm_w, hm_h = int(model_cfg.heatmap.size[0]), int(model_cfg.heatmap.size[1])
        out_size = (hm_h, hm_w)  # NCHW expects (H, W)

        backbone_cfg = getattr(model_cfg, "backbone", {})
        self.freeze_backbone = bool(backbone_cfg.get("freeze", True))

        if hasattr(model_cfg, "fpn"):
            self.variant = "convnext_fpn"
            self.backbone = ConvNeXtBackbone(
                ConvNeXtBackboneConfig(
                    pretrained_model_name=str(
                        model_cfg.get(
                            "pretrained_model_name",
                            "facebook/dinov3-convnext-base-pretrain-lvd1689m",
                        )
                    ),
                    return_stages=tuple(
                        int(s) for s in backbone_cfg.get("return_stages", (0, 1, 2, 3, 4))
                    ),
                    device_map=str(backbone_cfg.get("device_map", "auto"))
                    if backbone_cfg.get("device_map", "auto") is not None
                    else None,
                    local_files_only=bool(backbone_cfg.get("local_files_only", True)),
                )
            )
            self.decoder = FPNDecoder(
                FPNDecoderConfig(
                    in_channels=[int(c) for c in model_cfg.fpn.in_channels],
                    lateral_dim=int(model_cfg.fpn.get("lateral_dim", 256)),
                    fuse=str(model_cfg.fpn.get("fuse", "sum")),
                    out_size=out_size,
                )
            )
            self.head = HeatmapHead(int(model_cfg.fpn.get("lateral_dim", 256)))
        elif hasattr(model_cfg, "decoder"):
            self.variant = "vit_upsample"
            self.backbone = ViTBackbone(
                ViTBackboneConfig(
                    pretrained_model_name=str(
                        model_cfg.get(
                            "pretrained_model_name",
                            "facebook/dinov3-vits16-pretrain-lvd1689m",
                        )
                    ),
                    device_map=str(backbone_cfg.get("device_map", "auto"))
                    if backbone_cfg.get("device_map", "auto") is not None
                    else None,
                    local_files_only=bool(backbone_cfg.get("local_files_only", True)),
                    patch_size=int(backbone_cfg.get("patch_size", 16)),
                )
            )
            channels = [int(c) for c in model_cfg.decoder.channels]
            upfactors = [int(u) for u in model_cfg.decoder.upsample]
            self.decoder = UpsamplingDecoder(
                channels,
                upfactors,
                out_size=out_size,
                blocks_per_stage=int(model_cfg.decoder.get("blocks_per_stage", 1)),
                norm=str(model_cfg.decoder.get("norm", "gn")),
                activation=str(model_cfg.decoder.get("activation", "gelu")),
                use_depthwise=bool(model_cfg.decoder.get("use_depthwise", True)),
                use_se=bool(model_cfg.decoder.get("use_se", False)),
                se_reduction=int(model_cfg.decoder.get("se_reduction", 8)),
                dropout=float(model_cfg.decoder.get("dropout", 0.0)),
            )
            self.head = HeatmapHead(channels[-1])
        else:
            raise ValueError("model config must contain either 'decoder' (ViT) or 'fpn' (ConvNeXt)")

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.variant == "vit_upsample":
            tokens = self.backbone(images)  # [B, Hp, Wp, C]
            features = self.decoder(tokens)  # type: ignore[arg-type]
            return self.head(features)

        if self.variant == "convnext_fpn":
            feats = self.backbone(images)  # type: ignore[call-arg]
            pyramid = self.decoder(feats)  # type: ignore[arg-type]
            return self.head(pyramid)


def build_model(model_cfg: DictConfig) -> nn.Module:
    """Build a TrackNet heatmap model from configuration.

    Args:
        model_cfg: OmegaConf section describing the model.

    Returns:
        nn.Module: a constructed heatmap prediction model.
    """
    return HeatmapModel(model_cfg)
