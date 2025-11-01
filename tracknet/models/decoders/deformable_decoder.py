"""Deformable DETR decoder with multi-scale feature encoding for ConvNext.

This module implements a decoder that uses Deformable DETR's multi-scale 
deformable attention to encode ConvNext features and outputs FPN-style
feature pyramids for heatmap generation.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

# Import Deformable DETR components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../third_party/Deformable-DETR'))

from models.ops.modules import MSDeformAttn
from models.position_encoding import PositionEmbeddingSine


@dataclass
class DeformableDecoderConfig:
    """Configuration for Deformable DETR decoder.
    
    Attributes:
        in_channels: List of input channels from ConvNext backbone.
        d_model: Hidden dimension for transformer.
        nhead: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        num_feature_levels: Number of feature levels for multi-scale attention.
        n_points: Number of sampling points per attention head.
        lateral_dim: Dimension for FPN lateral connections.
        out_size: Output spatial size (H, W).
    """
    in_channels: List[int]
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 3
    num_feature_levels: int = 4
    n_points: int = 4
    lateral_dim: int = 256
    out_size: Optional[Tuple[int, int]] = None


class DeformableTransformerEncoderLayer(nn.Module):
    """Single layer of Deformable Transformer encoder."""
    
    def __init__(
        self,
        d_model: int = 256,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        
        # Self-attention with multi-scale deformable attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = getattr(F, activation)
        
    def forward(self, src: torch.Tensor, spatial_shapes: torch.Tensor, 
                level_start_index: torch.Tensor, pos_lvl_embed: torch.Tensor) -> torch.Tensor:
        """Forward pass of encoder layer.
        
        Args:
            src: Input features [B, H*W, C]
            spatial_shapes: Spatial shapes of each level [num_levels, 2]
            level_start_index: Start index of each level [num_levels]
            pos_lvl_embed: Position + level embeddings [B, H*W, C]
            
        Returns:
            Encoded features [B, H*W, C]
        """
        # Self attention
        src2 = self.self_attn(
            src + pos_lvl_embed,  # query
            spatial_shapes,       # spatial_shapes
            level_start_index,    # level_start_index
            src                   # value
        )
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class DeformableTransformerEncoder(nn.Module):
    """Deformable Transformer encoder for multi-scale feature encoding."""
    
    def __init__(self, encoder_layer: DeformableTransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, src: torch.Tensor, spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor, pos_lvl_embed: torch.Tensor) -> torch.Tensor:
        """Forward pass of encoder.
        
        Args:
            src: Input features [B, H*W, C]
            spatial_shapes: Spatial shapes of each level [num_levels, 2]
            level_start_index: Start index of each level [num_levels]
            pos_lvl_embed: Position + level embeddings [B, H*W, C]
            
        Returns:
            Encoded features [B, H*W, C]
        """
        output = src
        
        for layer in self.layers:
            output = layer(output, spatial_shapes, level_start_index, pos_lvl_embed)
            
        return output


class DeformableFPNDecoder(nn.Module):
    """Decoder that combines Deformable DETR encoding with FPN for heatmap output.
    
    This decoder:
    1. Projects ConvNext multi-scale features to transformer dimension
    2. Adds position and level embeddings
    3. Applies Deformable Transformer encoder for multi-scale feature fusion
    4. Reshapes features back to spatial format
    5. Applies FPN for pyramid feature generation
    """
    
    def __init__(self, cfg: DeformableDecoderConfig):
        super().__init__()
        self.cfg = cfg
        
        # Project ConvNext features to transformer dimension
        self.input_proj = nn.ModuleList([
            nn.Conv2d(in_channels, cfg.d_model, kernel_size=1)
            for in_channels in cfg.in_channels
        ])
        
        # Position encoding
        self.pos_embed = PositionEmbeddingSine(cfg.d_model // 2, normalize=True)
        
        # Level embeddings
        self.level_embed = nn.Parameter(torch.Tensor(cfg.num_feature_levels, cfg.d_model))
        
        # Deformable Transformer encoder
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model=cfg.d_model,
            dim_feedforward=cfg.d_model * 4,
            n_levels=cfg.num_feature_levels,
            n_heads=cfg.nhead,
            n_points=cfg.n_points,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, cfg.num_encoder_layers)
        
        # Output projection to FPN dimension
        self.output_proj = nn.Conv2d(cfg.d_model, cfg.lateral_dim, kernel_size=1)
        
        # FPN for final feature pyramid
        self.fpn = FeaturePyramidNetwork(
            [cfg.lateral_dim] * len(cfg.in_channels), 
            cfg.lateral_dim
        )
        
        # Initialize parameters
        nn.init.normal_(self.level_embed)
        
    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of deformable FPN decoder.
        
        Args:
            feats: List of ConvNext features [C2, C3, C4, C5] (high to low resolution)
            
        Returns:
            Fused heatmap features [B, lateral_dim, H, W]
        """
        # Project features to transformer dimension
        proj_feats = [proj(feat) for proj, feat in zip(self.input_proj, feats)]
        
        # Get spatial shapes
        spatial_shapes = torch.tensor(
            [[feat.shape[-2], feat.shape[-1]] for feat in proj_feats],
            dtype=torch.long,
            device=proj_feats[0].device
        )
        
        # Add position embeddings
        pos_embeds = [
            self.pos_embed(feat).flatten(2).permute(2, 0, 1)  # [H*W, B, C]
            for feat in proj_feats
        ]
        
        # Flatten features and concatenate
        feat_flattened = []
        for feat in proj_feats:
            feat_flat = feat.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
            feat_flattened.append(feat_flat)
        
        # Concatenate all levels
        src = torch.cat(feat_flattened, dim=0)  # [total_tokens, B, C]
        pos_embed = torch.cat(pos_embeds, dim=0)  # [total_tokens, B, C]
        
        # Create level start index
        level_start_index = torch.cat([
            torch.zeros(1, dtype=torch.long, device=src.device),
            spatial_shapes[:, 0] * spatial_shapes[:, 1].cumsum(0)[:-1]
        ])
        
        # Add level embeddings
        level_embed_list = []
        for i in range(len(proj_feats)):
            level_embed = self.level_embed[i].view(1, 1, -1).expand(
                spatial_shapes[i, 0] * spatial_shapes[i, 1], -1, -1
            )
            level_embed_list.append(level_embed)
        
        level_embed = torch.cat(level_embed_list, dim=0)  # [total_tokens, 1, C]
        level_embed = level_embed.permute(1, 0, 2)  # [1, total_tokens, C]
        
        # Combine position and level embeddings
        pos_lvl_embed = pos_embed + level_embed
        
        # Apply Deformable Transformer encoder
        encoded = self.encoder(
            src.permute(1, 0, 2),  # [B, total_tokens, C]
            spatial_shapes,
            level_start_index,
            pos_lvl_embed.permute(1, 0, 2)  # [B, total_tokens, C]
        ).permute(1, 0, 2)  # [total_tokens, B, C]
        
        # Split encoded features back to levels
        encoded_feats = []
        start_idx = 0
        for i, (h, w) in enumerate(spatial_shapes):
            end_idx = start_idx + h * w
            feat = encoded[start_idx:end_idx].permute(1, 2, 0)  # [B, C, H*W]
            feat = feat.view(-1, self.cfg.d_model, h, w)  # [B, C, H, W]
            encoded_feats.append(feat)
            start_idx = end_idx
        
        # Project to FPN dimension
        fpn_inputs = [
            self.output_proj(feat) for feat in encoded_feats
        ]
        
        # Apply FPN
        fpn_dict = OrderedDict([(f"c{i}", feat) for i, feat in enumerate(fpn_inputs)])
        fpn_outputs = self.fpn(fpn_dict)
        
        # Get FPN features and upsample to target size
        fpn_feats = list(fpn_outputs.values())
        target_h, target_w = self.cfg.out_size or fpn_feats[0].shape[-2:]
        
        upsampled = [
            F.interpolate(feat, size=(target_h, target_w), 
                         mode='bilinear', align_corners=False)
            for feat in fpn_feats
        ]
        
        # Sum all FPN features
        return torch.stack(upsampled, dim=0).sum(dim=0)


if __name__ == "__main__":
    # Simple test
    B = 2
    in_channels = [128, 256, 512, 1024]
    spatial_sizes = [(64, 64), (32, 32), (16, 16), (8, 8)]
    
    # Create dummy ConvNext features
    feats = [
        torch.randn(B, c, h, w) 
        for c, (h, w) in zip(in_channels, spatial_sizes)
    ]
    
    # Create decoder
    cfg = DeformableDecoderConfig(
        in_channels=in_channels,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_feature_levels=4,
        n_points=4,
        lateral_dim=256,
        out_size=(128, 128)
    )
    
    decoder = DeformableFPNDecoder(cfg)
    output = decoder(feats)
    
    print(f"Input shapes: {[f.shape for f in feats]}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [B, {cfg.lateral_dim}, {cfg.out_size[0]}, {cfg.out_size[1]}]")
