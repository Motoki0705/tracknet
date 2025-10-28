"""Model subpackage for TrackNet.

Includes backbones, decoders, and heads to produce heatmap predictions.
"""

from .backbones.vit_backbone import ViTBackbone, ViTBackboneConfig
from .backbones.convnext_backbone import ConvNeXtBackbone, ConvNeXtBackboneConfig
from .decoders.upsampling_decoder import UpsamplingDecoder
from .decoders.fpn_decoder import FPNDecoder, FPNDecoderConfig
from .heads.heatmap_head import HeatmapHead

__all__ = [
    "ViTBackbone",
    "ViTBackboneConfig",
    "UpsamplingDecoder",
    "HeatmapHead",
    "ConvNeXtBackbone",
    "ConvNeXtBackboneConfig",
    "FPNDecoder",
    "FPNDecoderConfig",
]

