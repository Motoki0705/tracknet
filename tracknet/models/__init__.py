"""Model subpackage for TrackNet.

Includes backbones, decoders, and heads to produce heatmap predictions.
"""

# Lazy imports to improve startup time
def _import_backbones():
    from .backbones.vit_backbone import ViTBackbone, ViTBackboneConfig
    from .backbones.convnext_backbone import ConvNeXtBackbone, ConvNeXtBackboneConfig
    return ViTBackbone, ViTBackboneConfig, ConvNeXtBackbone, ConvNeXtBackboneConfig

def _import_decoders():
    from .decoders.upsampling_decoder import UpsamplingDecoder
    from .decoders.fpn_decoder import FPNDecoderTorchvision, FPNDecoderConfig
    from .decoders.hrnet_decoder import HRDecoder, HRDecoderConfig
    return UpsamplingDecoder, FPNDecoderTorchvision, FPNDecoderConfig, HRDecoder, HRDecoderConfig

def _import_build():
    from .build import HeatmapModel, build_model
    return HeatmapModel, build_model

def _import_heads():
    from .heads.heatmap_head import HeatmapHead
    return HeatmapHead

__all__ = [
    "ViTBackbone",
    "ViTBackboneConfig",
    "UpsamplingDecoder",
    "HeatmapHead",
    "HeatmapModel",
    "build_model",
    "ConvNeXtBackbone",
    "ConvNeXtBackboneConfig",
    "FPNDecoderTorchvision",
    "FPNDecoderConfig",
    "HRDecoder",
    "HRDecoderConfig",
]

