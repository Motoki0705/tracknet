# tracknet/model/__init__.py
"""
Model subpackage for TrackNet.

Includes backbones, decoders, and heads to produce heatmap predictions.
Lazily exposes classes at module top-level to reduce import time.
"""

from typing import TYPE_CHECKING


def _import_backbones():
    from .backbones.convnext_backbone import ConvNeXtBackbone, ConvNeXtBackboneConfig
    from .backbones.vit_backbone import ViTBackbone, ViTBackboneConfig

    return ViTBackbone, ViTBackboneConfig, ConvNeXtBackbone, ConvNeXtBackboneConfig


def _import_decoders():
    from .decoders.fpn_decoder import FPNDecoderConfig, FPNDecoderTorchvision
    from .decoders.hrnet_decoder import HRDecoder, HRDecoderConfig
    from .decoders.upsampling_decoder import UpsamplingDecoder

    return (
        UpsamplingDecoder,
        FPNDecoderTorchvision,
        FPNDecoderConfig,
        HRDecoder,
        HRDecoderConfig,
    )


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

_lazy_map = {
    "ViTBackbone": lambda: _import_backbones()[0],
    "ViTBackboneConfig": lambda: _import_backbones()[1],
    "ConvNeXtBackbone": lambda: _import_backbones()[2],
    "ConvNeXtBackboneConfig": lambda: _import_backbones()[3],
    "UpsamplingDecoder": lambda: _import_decoders()[0],
    "FPNDecoderTorchvision": lambda: _import_decoders()[1],
    "FPNDecoderConfig": lambda: _import_decoders()[2],
    "HRDecoder": lambda: _import_decoders()[3],
    "HRDecoderConfig": lambda: _import_decoders()[4],
    "HeatmapModel": lambda: _import_build()[0],
    "build_model": lambda: _import_build()[1],
    "HeatmapHead": lambda: _import_heads(),
}


def __getattr__(name: str) -> Any:
    if name in _lazy_map:
        try:
            obj = _lazy_map[name]()
        except ImportError:
            raise
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys()) + __all__))


if TYPE_CHECKING:
    from .backbones.convnext_backbone import ConvNeXtBackbone, ConvNeXtBackboneConfig
    from .backbones.vit_backbone import ViTBackbone, ViTBackboneConfig
    from .build import HeatmapModel, build_model
    from .decoders.fpn_decoder import FPNDecoderConfig, FPNDecoderTorchvision
    from .decoders.hrnet_decoder import HRDecoder, HRDecoderConfig
    from .decoders.upsampling_decoder import UpsamplingDecoder
    from .heads.heatmap_head import HeatmapHead
