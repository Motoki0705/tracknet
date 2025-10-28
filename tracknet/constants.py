"""Global constants used across TrackNet.

Attributes:
    DEFAULT_PRETRAINED_MODEL (str):
        Default Hugging Face model identifier used by the ViT backbone.
        This mirrors the value used in ``demo/vit_demo.py`` so examples and
        training share the same default.
"""

DEFAULT_PRETRAINED_MODEL: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"

