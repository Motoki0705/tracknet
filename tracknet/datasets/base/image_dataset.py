"""Base classes for image-based datasets.

Provides abstract helpers to load single-image samples with an associated
target coordinate and visibility flag. Subclasses implement the indexing and
record management for specific datasets.

Design:
- Return per-sample dicts that include the raw image tensor, the target
  coordinate in pixel space, and a visibility flag. Heatmaps are produced later
  by the collate utilities to allow flexible output sizes.

Key fields returned by ``__getitem__``:
- ``image``: ``torch.FloatTensor`` of shape ``[C, H, W]`` in ``[0, 1]`` range.
- ``coord``: ``tuple[float, float]`` as ``(x, y)`` in pixels for the image.
- ``visibility``: ``int`` in ``{0, 1}``.
- ``meta``: dict containing auxiliary info such as ``{"size": (W, H)}`` and
  optional file path/context identifiers.

Note:
- Augmentations (e.g., horizontal flip, optional resize, normalization) are
  handled via ``tracknet.datasets.utils.augmentations``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from tracknet.datasets.utils.augmentations import (
    apply_augmentations_single,
    to_tensor_and_normalize,
)


@dataclass
class PreprocessConfig:
    """Configuration for dataset preprocessing/augmentations.

    Attributes:
        resize: Optional ``(width, height)`` to resize images and scale coords.
        normalize: Whether to apply ImageNet mean/std normalization.
        flip_prob: Probability of horizontal flip.
    """

    resize: Optional[Tuple[int, int]] = None
    normalize: bool = True
    flip_prob: float = 0.0


class BaseImageDataset(Dataset):
    """Abstract image dataset returning dict samples.

    Subclasses must implement ``_get_record`` to provide an access record for
    an index, and ``__len__``. The base class then loads the image, applies
    augmentations, and returns a standardized dict.
    """

    def __init__(self, preprocess: Optional[PreprocessConfig] = None) -> None:
        """Initialize the dataset.

        Args:
            preprocess: Preprocess/augmentation settings. If ``None``, defaults
                are used.
        """

        self.preprocess = preprocess or PreprocessConfig()

    # ----- Methods expected from subclasses -----
    def _get_record(self, index: int) -> Dict[str, Any]:  # pragma: no cover - abstract
        """Return a record dict with keys ``path``, ``coord``, ``visibility``.

        The base class expects the following keys:
        - ``path``: Path to an image file loadable by PIL.
        - ``coord``: ``(x, y)`` in pixels for the un-augmented image.
        - ``visibility``: ``0`` or ``1``.

        Subclasses may include additional keys which will be forwarded into the
        returned ``meta`` dict.
        """

        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    # ----- Core loading logic -----
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Load and return a single image sample.

        Args:
            index: Sample index.

        Returns:
            Dict with keys ``image``, ``coord``, ``visibility``, and ``meta``.
        """

        rec = self._get_record(index)
        path: str = rec["path"]
        coord: Tuple[float, float] = tuple(rec["coord"])  # type: ignore[assignment]
        visibility: int = int(rec.get("visibility", 1))

        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size

        # Apply geometric/color augmentations that also keep coord consistent.
        img, coord = apply_augmentations_single(
            img,
            coord,
            self.preprocess,
        )

        # Convert to tensor and optionally normalize
        img_tensor = to_tensor_and_normalize(img, normalize=self.preprocess.normalize)

        sample: Dict[str, Any] = {
            "image": img_tensor,
            "coord": coord,
            "visibility": visibility,
            "meta": {
                "size": (orig_w, orig_h),
                "path": path,
                **{k: v for k, v in rec.items() if k not in {"path", "coord", "visibility"}},
            },
        }
        return sample

