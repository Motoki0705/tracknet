"""Base classes for sequence-based datasets.

Provides helpers for returning fixed-length temporal windows consisting of
images, coordinates, and visibility flags. Heatmap generation is handled at
collate time for flexibility.

Return format for ``__getitem__``:
- ``images``: ``List[Tensor]`` or a stacked tensor of shape ``[T, C, H, W]``.
- ``coords``: ``List[tuple[float, float]]`` of length ``T``.
- ``visibility``: ``List[int]`` of length ``T``.
- ``meta``: dict including identifiers and image sizes:
  - ``"orig_sizes"``: List of ``(W, H)`` of original images before augmentations.
  - ``"sizes"``: List of ``(W, H)`` of images after augmentations (matches coord space).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from tracknet.datasets.base.image_dataset import PreprocessConfig
from tracknet.datasets.utils.augmentations import (
    apply_augmentations_single,
    to_tensor_and_normalize,
)


class BaseSequenceDataset(Dataset):
    """Abstract sequence dataset with windowing.

    Subclasses provide index-to-records mapping. Each sample is a temporal
    window of length ``self.length`` with a stride of ``self.stride``.
    """

    def __init__(
        self,
        length: int,
        stride: int,
        preprocess: PreprocessConfig | None = None,
    ) -> None:
        """Initialize the sequence dataset.

        Args:
            length: Window length ``T``.
            stride: Step between frames inside the window.
            preprocess: Preprocess/augmentation settings.
        """

        assert length > 0, "length must be > 0"
        assert stride > 0, "stride must be > 0"
        self.length = int(length)
        self.stride = int(stride)
        self.preprocess = preprocess or PreprocessConfig()

    # ----- Methods expected from subclasses -----
    def _get_window_records(
        self, index: int
    ) -> Sequence[dict[str, Any]]:  # pragma: no cover - abstract
        """Return an ordered sequence of records for this window.

        Each record must contain keys ``path``, ``coord``, and ``visibility``.
        Typically, subclasses precompute a list of windows where every window
        is represented by a list of frame records.
        """

        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    # ----- Core loading logic -----
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and return a sequence sample.

        Args:
            index: Window index.

        Returns:
            Dict with keys ``images`` (Tensor ``[T, C, H, W]``), ``coords``
            (list of ``(x, y)``), ``visibility`` (list of ints), and ``meta``.
        """

        recs = list(self._get_window_records(index))
        images: list[torch.Tensor] = []
        coords: list[tuple[float, float]] = []
        vis: list[int] = []
        orig_sizes: list[tuple[int, int]] = []
        curr_sizes: list[tuple[int, int]] = []

        # Use a consistent random decision per window (e.g. for flip)
        # by letting ``apply_augmentations_single`` handle RNG internally. Here
        # we simply apply the same preprocess config to each frame.
        for rec in recs:
            path: str = rec["path"]
            coord = tuple(rec["coord"])  # type: ignore[assignment]
            v = int(rec.get("visibility", 1))

            img = Image.open(path).convert("RGB")
            orig_sizes.append(img.size)

            img, adj_coord = apply_augmentations_single(img, coord, self.preprocess)
            curr_sizes.append(img.size)
            img_t = to_tensor_and_normalize(img, normalize=self.preprocess.normalize)

            images.append(img_t)
            coords.append(adj_coord)
            vis.append(v)

        images_t = torch.stack(images, dim=0)  # [T, C, H, W]

        return {
            "images": images_t,
            "coords": coords,
            "visibility": vis,
            "meta": {
                "orig_sizes": orig_sizes,
                "sizes": curr_sizes,
            },
        }
