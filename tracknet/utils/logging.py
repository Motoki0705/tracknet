"""Logging and visualization utilities for TrackNet.

Provides a lightweight scalar logger (CSV + optional TensorBoard) and helpers
to save heatmaps, plain images, and image/heatmap overlays during validation.
Each logger instance writes to an experiment-specific subdirectory to avoid
collisions across runs.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import torch
from PIL import Image

try:  # Optional TensorBoard
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - optional
    SummaryWriter = None  # type: ignore[assignment]


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class LoggerConfig:
    """Configuration for the scalar logger.

    Attributes:
        log_dir: Root directory where experiment logs are stored.
        run_id: Optional identifier used to create a per-run subdirectory.
        use_tensorboard: If True and available, also log to TensorBoard.
    """

    log_dir: str
    run_id: Optional[str] = None
    use_tensorboard: bool = False


class Logger:
    """Minimal scalar logger writing CSV and optionally TensorBoard."""

    def __init__(self, cfg: LoggerConfig) -> None:
        base_dir = Path(cfg.log_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # Select a per-run directory to avoid collisions between experiments.
        base_name = cfg.run_id or time.strftime("run-%Y%m%d-%H%M%S")
        run_dir = base_dir / base_name
        suffix = 1
        while run_dir.exists():
            run_dir = base_dir / f"{base_name}-{suffix:02d}"
            suffix += 1

        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "scalars.csv"
        self.csv_initialized = False
        self.tb = None
        if cfg.use_tensorboard and SummaryWriter is not None:
            self.tb = SummaryWriter(self.dir.as_posix())

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to CSV and optionally TensorBoard.

        Args:
            tag: Scalar name (e.g., ``"train/loss"``).
            value: Numeric value.
            step: Global step or epoch.
        """

        # CSV logging
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if not self.csv_initialized:
                writer.writerow(["step", "tag", "value"])  # header
                self.csv_initialized = True
            writer.writerow([step, tag, value])

        # TensorBoard logging
        if self.tb is not None:
            self.tb.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()


def _denormalize_image_tensor(t: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Denormalize a tensor image ``[C,H,W]`` given mean/std in RGB order."""

    assert t.ndim == 3 and t.shape[0] in (3, 1)
    c = t.clone().detach().cpu()
    if c.shape[0] == 3:
        m = torch.tensor(mean).view(3, 1, 1)
        s = torch.tensor(std).view(3, 1, 1)
        c = c * s + m
    c = c.clamp(0.0, 1.0)
    return c


def tensor_to_pil(img: torch.Tensor, denormalize: bool = True,
                  mean: Sequence[float] = IMAGENET_MEAN,
                  std: Sequence[float] = IMAGENET_STD) -> Image.Image:
    """Convert a tensor image ``[C,H,W]`` in 0..1 (or normalized) to PIL RGB."""

    x = img
    if denormalize:
        x = _denormalize_image_tensor(x, mean, std)
    x = (x * 255.0).round().byte()
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return Image.fromarray(x.permute(1, 2, 0).numpy(), mode="RGB")


def save_image_from_tensor(img: torch.Tensor, path: Path,
                           denormalize: bool = True,
                           mean: Sequence[float] = IMAGENET_MEAN,
                           std: Sequence[float] = IMAGENET_STD) -> None:
    """Save a tensor image ``[C,H,W]`` to disk as PNG.

    Args:
        img: Image tensor.
        path: Target path for the PNG image.
        denormalize: Whether to apply ImageNet de-normalization.
        mean: Channel-wise mean used when denormalizing.
        std: Channel-wise std used when denormalizing.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(img, denormalize=denormalize, mean=mean, std=std).save(path)


def save_heatmap_png(hm: torch.Tensor, path: Path) -> None:
    """Save a heatmap ``[1,H,W]`` or ``[B,1,H,W]`` to a colored PNG.

    Falls back to grayscale if matplotlib is not available.
    """

    import numpy as np
    try:  # prefer matplotlib for colormap
        import matplotlib.pyplot as plt  # type: ignore
        arr = hm.squeeze().detach().cpu().numpy()
        plt.imsave(path.as_posix(), arr, cmap="jet")
    except Exception:
        x = hm.squeeze().detach().cpu()
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = (x * 255.0).byte().numpy()
        Image.fromarray(x, mode="L").save(path)


def save_overlay_from_tensor(img_t: torch.Tensor, hm: torch.Tensor, path: Path,
                             alpha: float = 0.5,
                             denormalize: bool = True,
                             mean: Sequence[float] = IMAGENET_MEAN,
                             std: Sequence[float] = IMAGENET_STD) -> None:
    """Create an overlay of a heatmap on top of an image tensor and save.

    Args:
        img_t: Image tensor ``[C,H,W]`` (normalized or 0..1).
        hm: Heatmap tensor ``[1,Hh,Wh]``.
        path: Output path for the saved overlay PNG.
        alpha: Blending factor for the heatmap overlay.
        denormalize: Whether to apply ImageNet de-normalization.
    """

    base = tensor_to_pil(img_t, denormalize=denormalize, mean=mean, std=std).convert("RGBA")
    x = hm.squeeze().detach().cpu()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = (x * 255.0).byte().numpy()
    
    # Resize heatmap to match input image size to handle scale mismatch
    hm_img = Image.fromarray(x, mode="L").resize(base.size, Image.BILINEAR)
    
    # Create RGBA overlay with colormap (jet-like colors)
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.cm as cm  # type: ignore
        
        # Apply jet colormap to heatmap
        hm_normalized = hm.squeeze().detach().cpu().numpy()
        hm_normalized = (hm_normalized - hm_normalized.min()) / (hm_normalized.max() - hm_normalized.min() + 1e-8)
        hm_colored = cm.jet(hm_normalized)[:, :, :3]  # Take RGB only, drop alpha
        hm_colored = (hm_colored * 255.0).astype('uint8')
        hm_colored_img = Image.fromarray(hm_colored, mode="RGB").resize(base.size, Image.BILINEAR)
        
        # Create overlay with proper alpha channel
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        overlay.paste(hm_colored_img, (0, 0))
        
        # Apply alpha blending
        blended = Image.alpha_composite(base, overlay)
        blended = Image.blend(base, blended, alpha=alpha)
        
    except Exception:
        # Fallback to grayscale overlay if matplotlib is not available
        overlay = Image.merge("RGBA", (hm_img, Image.new("L", hm_img.size), Image.new("L", hm_img.size), hm_img))
        blended = Image.blend(base, overlay, alpha=alpha)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(path)
