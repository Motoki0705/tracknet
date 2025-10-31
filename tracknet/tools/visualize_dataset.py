"""Visualize TrackNet dataset with heatmap overlay.

This tool instantiates TrackNetFrameDataset from OmegaConf configuration
and visualizes images with heatmap overlays to verify data loading and
heatmap generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from tracknet.datasets.tracknet_frame import TrackNetFrameDataset, TrackNetFrameDatasetConfig
from tracknet.datasets.utils.collate import collate_frames


def load_dataset_from_config(config_path: str | Path) -> TrackNetFrameDataset:
    """Load TrackNetFrameDataset from OmegaConf configuration.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Instantiated TrackNetFrameDataset.
    """
    cfg = OmegaConf.load(config_path)
    
    # Extract dataset configuration
    dataset_cfg = TrackNetFrameDatasetConfig(
        root=cfg.dataset.root,
        games=cfg.dataset.games,
        preprocess=cfg.dataset.get("preprocess", None),
    )
    
    return TrackNetFrameDataset(dataset_cfg)


def generate_heatmap_for_sample(
    sample: dict, heatmap_size: Tuple[int, int] = (64, 64), sigma: float = 2.0
) -> torch.Tensor:
    """Generate a heatmap for a single sample.

    Args:
        sample: Sample dictionary from TrackNetFrameDataset.
        heatmap_size: Target heatmap size as (width, height).
        sigma: Gaussian sigma for heatmap generation.

    Returns:
        Heatmap tensor of shape [1, H, W].
    """
    # Log original information
    orig_size = sample["meta"]["size"]  # Original image size
    coord = sample["coord"]
    print(f"[DEBUG] Original image size: {orig_size}")
    print(f"[DEBUG] Original coordinate: {coord}")
    
    # Current tensor size (after preprocessing)
    current_img_size = (sample["image"].shape[2], sample["image"].shape[1])  # (W, H)
    print(f"[DEBUG] Current tensor size: {current_img_size}")
    print(f"[DEBUG] Current coordinate: {coord}")
    
    # Create a modified sample with correct size for heatmap generation
    modified_sample = sample.copy()
    modified_sample["meta"]["size"] = current_img_size
    
    print(f"[DEBUG] Heatmap size target: {heatmap_size}")
    
    batch = [modified_sample]
    collated = collate_frames(batch, heatmap_size, sigma)
    heatmap = collated["heatmaps"][0]  # [1, H, W]
    
    print(f"[DEBUG] Generated heatmap shape: {heatmap.shape}")
    return heatmap


def overlay_heatmap_on_image(
    image: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.4
) -> np.ndarray:
    """Overlay heatmap on image for visualization.

    Args:
        image: Image tensor of shape [C, H, W] in [0, 1] range.
        heatmap: Heatmap tensor of shape [1, H, W] or [H, W].
        alpha: Transparency factor for heatmap overlay.

    Returns:
        Combined image as numpy array of shape [H, W, 3].
    """
    print(f"[DEBUG] Input image shape: {image.shape}")
    print(f"[DEBUG] Input heatmap shape: {heatmap.shape}")
    
    # Convert tensors to numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    img_np = np.clip(img_np, 0, 1)  # Ensure valid range
    print(f"[DEBUG] Converted image shape: {img_np.shape}")
    
    # Handle heatmap format
    if heatmap.dim() == 3:
        heatmap_np = heatmap[0].cpu().numpy()  # [H, W]
    else:
        heatmap_np = heatmap.cpu().numpy()  # [H, W]
    print(f"[DEBUG] Converted heatmap shape: {heatmap_np.shape}")
    
    # Normalize heatmap to [0, 1]
    if heatmap_np.max() > 0:
        heatmap_np = heatmap_np / heatmap_np.max()
    print(f"[DEBUG] Heatmap range: [{heatmap_np.min():.3f}, {heatmap_np.max():.3f}]")
    
    # Resize heatmap to match image size
    import matplotlib.pyplot as plt
    from PIL import Image
    
    print(f"[DEBUG] Resizing heatmap from {heatmap_np.shape} to {img_np.shape[:2]}")
    
    # Convert heatmap to PIL Image and resize
    heatmap_pil = Image.fromarray((heatmap_np * 255).astype(np.uint8), mode='L')
    heatmap_resized = heatmap_pil.resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_resized) / 255.0  # Back to [0, 1]
    print(f"[DEBUG] Resized heatmap shape: {heatmap_resized.shape}")
    
    # Create colormap (jet-like for visibility)
    import matplotlib.cm as cm
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # [H, W, 3]
    print(f"[DEBUG] Colored heatmap shape: {heatmap_colored.shape}")
    
    # Overlay
    combined = (1 - alpha) * img_np + alpha * heatmap_colored
    print(f"[DEBUG] Final combined shape: {combined.shape}")
    
    return np.clip(combined, 0, 1)


def visualize_samples(
    dataset: TrackNetFrameDataset,
    num_samples: int = 8,
    heatmap_size: Tuple[int, int] = (64, 64),
    sigma: float = 2.0,
    save_path: str | Path | None = None,
) -> None:
    """Visualize dataset samples with heatmap overlays.

    Args:
        dataset: TrackNetFrameDataset instance.
        num_samples: Number of samples to visualize.
        heatmap_size: Target heatmap size as (width, height).
        sigma: Gaussian sigma for heatmap generation.
        save_path: Optional path to save the visualization.
    """
    num_samples = min(num_samples, len(dataset))
    
    # Create subplot grid
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i in range(num_samples):
        print(f"\n[DEBUG] === Processing sample {i} ===")
        sample = dataset[i]
        print(f"[DEBUG] Sample {i} metadata: {sample['meta']}")
        
        heatmap = generate_heatmap_for_sample(sample, heatmap_size, sigma)
        combined = overlay_heatmap_on_image(sample["image"], heatmap)
        
        ax = axes[i]
        ax.imshow(combined)
        ax.set_title(
            f"Sample {i}\n"
            f"Coord: ({sample['coord'][0]:.1f}, {sample['coord'][1]:.1f})\n"
            f"Vis: {sample['visibility']}\n"
            f"Game: {sample['meta']['game']}, Clip: {sample['meta']['clip']}"
        )
        ax.axis("off")
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main() -> None:
    """Main function for dataset visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize TrackNet dataset with heatmap overlays"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the dataset configuration file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples to visualize (default: 8)",
    )
    parser.add_argument(
        "--heatmap-size",
        type=int,
        nargs=2,
        default=[64, 64],
        help="Heatmap size as width height (default: 64 64)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma for heatmap generation (default: 2.0)",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Path to save the visualization image",
    )
    
    args = parser.parse_args()
    
    # Load dataset from config
    print(f"Loading dataset from config: {args.config}")
    dataset = load_dataset_from_config(args.config)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Visualize samples
    heatmap_size = tuple(args.heatmap_size)
    print(f"Visualizing {args.num_samples} samples with heatmap size {heatmap_size}")
    
    visualize_samples(
        dataset=dataset,
        num_samples=args.num_samples,
        heatmap_size=heatmap_size,
        sigma=args.sigma,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
