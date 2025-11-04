"""Simple prediction script for a single image.

Loads a model (per config), an optional checkpoint, runs forward on one image,
and saves the predicted heatmap and an overlay visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from tracknet.datasets.utils.augmentations import to_tensor_and_normalize
from tracknet.models import build_model
from tracknet.utils.config import add_config_cli_arguments, build_cfg


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="TrackNet prediction entrypoint")
    add_config_cli_arguments(parser)
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint .pt file"
    )
    parser.add_argument(
        "--outdir", type=str, default=None, help="Output directory for predictions"
    )
    known, unknown = parser.parse_known_args(argv)
    return known, unknown


def _find_best_ckpt(cfg: Any) -> Path | None:
    ckpt_dir = Path(cfg.runtime.ckpt_dir)
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob("best_*.pt"))
    return cands[-1] if cands else None


def _save_heatmap_png(hm: torch.Tensor, path: Path) -> None:
    import matplotlib.pyplot as plt  # optional; fallback to PIL if unavailable

    try:
        arr = hm.squeeze().cpu().numpy()
        plt.imsave(path.as_posix(), arr, cmap="jet")
    except Exception:
        # Fallback: scale to 0-255 and save grayscale via PIL
        arr = hm.squeeze().cpu()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255.0).byte().numpy()
        Image.fromarray(arr, mode="L").save(path)


def _save_overlay(
    img: Image.Image, hm: torch.Tensor, path: Path, alpha: float = 0.5
) -> None:
    base = img.convert("RGBA")
    h, w = hm.shape[-2:]
    # Normalize heatmap to 0..255
    x = hm.squeeze().cpu()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = (x * 255.0).byte().numpy()
    # Create red overlay
    Image.new("RGBA", base.size, (0, 0, 0, 0))
    hm_img = Image.fromarray(x, mode="L").resize(base.size, Image.BILINEAR)
    r = Image.merge(
        "RGBA",
        (hm_img, Image.new("L", hm_img.size), Image.new("L", hm_img.size), hm_img),
    )
    blended = Image.blend(base, r, alpha=alpha)
    blended.save(path)


def main(argv: list[str] | None = None) -> int:
    import sys

    if argv is None:
        argv = sys.argv[1:]

    args, overrides = parse_args(argv)
    cfg = build_cfg(
        data_name=args.data_name,
        model_name=args.model_name,
        training_name=args.training_name,
        overrides=overrides,
        seed=args.seed,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else _find_best_ckpt(cfg)
    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"], strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint provided/found. Using current weights.")

    img_path = Path(args.image)
    img = Image.open(img_path).convert("RGB")
    # For predict: keep [0,1] tensor (do not normalize) to be safe for HF processor path
    x = to_tensor_and_normalize(img, normalize=False).unsqueeze(0).to(device)

    # Precision handling (inference)
    prec = str(cfg.training.get("precision", "fp32")).lower()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if prec == "fp16" and device_type == "cuda":
        autocast_dtype = torch.float16
    elif prec == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None

    model.eval()
    with torch.inference_mode():
        if autocast_dtype is not None:
            ctx = torch.autocast(device_type=device_type, dtype=autocast_dtype)
        else:
            from contextlib import nullcontext

            ctx = nullcontext()
        with ctx:
            hm = model(x)  # [1,1,Hh,Wh]

    outdir = (
        Path(args.outdir)
        if args.outdir
        else Path(cfg.runtime.output_root) / "predictions"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    hm_path = outdir / f"{img_path.stem}_heatmap.png"
    ov_path = outdir / f"{img_path.stem}_overlay.png"
    _save_heatmap_png(hm, hm_path)
    _save_overlay(img, hm, ov_path)
    print(f"Saved: {hm_path}\nSaved: {ov_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
