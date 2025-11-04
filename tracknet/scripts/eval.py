"""Evaluation script for TrackNet.

Builds config, loads a checkpoint, runs evaluation on the val split (or train
split if val is not defined), and prints aggregate metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tracknet.models import build_model
from tracknet.training import (
    HeatmapLossConfig,
    build_heatmap_loss,
    heatmap_argmax_coords,
    l2_error,
    pck_at_r,
    visible_from_mask,
)
from tracknet.training.trainer import Trainer
from tracknet.utils.config import add_config_cli_arguments, build_cfg


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="TrackNet evaluation entrypoint")
    add_config_cli_arguments(parser)
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint .pt file"
    )
    known, unknown = parser.parse_known_args(argv)
    return known, unknown


def _find_best_ckpt(cfg) -> Path | None:
    ckpt_dir = Path(cfg.runtime.ckpt_dir)
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob("best_*.pt"))
    return cands[-1] if cands else None


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

    # Build dataloaders via Trainer helper
    trainer = Trainer(cfg)
    train_loader, val_loader = trainer._build_dataloaders()
    loader = val_loader or train_loader
    if loader is None:
        print("No data split available for evaluation.")
        return 1

    # Build model and loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    loss = build_heatmap_loss(
        HeatmapLossConfig(name=str(cfg.training.get("loss", {}).get("name", "mse")))
    )

    # Load checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _find_best_ckpt(cfg)
    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"], strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint provided/found. Evaluating current weights.")

    # Precision handling (eval only)
    prec = str(cfg.training.get("precision", "fp32")).lower()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if prec == "fp16" and device_type == "cuda":
        autocast_dtype = torch.float16
    elif prec == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None

    # Eval loop
    model.eval()
    tot_loss = 0.0
    n = 0
    tot_l2 = 0.0
    tot_pck = 0.0
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device)
            targets = batch["heatmaps"].to(device)
            masks = batch["masks"].to(device)
            if autocast_dtype is not None:
                ctx = torch.autocast(device_type=device_type, dtype=autocast_dtype)  # type: ignore[arg-type]
            else:
                from contextlib import nullcontext

                ctx = nullcontext()
            with ctx:
                preds = model(images)
            loss_val = loss(preds, targets, masks)
            tot_loss += float(loss_val.cpu())
            # Argmax-based metrics on heatmap space
            pred_xy = heatmap_argmax_coords(preds)
            tgt_xy = heatmap_argmax_coords(targets)
            vis = visible_from_mask(masks)
            tot_l2 += float(l2_error(pred_xy, tgt_xy, vis).cpu())
            tot_pck += float(pck_at_r(pred_xy, tgt_xy, vis, r=3.0).cpu())
            n += 1
    if n == 0:
        print("Empty loader.")
        return 1
    print(f"eval_loss={tot_loss/n:.4f} l2={tot_l2/n:.3f} pck@3={tot_pck/n:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
