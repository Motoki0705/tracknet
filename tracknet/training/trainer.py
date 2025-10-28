"""Training orchestration for TrackNet.

This module wires together datasets, models, losses, optimizer, scheduler, and
callbacks to run a minimal but complete training/validation loop.

Key features:
- Supports frame and (basic) sequence modes based on cfg.data.sequence.enabled.
- Builds either ViT+Upsampling or ConvNeXt+FPN models depending on model config.
- Uses masked heatmap losses (default MSE) and simple metrics.
- Optional AMP, gradient clipping, early stopping, and checkpointing.

Usage:
    from tracknet.training.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.train()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from omegaconf import DictConfig

from tracknet.datasets import (
    PreprocessConfig,
    TrackNetFrameDataset,
    TrackNetFrameDatasetConfig,
    TrackNetSequenceDataset,
    TrackNetSequenceDatasetConfig,
    collate_frames,
    collate_sequences,
)
from tracknet.models import (
    ViTBackbone, ViTBackboneConfig,
    UpsamplingDecoder,
    ConvNeXtBackbone, ConvNeXtBackboneConfig,
    FPNDecoder, FPNDecoderConfig,
    HeatmapHead,
)
from tracknet.training import (
    HeatmapLossConfig, build_heatmap_loss,
    heatmap_argmax_coords, visible_from_mask,
)


class HeatmapModel(nn.Module):
    """Unified model interface for ViT+Upsampling and ConvNeXt+FPN variants.

    The variant is selected by the presence of ``fpn`` or ``decoder`` in the
    ``model_cfg``.
    """

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__()
        hm_w, hm_h = int(model_cfg.heatmap.size[0]), int(model_cfg.heatmap.size[1])
        out_size = (hm_h, hm_w)  # NCHW expects (H,W)

        if hasattr(model_cfg, "fpn"):
            bb = ConvNeXtBackbone(
                ConvNeXtBackboneConfig(
                    pretrained_model_name=str(model_cfg.get("pretrained_model_name", "facebook/dinov3-convnext-base-pretrain-lvd1689m")),
                    use_pretrained=bool(model_cfg.get("backbone", {}).get("use_pretrained", False)),
                    tv_model=str(model_cfg.get("backbone", {}).get("tv_model", "convnext_tiny")),
                )
            )
            # Probe in_channels via a dummy pass with a small tensor
            self.backbone = bb
            self.variant = "convnext_fpn"
            # in_channels will be inferred on first forward call
            self.fpn_cfg = FPNDecoderConfig(
                lateral_dim=int(model_cfg.fpn.get("lateral_dim", 256)),
                use_p2=bool(model_cfg.fpn.get("use_p2", False)),
                fuse=str(model_cfg.fpn.get("fuse", "sum")),
                out_size=out_size,
            )
            self.decoder = None  # lazy
            self.head = HeatmapHead(self.fpn_cfg.lateral_dim)
        elif hasattr(model_cfg, "decoder"):
            self.variant = "vit_upsample"
            bb = ViTBackbone(
                ViTBackboneConfig(
                    pretrained_model_name=str(model_cfg.get("pretrained_model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m")),
                    # Default to fallback in offline envs; advanced users can extend configs
                    use_pretrained=bool(model_cfg.get("backbone", {}).get("use_pretrained", False)),
                )
            )
            self.backbone = bb
            channels = [int(c) for c in model_cfg.decoder.channels]
            upfactors = [int(u) for u in model_cfg.decoder.upsample]
            self.decoder = UpsamplingDecoder(channels, upfactors, out_size=out_size)
            self.head = HeatmapHead(channels[-1])
        else:
            raise ValueError("model config must contain either 'decoder' (ViT) or 'fpn' (ConvNeXt)")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.variant == "vit_upsample":
            tokens = self.backbone(images)  # [B,Hp,Wp,C]
            feat = self.decoder(tokens)
            out = self.head(feat)
            return out
        else:
            feats = self.backbone(images)  # list of [B,C_i,H_i,W_i]
            if self.decoder is None:
                in_chs = [int(f.shape[1]) for f in feats]
                self.decoder = FPNDecoder(in_chs, self.fpn_cfg)
            feat = self.decoder(feats)
            out = self.head(feat)
            return out


@dataclass
class Trainer:
    """Trainer orchestrates data, model, loss, and optimization."""

    cfg: DictConfig

    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Data -----------------
    def _build_dataloaders(self) -> tuple[DataLoader, Optional[DataLoader]]:
        dcfg = self.cfg.data
        mcfg = self.cfg.model

        pp = PreprocessConfig(
            resize=None if dcfg.preprocess.get("resize", None) in (None, "null") else tuple(dcfg.preprocess.resize),
            normalize=bool(dcfg.preprocess.get("normalize", True)),
            flip_prob=float(dcfg.preprocess.get("flip_prob", 0.0)),
        )
        heatmap_size = (int(mcfg.heatmap.size[0]), int(mcfg.heatmap.size[1]))
        sigma = float(mcfg.heatmap.sigma)

        if bool(dcfg.sequence.get("enabled", False)):
            length = int(dcfg.sequence.get("length", 3))
            stride = int(dcfg.sequence.get("stride", 1))
            train_ds = TrackNetSequenceDataset(
                TrackNetSequenceDatasetConfig(
                    root=str(dcfg.root),
                    games=list(dcfg.split.train_games),
                    length=length,
                    stride=stride,
                    preprocess=pp,
                )
            )
            val_ds = TrackNetSequenceDataset(
                TrackNetSequenceDatasetConfig(
                    root=str(dcfg.root),
                    games=list(dcfg.split.val_games),
                    length=length,
                    stride=stride,
                    preprocess=pp,
                )
            ) if dcfg.split.get("val_games") else None

            collate_fn = lambda b: collate_sequences(b, heatmap_size=heatmap_size, sigma=sigma)
        else:
            train_ds = TrackNetFrameDataset(
                TrackNetFrameDatasetConfig(
                    root=str(dcfg.root),
                    games=list(dcfg.split.train_games),
                    preprocess=pp,
                )
            )
            val_ds = TrackNetFrameDataset(
                TrackNetFrameDatasetConfig(
                    root=str(dcfg.root),
                    games=list(dcfg.split.val_games),
                    preprocess=pp,
                )
            ) if dcfg.split.get("val_games") else None

            collate_fn = lambda b: collate_frames(b, heatmap_size=heatmap_size, sigma=sigma)

        bs = int(self.cfg.training.batch_size)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn) if val_ds else None
        return train_loader, val_loader

    # ---------------- Model & Optim -----------------
    def _build_model(self) -> nn.Module:
        model = HeatmapModel(self.cfg.model)
        return model.to(self._device())

    def _build_loss(self) -> nn.Module:
        tcfg = self.cfg.training
        lname = str(tcfg.get("loss", {}).get("name", "mse")) if hasattr(tcfg, "loss") else "mse"
        alpha = float(tcfg.get("loss", {}).get("alpha", 2.0)) if hasattr(tcfg, "loss") else 2.0
        beta = float(tcfg.get("loss", {}).get("beta", 4.0)) if hasattr(tcfg, "loss") else 4.0
        return build_heatmap_loss(HeatmapLossConfig(name=lname, alpha=alpha, beta=beta))

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        ocfg = self.cfg.training.optimizer
        name = str(ocfg.get("name", "adamw")).lower()
        lr = float(ocfg.get("lr", 5e-4))
        wd = float(ocfg.get("weight_decay", 0.0))
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if name == "sgd":
            momentum = float(ocfg.get("momentum", 0.9))
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        raise ValueError(f"Unsupported optimizer: {name}")

    def _build_scheduler(self, optim: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:  # type: ignore[attr-defined]
        scfg = self.cfg.training.get("scheduler", {})
        name = str(scfg.get("name", "cosine")).lower()
        epochs = int(self.cfg.training.get("epochs", 1))
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))
        return None

    # ---------------- Training loop -----------------
    def train(self) -> None:
        device = self._device()
        train_loader, val_loader = self._build_dataloaders()
        model = self._build_model()
        criterion = self._build_loss()
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)

        epochs = int(self.cfg.training.get("epochs", 1))
        grad_clip = float(self.cfg.training.get("grad_clip", 0.0))
        use_amp = bool(self.cfg.training.get("amp", False))
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # type: ignore[attr-defined]

        # Limits for quick smoke tests
        limit_train = int(self.cfg.training.get("limit_train_batches", 0))
        limit_val = int(self.cfg.training.get("limit_val_batches", 0))

        # Checkpoint paths
        ckpt_dir = Path(self.cfg.runtime.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_val = math.inf

        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            train_loss = 0.0
            num_batches = 0
            for bi, batch in enumerate(train_loader, start=1):
                images = batch["images"].to(device)
                targets = batch["heatmaps"].to(device)
                masks = batch["masks"].to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):  # type: ignore[attr-defined]
                    outputs = model(images)
                    loss = criterion(outputs, targets, masks)
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                train_loss += float(loss.detach().cpu())
                num_batches += 1
                if limit_train and bi >= limit_train:
                    break

            train_loss /= max(1, num_batches)

            # Validation
            val_loss = None
            if val_loader is not None:
                model.eval()
                vloss = 0.0
                vnum = 0
                with torch.inference_mode():
                    for vi, batch in enumerate(val_loader, start=1):
                        images = batch["images"].to(device)
                        targets = batch["heatmaps"].to(device)
                        masks = batch["masks"].to(device)
                        outputs = model(images)
                        loss = criterion(outputs, targets, masks)
                        vloss += float(loss.detach().cpu())
                        vnum += 1
                        if limit_val and vi >= limit_val:
                            break
                val_loss = vloss / max(1, vnum)

            # Scheduler step per epoch
            if scheduler is not None:
                try:
                    scheduler.step(val_loss if val_loss is not None else train_loss)
                except TypeError:
                    scheduler.step()

            dt = time.time() - t0
            if val_loss is None:
                print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} time={dt:.1f}s")
            else:
                print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={dt:.1f}s")

            # Save best checkpoint by val loss if available else train loss
            score = val_loss if val_loss is not None else train_loss
            if score < best_val:
                best_val = score
                path = ckpt_dir / f"best_{self.cfg.runtime.run_id}.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "cfg": self.cfg,
                }, path)
                print(f"Saved best checkpoint to {path}")

