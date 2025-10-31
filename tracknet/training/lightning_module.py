"""PyTorch LightningModule for TrackNet heatmap training.

This module encapsulates the model, loss, optimizer, and train/val steps
while delegating core architecture and datasets to existing project modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from tracknet.models import build_model
from tracknet.training import HeatmapLossConfig, build_heatmap_loss
from tracknet.utils.logging import save_image_from_tensor, save_overlay_from_tensor


@dataclass
class OptimConfig:
    """Lightweight optimizer config extracted from cfg for clarity."""

    name: str
    lr: float
    weight_decay: float
    momentum: float | None = None


class PLHeatmapModule(pl.LightningModule):
    """LightningModule orchestrating model, loss, and optimization.

    Args:
        cfg: Unified OmegaConf configuration.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters({"run_id": str(cfg.runtime.run_id)})
        self.cfg = cfg
        self.model = build_model(cfg.model)

        # Loss
        tcfg = cfg.training
        lname = str(tcfg.get("loss", {}).get("name", "mse")) if hasattr(tcfg, "loss") else "mse"
        alpha = float(tcfg.get("loss", {}).get("alpha", 2.0)) if hasattr(tcfg, "loss") else 2.0
        beta = float(tcfg.get("loss", {}).get("beta", 4.0)) if hasattr(tcfg, "loss") else 4.0
        self.criterion = build_heatmap_loss(HeatmapLossConfig(name=lname, alpha=alpha, beta=beta))

        # Freeze schedule: freeze backbone for the first N epochs, then unfreeze
        self._freeze_epochs: int = int(self.cfg.training.get("backbone_freeze_epochs", 0))
        # Detect current frozen state (may be frozen by model config or by us below)
        self._backbone_frozen: bool = False
        try:
            if hasattr(self.model, "backbone"):
                self._backbone_frozen = any((not p.requires_grad) for p in self.model.backbone.parameters())
        except Exception:
            self._backbone_frozen = False

        # If a freeze schedule is defined, enforce initial freeze regardless of model default
        if self._freeze_epochs > 0 and hasattr(self.model, "backbone"):
            self._set_backbone_requires_grad(False)
            self._backbone_frozen = True

        # Preview saving state
        self._preview_saved_epoch: int = -1

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(images)

    # -------------------------- Training / Validation --------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        images = batch["images"]
        targets = batch["heatmaps"]
        masks = batch["masks"]
        outputs = self(images)
        loss = self.criterion(outputs, targets, masks)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:  # type: ignore[override]
        images = batch["images"]
        targets = batch["heatmaps"]
        masks = batch["masks"]
        outputs = self(images)
        loss = self.criterion(outputs, targets, masks)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))

        # Save small preview for the first batch of each epoch
        if batch_idx == 0 and self.current_epoch != self._preview_saved_epoch:
            try:
                self._save_validation_preview(images, outputs)
                self._preview_saved_epoch = self.current_epoch
            except Exception:
                # Best-effort; previews are not critical to training
                pass
        return loss

    # -------------------------- Optimizer / Scheduler --------------------------
    def configure_optimizers(self):  # type: ignore[override]
        ocfg = self.cfg.training.optimizer
        name = str(ocfg.get("name", "adamw")).lower()
        lr = float(ocfg.get("lr", 5e-4))
        wd = float(ocfg.get("weight_decay", 0.0))
        # Include all parameters so that late unfreeze takes effect without
        # reconstructing the optimizer.
        params = list(self.model.parameters())

        if name == "adamw":
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        elif name == "sgd":
            momentum = float(ocfg.get("momentum", 0.9))
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

        scfg = self.cfg.training.get("scheduler", {})
        sname = str(scfg.get("name", "cosine")).lower()
        epochs = int(self.cfg.training.get("epochs", 1))
        warmup_epochs = int(scfg.get("warmup_epochs", 0))
        warmup_start = float(scfg.get("warmup_start_factor", 0.1))
        if sname == "cosine":
            # Optional linear warmup followed by cosine decay
            if warmup_epochs > 0:
                total_after = max(1, epochs - warmup_epochs)
                linear = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=max(1e-6, min(1.0, warmup_start)),
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_after,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[linear, cosine],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val/loss",
                },
            }
        return optimizer

    # -------------------------- Utilities --------------------------
    def _save_validation_preview(self, images: torch.Tensor, outputs: torch.Tensor) -> None:
        """Save a small set of input/overlay previews under the log directory.

        Args:
            images: Batch input images (``[B, 3, H, W]``).
            outputs: Predicted heatmaps (``[B, 1, H, W]``).
        """

        base_dir = Path(self.cfg.runtime.log_dir) / self.cfg.runtime.run_id
        overlay_dir = base_dir / "overlays" / f"epoch{self.current_epoch:03d}"
        image_dir = base_dir / "inputs" / f"epoch{self.current_epoch:03d}"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        bsz = images.shape[0]
        sample = min(4, bsz)
        denorm = bool(self.cfg.data.preprocess.get("normalize", True))
        for idx in range(sample):
            img_t = images[idx].detach().cpu()
            hm_t = outputs[idx].detach().cpu()
            save_image_from_tensor(
                img_t,
                image_dir / f"sample_{idx:02d}.png",
                denormalize=denorm,
            )
            save_overlay_from_tensor(
                img_t,
                hm_t,
                overlay_dir / f"sample_{idx:02d}.png",
                denormalize=denorm,
            )

    # -------------------------- Epoch hooks --------------------------
    def on_train_epoch_start(self) -> None:  # type: ignore[override]
        """Handle scheduled backbone (un)freezing at epoch boundaries.

        Freezes the backbone for the first ``training.backbone_freeze_epochs``
        epochs, then unfreezes it afterward.
        """
        if not hasattr(self.model, "backbone"):
            return

        if self._freeze_epochs <= 0:
            return

        if self.current_epoch < self._freeze_epochs:
            # Ensure frozen during warmup period
            if not self._backbone_frozen:
                self._set_backbone_requires_grad(False)
                self._backbone_frozen = True
                self.print(f"[Backbone] Frozen (epoch {self.current_epoch} < {self._freeze_epochs})")
        else:
            # Unfreeze once warmup period is over
            if self._backbone_frozen:
                self._set_backbone_requires_grad(True)
                self._backbone_frozen = False
                self.print(f"[Backbone] Unfrozen at epoch {self.current_epoch}")

    def _set_backbone_requires_grad(self, flag: bool) -> None:
        try:
            for p in self.model.backbone.parameters():
                p.requires_grad = flag
        except Exception:
            pass
