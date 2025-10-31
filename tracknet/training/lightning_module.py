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
        params = [p for p in self.model.parameters() if p.requires_grad]

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
        if sname == "cosine":
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

