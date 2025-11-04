"""PyTorch LightningModule for TrackNet heatmap training.

This module encapsulates the model, loss, optimizer, and train/val steps
while delegating core architecture and datasets to existing project modules.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from tracknet.models import build_model
from tracknet.training import HeatmapLossConfig, build_heatmap_loss
from tracknet.utils.logging import save_overlay_from_tensor


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
        lname = (
            str(tcfg.get("loss", {}).get("name", "mse"))
            if hasattr(tcfg, "loss")
            else "mse"
        )
        alpha = (
            float(tcfg.get("loss", {}).get("alpha", 2.0))
            if hasattr(tcfg, "loss")
            else 2.0
        )
        beta = (
            float(tcfg.get("loss", {}).get("beta", 4.0))
            if hasattr(tcfg, "loss")
            else 4.0
        )
        self.criterion = build_heatmap_loss(
            HeatmapLossConfig(name=lname, alpha=alpha, beta=beta)
        )

        # Freeze schedule: freeze backbone for the first N epochs, then unfreeze
        self._freeze_epochs: int = int(
            self.cfg.training.get("backbone_freeze_epochs", 0)
        )
        # Detect current frozen state (may be frozen by model config or by us below)
        self._backbone_frozen: bool = False
        try:
            if hasattr(self.model, "backbone"):
                self._backbone_frozen = any(
                    (not p.requires_grad) for p in self.model.backbone.parameters()
                )
        except Exception:
            self._backbone_frozen = False

        # If a freeze schedule is defined, enforce initial freeze regardless of model default
        if self._freeze_epochs > 0 and hasattr(self.model, "backbone"):
            self._set_backbone_requires_grad(False)
            self._backbone_frozen = True

        # === NEW: 自動マイクロバッチ設定 ===
        tcfg = cfg.training
        self._auto_mb: bool = bool(tcfg.get("adaptive_micro_batch", True))
        self._min_mb: int = int(tcfg.get("min_micro_batch_size", 1))
        self._backoff: int = int(tcfg.get("mb_backoff_factor", 2))  # 2で半減
        self._oom_retries: int = int(tcfg.get("oom_retries", 3))
        self._fixed_mb: int = int(tcfg.get("micro_batch_size", 0))  # >0なら固定
        self._clip_norm: float = float(tcfg.get("grad_clip_norm", 0.0))

        # ランタイムに決まる"現在のマイクロバッチサイズ"（OOM後に確定）
        self._runtime_mb: int | None = None

        # 以降は手動最適化で統一（通常時は1チャンク=元バッチと同義）
        self.automatic_optimization = False
        self._preview_saved_epoch: int = -1

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(images)

    # -------------------------- Training --------------------------
    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:  # type: ignore[override]
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        opt.zero_grad(set_to_none=True)

        images, targets, masks = batch["images"], batch["heatmaps"], batch["masks"]
        B = images.size(0)

        # 1) 固定マイクロバッチ or 自動探索済みならそれを使う
        if self._fixed_mb > 0:
            mb = min(self._fixed_mb, B)
            loss = self._run_micro_batches(opt, images, targets, masks, mb)
            return loss

        if self._auto_mb and self._runtime_mb is not None:
            mb = min(self._runtime_mb, B)
            loss = self._run_micro_batches(opt, images, targets, masks, mb)
            return loss

        # 2) まずは"分割なし"（mb=B）で試す → OOMなら自動で縮めて再試行
        try:
            loss = self._run_micro_batches(
                opt, images, targets, masks, B
            )  # 等価に1回更新
            if self._auto_mb:
                self._runtime_mb = B  # 余裕があることを記録
            return loss
        except RuntimeError as e:
            if not self._auto_mb or "out of memory" not in str(e).lower():
                raise
            self._handle_oom(opt)

        # 3) バックオフしながら再試行（B, B/2, B/4, ...）
        mb = max(self._min_mb, B // self._backoff)
        retries = self._oom_retries
        while True:
            try:
                loss = self._run_micro_batches(opt, images, targets, masks, mb)
                self._runtime_mb = mb  # 成功サイズを記録（以後も使用）
                return loss
            except RuntimeError as e:
                if (
                    "out of memory" not in str(e).lower()
                    or retries <= 0
                    or mb <= self._min_mb
                ):
                    raise
                self._handle_oom(opt)
                mb = max(self._min_mb, mb // self._backoff)
                retries -= 1

    # マイクロバッチで forward/backward を回し、1回だけ step
    def _run_micro_batches(self, opt, images, targets, masks, mb: int) -> torch.Tensor:
        B = images.size(0)
        num_micro = math.ceil(B / mb)
        total_loss = images.new_tensor(0.0)

        # AMP 互換の autocast（PL 2.x：precision plugin が提供）
        cm = getattr(self, "autocast_context_manager", None)
        cm = cm() if callable(cm) else contextlib.nullcontext()

        for i in range(0, B, mb):
            xb = images[i : i + mb]
            yb = targets[i : i + mb]
            mbb = masks[i : i + mb]
            with cm:
                out = self(xb)
                loss_mb = self.criterion(out, yb, mbb) / num_micro
                self.manual_backward(loss_mb)  # 勾配だけ貯める
                total_loss += loss_mb.detach()

        if self._clip_norm and self._clip_norm > 0:
            self.clip_gradients(
                opt, gradient_clip_val=self._clip_norm, gradient_clip_algorithm="norm"
            )

        opt.step()
        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
        )
        return total_loss

    def _handle_oom(self, opt) -> None:
        # OOM後は一度勾配を解放してキャッシュを掃除
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # -------------------------- Validation（必要なら同様に自動分割） --------------------------
    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor | None:  # type: ignore[override]
        images, targets, masks = batch["images"], batch["heatmaps"], batch["masks"]
        B = images.size(0)

        # まずは一括で試す → OOMなら分割
        try:
            outputs = self(images)
            loss = self.criterion(outputs, targets, masks)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            self._handle_oom(opt=None)
            mb = max(self._min_mb, (self._runtime_mb or B) // self._backoff)
            total = images.new_tensor(0.0)
            for i in range(0, B, mb):
                out = self(images[i : i + mb])
                total += self.criterion(
                    out, targets[i : i + mb], masks[i : i + mb]
                ).detach()
            loss = total / math.ceil(B / mb)

        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B
        )

        if batch_idx == 0 and self.current_epoch != self._preview_saved_epoch:
            try:
                out_prev = outputs if "outputs" in locals() else self(images[:4])
                self._save_validation_preview(images[:4], out_prev[:4], targets[:4])
                self._preview_saved_epoch = self.current_epoch
            except Exception:
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
            optimizer = torch.optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=wd
            )
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
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, epochs)
                )

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
    def _save_validation_preview(
        self, images: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Save a small set of prediction and target overlay previews under the log directory.

        Args:
            images: Batch input images (``[B, 3, H, W]``).
            outputs: Predicted heatmaps (``[B, 1, H, W]``).
            targets: Target heatmaps (``[B, 1, H, W]``).
        """

        base_dir = Path(self.cfg.runtime.log_dir) / self.cfg.runtime.run_id
        pred_overlay_dir = base_dir / "overlays_pred" / f"epoch{self.current_epoch:03d}"
        target_overlay_dir = (
            base_dir / "overlays_target" / f"epoch{self.current_epoch:03d}"
        )
        pred_overlay_dir.mkdir(parents=True, exist_ok=True)
        target_overlay_dir.mkdir(parents=True, exist_ok=True)

        bsz = images.shape[0]
        sample = min(4, bsz)
        denorm = bool(self.cfg.data.preprocess.get("normalize", True))
        for idx in range(sample):
            img_t = images[idx].detach().cpu()
            pred_hm_t = outputs[idx].detach().cpu()
            target_hm_t = targets[idx].detach().cpu()

            # Save prediction overlay
            save_overlay_from_tensor(
                img_t,
                pred_hm_t,
                pred_overlay_dir / f"sample_{idx:02d}.png",
                denormalize=denorm,
            )

            # Save target overlay
            save_overlay_from_tensor(
                img_t,
                target_hm_t,
                target_overlay_dir / f"sample_{idx:02d}.png",
                denormalize=denorm,
            )

    # -------------------------- Epoch hooks（解凍で再探索） --------------------------
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
                self.print(
                    f"[Backbone] Frozen (epoch {self.current_epoch} < {self._freeze_epochs})"
                )
        else:
            # Unfreeze once warmup period is over
            if self._backbone_frozen:
                self._set_backbone_requires_grad(True)
                self._backbone_frozen = False
                self.print(f"[Backbone] Unfrozen at epoch {self.current_epoch}")
                # NEW: 解凍タイミングで"安全な"マイクロバッチサイズを再探索させる
                if self._auto_mb:
                    self._runtime_mb = None  # いったん忘れてBから再挑戦

    def on_train_epoch_end(self) -> None:  # type: ignore[override]
        # scheduler を手動で進める（手動最適化のため）
        scheds = self.lr_schedulers()
        if scheds is None:
            return
        if isinstance(scheds, (list, tuple)):
            for s in scheds:
                s.step()
        else:
            scheds.step()

    def _set_backbone_requires_grad(self, flag: bool) -> None:
        try:
            for p in self.model.backbone.parameters():
                p.requires_grad = flag
        except Exception:
            pass
