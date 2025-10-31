"""Training orchestration for TrackNet.

This module wires together datasets, model factory, losses, optimizer,
scheduler, and callbacks to run a minimal but complete training/validation
loop.

Key features:
- Supports frame and (basic) sequence modes based on cfg.data.sequence.enabled.
- Delegates model assembly to ``tracknet.models.build_model``.
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
from pprint import pformat
from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging
import math
import random
import time

import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig, OmegaConf

from tracknet.datasets import (
    PreprocessConfig,
    TrackNetFrameDataset,
    TrackNetFrameDatasetConfig,
    TrackNetSequenceDataset,
    TrackNetSequenceDatasetConfig,
    collate_frames,
    collate_sequences,
)
from tracknet.models import build_model
from tracknet.training import HeatmapLossConfig, build_heatmap_loss, heatmap_argmax_coords, visible_from_mask
from tracknet.utils.logging import Logger, LoggerConfig, save_image_from_tensor, save_overlay_from_tensor

# Optional progress bars via tqdm
try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm as _tqdm
except Exception:  # Fallback: identity iterator
    def _tqdm(it, total=None, desc=None):  # type: ignore
        return it


@dataclass
class Trainer:
    """Trainer orchestrates data, model, loss, and optimization."""

    cfg: DictConfig

    def __post_init__(self) -> None:
        """Emit a configuration summary as soon as the trainer is constructed."""

        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(level=logging.INFO)

        summary = self._configuration_snapshot()
        formatted = pformat(summary, indent=2)
        logging.getLogger(__name__).info("Trainer initialization summary:\n%s", formatted)
        print(f"[Trainer] Initialization summary:\n{formatted}")

    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_precision(self) -> Tuple[str, str, Optional[torch.dtype], bool]:
        """Resolve precision-related runtime settings.

        Returns:
            A tuple ``(mode, device_type, autocast_dtype, use_grad_scaler_fp16)``.
        """

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        mode = str(self.cfg.training.get("precision", "auto")).lower()
        if mode == "auto":
            mode = "fp16" if bool(self.cfg.training.get("amp", False)) else "fp32"

        use_fp16 = mode == "fp16" and device_type == "cuda"
        use_bf16 = mode == "bf16" and device_type in ("cuda", "cpu")

        if use_fp16:
            autocast_dtype = torch.float16
        elif use_bf16:
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = None

        return mode, device_type, autocast_dtype, use_fp16

    def _configuration_snapshot(self) -> Dict[str, Any]:
        """Build a structured summary of key configuration fields.

        Returns:
            A dictionary that captures runtime, data, training, and model highlights.
        """

        data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        model_cfg = OmegaConf.to_container(self.cfg.model, resolve=True)
        training_cfg = OmegaConf.to_container(self.cfg.training, resolve=True)
        runtime_cfg = OmegaConf.to_container(self.cfg.runtime, resolve=True)

        mode, device_type, autocast_dtype, use_grad_scaler = self._resolve_precision()
        data_loader_cfg = training_cfg.get("data_loader", {}) if isinstance(training_cfg, dict) else {}
        sequence_cfg = data_cfg.get("sequence", {}) if isinstance(data_cfg, dict) else {}
        split_cfg = data_cfg.get("split", {}) if isinstance(data_cfg, dict) else {}

        summary: Dict[str, Any] = {
            "runtime": {
                "run_id": runtime_cfg.get("run_id") if isinstance(runtime_cfg, dict) else None,
                "seed": runtime_cfg.get("seed") if isinstance(runtime_cfg, dict) else None,
                "log_dir": runtime_cfg.get("log_dir") if isinstance(runtime_cfg, dict) else None,
                "ckpt_dir": runtime_cfg.get("ckpt_dir") if isinstance(runtime_cfg, dict) else None,
                "device": str(self._device()),
                "device_type": device_type,
                "precision": mode,
                "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
                "grad_scaler_enabled": use_grad_scaler,
            },
            "data": {
                "root": data_cfg.get("root") if isinstance(data_cfg, dict) else None,
                "mode": "sequence" if bool(sequence_cfg.get("enabled", False)) else "frame",
                "train_games": len(split_cfg.get("train_games", [])) if isinstance(split_cfg, dict) else 0,
                "val_games": len(split_cfg.get("val_games", [])) if isinstance(split_cfg, dict) else 0,
                "preprocess": data_cfg.get("preprocess") if isinstance(data_cfg, dict) else None,
                "sequence": sequence_cfg,
            },
            "training": {
                "batch_size": training_cfg.get("batch_size") if isinstance(training_cfg, dict) else None,
                "epochs": training_cfg.get("epochs") if isinstance(training_cfg, dict) else None,
                "grad_clip": training_cfg.get("grad_clip", 0.0) if isinstance(training_cfg, dict) else 0.0,
                "limit_train_batches": training_cfg.get("limit_train_batches", 0) if isinstance(training_cfg, dict) else 0,
                "limit_val_batches": training_cfg.get("limit_val_batches", 0) if isinstance(training_cfg, dict) else 0,
                "data_loader": data_loader_cfg,
                "optimizer": training_cfg.get("optimizer") if isinstance(training_cfg, dict) else None,
                "scheduler": training_cfg.get("scheduler") if isinstance(training_cfg, dict) else None,
            },
            "model": {
                "pretrained_model_name": model_cfg.get("pretrained_model_name") if isinstance(model_cfg, dict) else None,
                "backbone": model_cfg.get("backbone") if isinstance(model_cfg, dict) else None,
                "decoder": model_cfg.get("decoder") if isinstance(model_cfg, dict) else None,
                "heatmap": model_cfg.get("heatmap") if isinstance(model_cfg, dict) else None,
            },
        }
        return summary

    # ---------------- Data -----------------
    def _build_dataloaders(self) -> tuple[DataLoader, Optional[DataLoader]]:
        dcfg = self.cfg.data
        mcfg = self.cfg.model

        pp = PreprocessConfig(
            resize=None
            if dcfg.preprocess.get("resize", None) in (None, "null")
            else tuple(dcfg.preprocess.resize),
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
            val_ds = (
                TrackNetSequenceDataset(
                    TrackNetSequenceDatasetConfig(
                        root=str(dcfg.root),
                        games=list(dcfg.split.val_games),
                        length=length,
                        stride=stride,
                        preprocess=pp,
                    )
                )
                if dcfg.split.get("val_games")
                else None
            )

            collate_fn = lambda b: collate_sequences(b, heatmap_size=heatmap_size, sigma=sigma)
        else:
            train_ds = TrackNetFrameDataset(
                TrackNetFrameDatasetConfig(
                    root=str(dcfg.root),
                    games=list(dcfg.split.train_games),
                    preprocess=pp,
                )
            )
            val_ds = (
                TrackNetFrameDataset(
                    TrackNetFrameDatasetConfig(
                        root=str(dcfg.root),
                        games=list(dcfg.split.val_games),
                        preprocess=pp,
                    )
                )
                if dcfg.split.get("val_games")
                else None
            )

            collate_fn = lambda b: collate_frames(b, heatmap_size=heatmap_size, sigma=sigma)

        bs = int(self.cfg.training.batch_size)
        dl_cfg = self.cfg.training.get("data_loader", {})
        num_workers = int(dl_cfg.get("num_workers", 0))
        pin_memory = bool(dl_cfg.get("pin_memory", False))
        persistent_workers = bool(dl_cfg.get("persistent_workers", False)) and num_workers > 0
        prefetch_factor = int(dl_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
        drop_last = bool(dl_cfg.get("drop_last", False))

        common_kwargs = dict(
            batch_size=bs,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        if prefetch_factor is not None:
            common_kwargs["prefetch_factor"] = prefetch_factor

        train_loader = DataLoader(train_ds, shuffle=True, **common_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs) if val_ds else None
        return train_loader, val_loader

    # ---------------- Model & Optim -----------------
    def _build_model(self) -> torch.nn.Module:
        model = build_model(self.cfg.model)
        return model.to(self._device())

    def _build_loss(self) -> torch.nn.Module:
        tcfg = self.cfg.training
        lname = str(tcfg.get("loss", {}).get("name", "mse")) if hasattr(tcfg, "loss") else "mse"
        alpha = float(tcfg.get("loss", {}).get("alpha", 2.0)) if hasattr(tcfg, "loss") else 2.0
        beta = float(tcfg.get("loss", {}).get("beta", 4.0)) if hasattr(tcfg, "loss") else 4.0
        return build_heatmap_loss(HeatmapLossConfig(name=lname, alpha=alpha, beta=beta))

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        ocfg = self.cfg.training.optimizer
        name = str(ocfg.get("name", "adamw")).lower()
        lr = float(ocfg.get("lr", 5e-4))
        wd = float(ocfg.get("weight_decay", 0.0))
        params = [p for p in model.parameters() if p.requires_grad]
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if name == "sgd":
            momentum = float(ocfg.get("momentum", 0.9))
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        raise ValueError(f"Unsupported optimizer: {name}")

    def _build_scheduler(
        self, optim: torch.optim.Optimizer
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:  # type: ignore[attr-defined]
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
        # Precision selection: prefer new `training.precision`, fallback to legacy `amp`
        prec, device_type, autocast_dtype, use_fp16 = self._resolve_precision()
        use_bf16 = autocast_dtype == torch.bfloat16

        # GradScaler only for fp16 on CUDA
        scaler = torch.amp.GradScaler(enabled=use_fp16)  # type: ignore[attr-defined]

        # Limits for quick smoke tests
        limit_train = int(self.cfg.training.get("limit_train_batches", 0))
        limit_val = int(self.cfg.training.get("limit_val_batches", 0))

        # Checkpoint paths
        ckpt_dir = Path(self.cfg.runtime.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_val = math.inf

        runtime_cfg = self.cfg.runtime
        run_id = runtime_cfg.get("run_id") if hasattr(runtime_cfg, "get") else getattr(runtime_cfg, "run_id", None)
        logger = Logger(
            LoggerConfig(
                log_dir=str(runtime_cfg.get("log_dir") if hasattr(runtime_cfg, "get") else self.cfg.runtime.log_dir),
                run_id=str(run_id) if run_id is not None else None,
                use_tensorboard=False,
            )
        )
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            train_loss = 0.0
            num_batches = 0
            # Wrap train loader with tqdm (if available)
            try:
                total_train = None
                if limit_train:
                    total_train = limit_train
                    try:
                        total_train = min(limit_train, len(train_loader))
                    except Exception:
                        pass
                iter_train = _tqdm(train_loader, total=total_train, desc=f"Train {epoch:03d}")
            except Exception:
                iter_train = train_loader

            for bi, batch in enumerate(iter_train, start=1):
                non_blocking = bool(self.cfg.training.get("data_loader", {}).get("pin_memory", False))
                images = batch["images"].to(device, dtype=torch.float32, non_blocking=non_blocking)
                targets = batch["heatmaps"].to(device, dtype=torch.float32, non_blocking=non_blocking)
                masks = batch["masks"].to(device, dtype=torch.float32, non_blocking=non_blocking)

                optimizer.zero_grad(set_to_none=True)
                if autocast_dtype is not None:
                    ctx = torch.autocast(device_type=device_type, dtype=autocast_dtype)  # type: ignore[arg-type]
                else:
                    from contextlib import nullcontext

                    ctx = nullcontext()
                with ctx:
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
                saved_preview = False
                with torch.inference_mode():
                    try:
                        total_val = None
                        if limit_val:
                            total_val = limit_val
                            try:
                                total_val = min(limit_val, len(val_loader))
                            except Exception:
                                pass
                        iter_val = _tqdm(val_loader, total=total_val, desc=f"Val   {epoch:03d}")
                    except Exception:
                        iter_val = val_loader

                    for vi, batch in enumerate(iter_val, start=1):
                        non_blocking = bool(self.cfg.training.get("data_loader", {}).get("pin_memory", False))
                        images = batch["images"].to(device, dtype=torch.float32, non_blocking=non_blocking)
                        targets = batch["heatmaps"].to(device, dtype=torch.float32, non_blocking=non_blocking)
                        masks = batch["masks"].to(device, dtype=torch.float32, non_blocking=non_blocking)
                        if autocast_dtype is not None:
                            ctx = torch.autocast(device_type=device_type, dtype=autocast_dtype)  # type: ignore[arg-type]
                        else:
                            from contextlib import nullcontext

                            ctx = nullcontext()
                        with ctx:
                            outputs = model(images)
                        loss = criterion(outputs, targets, masks)
                        vloss += float(loss.detach().cpu())
                        vnum += 1
                if not saved_preview:
                    base_dir = Path(logger.dir)
                    overlay_dir = base_dir / "overlays" / f"epoch{epoch:03d}"
                    image_dir = base_dir / "inputs" / f"epoch{epoch:03d}"
                    overlay_dir.mkdir(parents=True, exist_ok=True)
                    image_dir.mkdir(parents=True, exist_ok=True)
                    batch_size = images.shape[0]
                    sample_size = min(4, batch_size)
                    indices = random.sample(range(batch_size), sample_size)
                    denorm = bool(self.cfg.data.preprocess.get("normalize", True))
                    for idx in indices:
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
                    saved_preview = True
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
            logger.log_scalar("train/loss", float(train_loss), epoch)
            if val_loss is not None:
                logger.log_scalar("val/loss", float(val_loss), epoch)

            score = val_loss if val_loss is not None else train_loss
            if score < best_val:
                best_val = score
                path = ckpt_dir / f"best_{self.cfg.runtime.run_id}.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "cfg": self.cfg,
                    },
                    path,
                )
                print(f"Saved best checkpoint to {path}")
        logger.close()
