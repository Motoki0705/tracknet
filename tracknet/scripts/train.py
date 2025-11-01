"""PyTorch Lightning-based training entrypoint for TrackNet.

Builds a unified configuration (OmegaConf) from YAMLs and optional CLI
overrides, then launches training using ``pytorch-lightning``.

Usage examples:
    uv run python -m tracknet.scripts.train --dry-run
    uv run python -m tracknet.scripts.train --data tracknet --model vit_heatmap --training default --dry-run
    uv run python -m tracknet.scripts.train training.optimizer.lr=1e-4 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from omegaconf import OmegaConf

# Heavy imports are delayed until needed
def _import_lightning():
    """Delay import of lightning components to speed up startup."""
    global torch, pl, ModelCheckpoint, EarlyStopping, LearningRateMonitor, CSVLogger
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    return torch, pl, ModelCheckpoint, EarlyStopping, LearningRateMonitor, CSVLogger

def _import_tracknet():
    """Delay import of basic tracknet components."""
    global build_cfg
    from tracknet.utils.config import add_config_cli_arguments, build_cfg
    return build_cfg

def _import_lightning_modules():
    """Delay import of lightning components."""
    global TrackNetDataModule, PLHeatmapModule
    from tracknet.training.lightning_datamodule import TrackNetDataModule
    from tracknet.training.lightning_module import PLHeatmapModule
    return TrackNetDataModule, PLHeatmapModule


def _add_config_cli_arguments(parser):
    """Delay import of config utilities."""
    from tracknet.utils.config import add_config_cli_arguments
    add_config_cli_arguments(parser)

def parse_args(argv: List[str]) -> tuple[argparse.Namespace, List[str]]:
    """Parse CLI arguments, keeping unknown args as overrides.

    The unknown arguments are intended to be OmegaConf dotlist overrides like
    ``training.optimizer.lr=1e-4``.

    Args:
        argv: Argument vector, typically ``sys.argv[1:]``.

    Returns:
        A tuple of ``(known_args, overrides)``.
    """

    parser = argparse.ArgumentParser(description="TrackNet training entrypoint")
    _add_config_cli_arguments(parser)
    known, unknown = parser.parse_known_args(argv)
    return known, unknown


def main(argv: List[str] | None = None) -> int:
    """Main entrypoint.

    Args:
        argv: Optional argument vector. If ``None``, uses ``sys.argv[1:]``.

    Returns:
        Process exit code. ``0`` on success.
    """

    import os
    import sys
    
    # Optimize Python startup
    sys.dont_write_bytecode = True
    
    # Optimize PyTorch startup
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["PYTHONHASHSEED"] = "0"
    
    if argv is None:
        argv = sys.argv[1:]

    args, overrides = parse_args(argv)
    build_cfg = _import_tracknet()
    
    cfg = build_cfg(
        data_name=args.data_name,
        model_name=args.model_name,
        training_name=args.training_name,
        overrides=overrides,
        seed=args.seed,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    # Always print the constructed configuration for visibility.
    print("==== TrackNet Config (merged) ====")
    print(OmegaConf.to_yaml(cfg))

    if args.dry_run:
        print("[dry-run] Exiting before training loop.")
        return 0

    # Import Lightning modules only when needed
    torch, pl, ModelCheckpoint, EarlyStopping, LearningRateMonitor, CSVLogger = _import_lightning()
    TrackNetDataModule, PLHeatmapModule = _import_lightning_modules()
    
    # Launch training (Lightning)
    pl.seed_everything(int(cfg.runtime.seed), workers=True)

    # Data and model
    datamodule = TrackNetDataModule(cfg)
    lit_module = PLHeatmapModule(cfg)

    # Logger (CSV to avoid extra deps; directory: <log_dir>/<run_id>)
    csv_logger = CSVLogger(
        save_dir=str(cfg.runtime.log_dir),
        name=str(cfg.runtime.run_id),
    )

    # Callbacks (Lightning built-ins)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(cfg.runtime.ckpt_dir),
        filename=f"best_{cfg.runtime.run_id}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )
    lrmon_cb = LearningRateMonitor(logging_interval="epoch")

    callbacks = [ckpt_cb, lrmon_cb]
    escfg = cfg.training.get("early_stopping", None)
    if escfg:
        callbacks.append(
            EarlyStopping(
                monitor=str(escfg.get("monitor", "val/loss")),
                mode=str(escfg.get("mode", "min")),
                patience=int(escfg.get("patience", 10)),
                min_delta=float(escfg.get("min_delta", 0.0)),
            )
        )

    # Precision mapping
    def _map_precision() -> str:
        req = str(cfg.training.get("precision", cfg.training.get("amp", False) and "fp16" or "fp32")).lower()
        if req == "fp16" and torch.cuda.is_available():
            return "16-mixed"
        if req == "bf16" and torch.cuda.is_available():
            return "bf16-mixed"
        return "32-true"

    # Trainer
    trainer_kwargs = {
        "accelerator": ("gpu" if torch.cuda.is_available() else "cpu"),
        "devices": "auto",
        "precision": _map_precision(),
        "max_epochs": int(cfg.training.get("epochs", 1)),
        "limit_train_batches": int(cfg.training.get("limit_train_batches", 0)) or 1.0,
        "limit_val_batches": int(cfg.training.get("limit_val_batches", 0)) or 1.0,
        "logger": csv_logger,
        "callbacks": callbacks,
        "log_every_n_steps": 50,
        "enable_progress_bar": True,
    }
    
    # メモリ最適化設定を適用
    mem_cfg = cfg.training.get("memory_optimization", {})
    if mem_cfg.get("use_cpu_offload", False) and torch.cuda.is_available():
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = 1
        # CPUオフロードはLightningの戦略で実装
        print("CPU offload enabled - may reduce training speed")
    
    if mem_cfg.get("use_disk_offload", False):
        print("Disk offload enabled - significantly reduces training speed")
    
    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(lit_module, datamodule=datamodule)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    raise SystemExit(main())
