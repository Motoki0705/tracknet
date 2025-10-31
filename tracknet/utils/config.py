"""Configuration utilities for TrackNet.

This module provides helpers to build a unified OmegaConf configuration by
loading YAML files from `configs/data/`, `configs/model/`, and
`configs/training/`, merging them, and applying CLI overrides. It also
initializes runtime aspects such as random seeds and output directories.

Example:
    >>> from tracknet.utils.config import build_cfg
    >>> cfg = build_cfg(data_name="tracknet", model_name="vit_heatmap", training_name="default")
    >>> print(cfg.data.root)

Notes:
    - Uses OmegaConf for flexible hierarchical configuration.
    - Creates a `runtime` section to hold non-static values such as seed,
      timestamp, and resolved output directories.
    - Directory creation can be skipped via `dry_run=True`.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from omegaconf import DictConfig, OmegaConf


# ------------------------------
# Internal utilities
# ------------------------------

# Simple cache for YAML configs to avoid repeated file I/O
_yaml_cache: dict[tuple[str, str], DictConfig] = {}

def _project_root() -> Path:
    """Return the project root directory path.

    This resolves from the current file position: `tracknet/utils/config.py`
    -> `tracknet/` -> project root.

    Returns:
        Path: Absolute path to the repository root.
    """

    return Path(__file__).resolve().parents[2]


def _configs_dir() -> Path:
    """Return the absolute path to the `configs/` directory.

    Returns:
        Path: `.../<repo-root>/configs`
    """

    return _project_root() / "configs"

def _load_yaml(category: str, name: str) -> DictConfig:
    """Load a YAML config from `configs/<category>/<name>.yaml`.

    Args:
        category: One of ``data``, ``model``, or ``training``.
        name: The base filename without extension.

    Returns:
        DictConfig: The loaded OmegaConf node.

    Raises:
        FileNotFoundError: If the expected YAML file does not exist.
    """

    cache_key = (category, name)
    if cache_key in _yaml_cache:
        return _yaml_cache[cache_key]
    
    path = _configs_dir() / category / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    
    config = OmegaConf.load(path)
    _yaml_cache[cache_key] = config
    return config


def _seed_all(seed: int) -> None:
    """Seed PRNGs for reproducibility.

    Args:
        seed: The seed value to apply.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Delay torch import until needed
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - depends on environment
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not available, skip torch seeding


@dataclass
class BuildConfigArgs:
    """Arguments used to construct a configuration.

    Attributes:
        data_name: Name of the data YAML in `configs/data/`.
        model_name: Name of the model YAML in `configs/model/`.
        training_name: Name of the training YAML in `configs/training/`.
        overrides: Optional dotlist-style overrides (e.g.,
            ["training.optimizer.lr=1e-4"]).
        seed: Optional seed override. If ``None``, uses config/default.
        output_dir: Optional output directory root override.
        dry_run: If True, do not create directories while building cfg.
    """

    data_name: str = "tracknet"
    model_name: str = "convnext_fpn_heatmap"
    training_name: str = "default"
    overrides: Optional[Iterable[str]] = None
    seed: Optional[int] = None
    output_dir: Optional[str] = None
    dry_run: bool = False


def build_cfg(
    data_name: str = "tracknet",
    model_name: str = "convnext_fpn_heatmap",
    training_name: str = "default",
    overrides: Optional[Iterable[str]] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
) -> DictConfig:
    """Build a unified configuration using OmegaConf.

    Args:
        data_name: Base name of the data YAML in ``configs/data``.
        model_name: Base name of the model YAML in ``configs/model``.
        training_name: Base name of the training YAML in ``configs/training``.
        overrides: Dotlist overrides like ``["training.optimizer.lr=1e-4"]``.
        seed: Optional RNG seed override.
        output_dir: Optional output directory root override.
        dry_run: If True, skip directory creation.

    Returns:
        DictConfig: A merged configuration with sections ``data``, ``model``,
            ``training``, and ``runtime``.
    """

    data_cfg = _load_yaml("data", data_name)
    model_cfg = _load_yaml("model", model_name)
    training_cfg = _load_yaml("training", training_name)

    cfg = OmegaConf.create({
        "data": data_cfg,
        "model": model_cfg,
        "training": training_cfg,
    })

    # Apply dotlist overrides if provided.
    if overrides:
        dot = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, dot)

    # Resolve seed preference order: explicit arg -> training.seed -> default
    resolved_seed = (
        int(seed)
        if seed is not None
        else int(cfg.training.get("seed", 42))
    )

    # Resolve output directories. Use training.{output_dir,log_dir,ckpt_dir} if present.
    root = _project_root()
    out_root = (
        Path(output_dir)
        if output_dir is not None
        else Path(cfg.training.get("output_dir", root / "outputs"))
    )
    log_dir = Path(cfg.training.get("log_dir", out_root / "logs"))
    ckpt_dir = Path(cfg.training.get("ckpt_dir", out_root / "checkpoints"))

    # Prepare runtime block with timestamped run_id.
    run_ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"run-{run_ts}-s{resolved_seed}"
    runtime = {
        "seed": resolved_seed,
        "project_root": str(root),
        "output_root": str(out_root),
        "log_dir": str(log_dir),
        "ckpt_dir": str(ckpt_dir),
        "timestamp": run_ts,
        "run_id": run_id,
    }

    cfg = OmegaConf.merge(cfg, OmegaConf.create({"runtime": runtime}))

    # Seed RNGs
    _seed_all(resolved_seed)

    # Ensure directories exist unless dry-run.
    if not dry_run:
        for p in (out_root, log_dir, ckpt_dir):
            Path(p).mkdir(parents=True, exist_ok=True)

    return cfg


def add_config_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach common configuration CLI arguments to a parser.

    Args:
        parser: An ``argparse.ArgumentParser`` to extend.

    Returns:
        argparse.ArgumentParser: The same parser for chaining.
    """

    parser.add_argument("--data", dest="data_name", default="tracknet", type=str,
                        help="Data config name in configs/data (without .yaml)")
    parser.add_argument("--model", dest="model_name", default="convnext_fpn_heatmap", type=str,
                        help="Model config name in configs/model (without .yaml)")
    parser.add_argument("--training", dest="training_name", default="default", type=str,
                        help="Training config name in configs/training (without .yaml)")
    parser.add_argument("--seed", dest="seed", default=None, type=int,
                        help="Seed override (if omitted, uses config/default)")
    parser.add_argument("--output-dir", dest="output_dir", default=None, type=str,
                        help="Output root directory override")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Build cfg and print without side effects")
    return parser


