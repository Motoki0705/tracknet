"""Training entrypoint for TrackNet.

Builds a unified configuration (OmegaConf) from YAMLs and optional CLI
overrides, then launches training. For Section 1 implementation, this script
supports a dry-run mode that constructs and prints the configuration without
performing any training side effects.

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

from tracknet.utils.config import add_config_cli_arguments, build_cfg


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
    add_config_cli_arguments(parser)
    known, unknown = parser.parse_known_args(argv)
    return known, unknown


def main(argv: List[str] | None = None) -> int:
    """Main entrypoint.

    Args:
        argv: Optional argument vector. If ``None``, uses ``sys.argv[1:]``.

    Returns:
        Process exit code. ``0`` on success.
    """

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

    # Always print the constructed configuration for visibility.
    print("==== TrackNet Config (merged) ====")
    print(OmegaConf.to_yaml(cfg))

    if args.dry_run:
        print("[dry-run] Exiting before training loop.")
        return 0

    # Placeholder for future training logic (Section 5).
    # from tracknet.training.trainer import Trainer
    # trainer = Trainer(cfg)
    # trainer.train()
    print("Training stub not yet implemented. Coming in later sections.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    raise SystemExit(main())

