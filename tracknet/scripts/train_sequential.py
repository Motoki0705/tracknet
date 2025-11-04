"""Sequential training script for TrackNet models.

Trains multiple models in sequence: vit -> convnext_fpn -> convnext_hrnet.
Each model uses the same training configuration but can be customized
via command-line overrides.

Usage examples:
    uv run python -m tracknet.scripts.train_sequential --dry-run
    uv run python -m tracknet.scripts.train_sequential --epochs 10 --batch-size 8
    uv run python -m tracknet.scripts.train_sequential --models vit_heatmap convnext_fpn_heatmap
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sequential training of TrackNet models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default sequence with dry run
  uv run python -m tracknet.scripts.train_sequential --dry-run
  
  # Custom epochs and batch size
  uv run python -m tracknet.scripts.train_sequential --epochs 20 --batch-size 16
  
  # Custom model sequence
  uv run python -m tracknet.scripts.train_sequential --models vit_heatmap convnext_fpn_heatmap
  
  # With training overrides
  uv run python -m tracknet.scripts.train_sequential training.optimizer.lr=1e-4
        """,
    )

    # Basic options
    parser.add_argument(
        "--data", default="tracknet", help="Data configuration name (default: tracknet)"
    )
    parser.add_argument(
        "--training",
        default="default",
        help="Training configuration name (default: default)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["vit_heatmap", "convnext_fpn_heatmap", "convnext_hrnet_heatmap"],
        help="Models to train in sequence (default: vit_heatmap convnext_fpn_heatmap convnext_hrnet_heatmap)",
    )

    # Training overrides
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides training.epochs)",
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size (overrides training.batch_size)"
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue training next model even if current model fails",
    )
    parser.add_argument(
        "--sleep-between",
        type=int,
        default=0,
        help="Seconds to sleep between model training (default: 0)",
    )

    # Output options
    parser.add_argument(
        "--output-dir", help="Output directory (overrides training.output_dir)"
    )

    # Parse known args to allow training overrides
    args, training_overrides = parser.parse_known_args(argv)
    args.training_overrides = training_overrides

    return args


def build_train_command(args: argparse.Namespace, model: str) -> list[str]:
    """Build the training command for a specific model."""
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "tracknet.scripts.train",
        "--data",
        args.data,
        "--model",
        model,
        "--training",
        args.training,
    ]

    # Add training overrides
    if args.epochs:
        cmd.append(f"training.epochs={args.epochs}")
    if args.batch_size:
        cmd.append(f"training.batch_size={args.batch_size}")
    if args.output_dir:
        cmd.append(f"training.output_dir={args.output_dir}")

    # Add any additional training overrides
    cmd.extend(args.training_overrides)

    # Add dry run flag
    if args.dry_run:
        cmd.append("--dry-run")

    return cmd


def run_command(cmd: list[str], dry_run: bool = False) -> int:
    """Execute a command and return the exit code."""
    if dry_run:
        print(f"Would execute: {' '.join(cmd)}")
        return 0

    print(f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError as e:
        print(f"Command not found: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    print("=== TrackNet Sequential Training ===")
    print(f"Models: {' -> '.join(args.models)}")
    print(f"Data config: {args.data}")
    print(f"Training config: {args.training}")

    if args.epochs:
        print(f"Epochs: {args.epochs}")
    if args.batch_size:
        print(f"Batch size: {args.batch_size}")
    if args.output_dir:
        print(f"Output dir: {args.output_dir}")
    if args.training_overrides:
        print(f"Training overrides: {' '.join(args.training_overrides)}")

    print(f"Continue on error: {args.continue_on_error}")
    print(f"Sleep between models: {args.sleep_between}s")
    print()

    # Track results
    results = {}
    total_start_time = time.time()

    for i, model in enumerate(args.models):
        print(f"=== Training Model {i+1}/{len(args.models)}: {model} ===")

        start_time = time.time()
        cmd = build_train_command(args, model)
        exit_code = run_command(cmd, args.dry_run)
        end_time = time.time()

        duration = end_time - start_time
        results[model] = {"exit_code": exit_code, "duration": duration}

        if args.dry_run:
            print(f"Would complete in {duration:.2f}s")
        else:
            if exit_code == 0:
                print(f"✓ Completed successfully in {duration:.2f}s")
            else:
                print(f"✗ Failed with exit code {exit_code} after {duration:.2f}s")

        # Sleep between models if not the last one
        if i < len(args.models) - 1 and args.sleep_between > 0 and not args.dry_run:
            print(f"Sleeping {args.sleep_between}s before next model...")
            time.sleep(args.sleep_between)

        print()

    # Summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print("=== Training Summary ===")
    successful = sum(1 for r in results.values() if r["exit_code"] == 0)
    total = len(results)

    for model, result in results.items():
        status = "✓" if result["exit_code"] == 0 else "✗"
        print(
            f"{status} {model}: {result['duration']:.2f}s (exit: {result['exit_code']})"
        )

    print(f"\nOverall: {successful}/{total} models successful")
    print(f"Total time: {total_duration:.2f}s")

    # Return appropriate exit code
    if args.dry_run:
        return 0
    elif args.continue_on_error:
        return 0 if successful > 0 else 1
    else:
        return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())
