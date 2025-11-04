"""Manage per-game court annotations via the interactive form."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from tracknet.tools.annotation_common import (
    iso_timestamp,
    iter_games_clips,
    read_json,
    write_json_atomic,
)
from tracknet.tools.utils.ui.court_annotation_form import CourtAnnotationForm


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the court annotation manager.

    Args:
        argv: Optional raw argument list. ``None`` defaults to ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/tracknet"),
        help="Dataset root path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/court_annotations/games.json"),
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--games",
        nargs="*",
        help="Specific game identifiers to annotate (default: all available).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing entries without prompting.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable Matplotlib previews (useful in headless environments).",
    )
    return parser.parse_args(argv)


def discover_games(data_root: Path) -> list[str]:
    """Discover game identifiers from the dataset.

    Args:
        data_root: Directory containing game subfolders.

    Returns:
        List[str]: Sorted list of game identifiers.
    """

    games = [game_id for game_id, _ in iter_games_clips(data_root)]
    if games:
        return games
    # Fallback to canonical numbering when dataset is unavailable.
    return [f"game{idx}" for idx in range(1, 11)]


def manage_court_annotations(args: argparse.Namespace) -> dict[str, Any]:
    """Run the annotation workflow.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Dict[str, Any]: Updated court annotation payload.
    """

    existing = read_json(args.output, default={})
    payload: MutableMapping[str, Any] = (
        existing
        if isinstance(existing, MutableMapping)
        else {"version": "1.0.0", "games": {}}
    )
    games_block: MutableMapping[str, Any] = payload.setdefault("games", {})

    target_games: Iterable[str]
    target_games = args.games or discover_games(args.data_root)

    form = CourtAnnotationForm(enable_plot=not args.no_plot)
    for game_id in target_games:
        current = games_block.get(game_id)
        if current and not args.force:
            print(f"Skipping {game_id} (already annotated). Use --force to overwrite.")
            continue
        games_block[game_id] = form.collect(game_id, current)

    payload["updated_at"] = iso_timestamp()
    payload.setdefault("version", "1.0.0")
    if not args.no_plot:
        print("Court annotation complete. Close the Matplotlib window to finish.")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional raw argument list.
    """

    args = parse_args(argv)
    payload = manage_court_annotations(args)
    write_json_atomic(args.output, payload)


if __name__ == "__main__":
    main()
