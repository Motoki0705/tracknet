"""Common utilities and data structures for the annotation pipeline.

This module centralises constants (court keypoints, skeleton edges) and helper
functions used across converter, tracker, and UI tooling. Keeping all shared
logic in one place guarantees the schema stays consistent between producers
and validators.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

COURT_KEYPOINTS: list[dict[str, Any]] = [
    {"index": 0, "name": "far doubles corner left"},
    {"index": 1, "name": "far doubles corner right"},
    {"index": 2, "name": "near doubles corner left"},
    {"index": 3, "name": "near doubles corner right"},
    {"index": 4, "name": "far singles corner left"},
    {"index": 5, "name": "near singles corner left"},
    {"index": 6, "name": "far singles corner right"},
    {"index": 7, "name": "near singles corner right"},
    {"index": 8, "name": "far service-line endpoint left"},
    {"index": 9, "name": "far service-line endpoint right"},
    {"index": 10, "name": "near service-line endpoint left"},
    {"index": 11, "name": "near service-line endpoint right"},
    {"index": 12, "name": "far service T"},
    {"index": 13, "name": "near service T"},
    {"index": 14, "name": "net center"},
]


COURT_SKELETON_EDGES: list[list[int]] = [
    [1, 2],
    [3, 4],
    [1, 3],
    [2, 4],
    [5, 6],
    [7, 8],
    [9, 10],
    [11, 12],
    [13, 14],
]


def iso_timestamp() -> str:
    """Return a UTC ISO-8601 timestamp suitable for metadata.

    Returns:
        str: Timestamp string with timezone information.
    """

    return datetime.now(UTC).isoformat()


def ensure_directory(path: Path) -> None:
    """Ensure the parent directory of ``path`` exists.

    Args:
        path: Target file path whose parent directories should be created.
    """

    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any | None = None) -> Any:
    """Read JSON from ``path`` if it exists.

    Args:
        path: File to read.
        default: Value returned when the file does not exist.

    Returns:
        Any: Parsed JSON content or ``default`` if missing.
    """

    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, data: Any) -> None:
    """Write ``data`` to ``path`` atomically.

    Args:
        path: Destination file path.
        data: Serializable payload to write.
    """

    ensure_directory(path)
    with NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=str(path.parent)
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def merge_nested_dict(
    destination: MutableMapping[str, Any],
    source: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Merge ``source`` into ``destination`` recursively.

    Args:
        destination: Dictionary mutated in-place.
        source: Dictionary providing overrides or additional keys.

    Returns:
        MutableMapping[str, Any]: The mutated ``destination`` mapping.
    """

    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(
            destination.get(key), MutableMapping
        ):
            merge_nested_dict(destination[key], value)  # type: ignore[index]
        else:
            destination[key] = value
    return destination


def iter_games_clips(base_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield ``(game_id, game_path)`` pairs sorted by numeric order.

    Args:
        base_dir: Root directory that potentially contains game folders.

    Yields:
        tuple[str, Path]: Game identifier and its path.
    """

    if not base_dir.exists():
        return []

    def extract_game_number(game_name: str) -> int:
        """Extract numeric part from game name (e.g., 'game1' -> 1)."""
        import re

        match = re.search(r"game(\d+)", game_name.lower())
        return int(match.group(1)) if match else 0

    game_dirs = [
        (child.name, child)
        for child in base_dir.iterdir()
        if child.is_dir() and child.name.lower().startswith("game")
    ]

    # Sort by numeric game number
    game_dirs.sort(key=lambda x: extract_game_number(x[0]))

    yield from game_dirs


def iter_clip_dirs(game_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield ``(clip_id, clip_path)`` pairs under ``game_dir`` sorted by numeric order.

    Args:
        game_dir: Path to a specific game directory.

    Yields:
        tuple[str, Path]: Clip identifier and directory path.
    """

    def extract_clip_number(clip_name: str) -> int:
        """Extract numeric part from clip name (e.g., 'Clip1' -> 1, 'clip10' -> 10)."""
        import re

        match = re.search(r"clip(\d+)", clip_name.lower())
        return int(match.group(1)) if match else 0

    clip_dirs = [(child.name, child) for child in game_dir.iterdir() if child.is_dir()]

    # Sort by numeric clip number
    clip_dirs.sort(key=lambda x: extract_clip_number(x[0]))

    yield from clip_dirs
