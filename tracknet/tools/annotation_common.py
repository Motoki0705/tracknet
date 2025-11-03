"""Common utilities and data structures for the annotation pipeline.

This module centralises constants (court keypoints, skeleton edges) and helper
functions used across converter, tracker, and UI tooling. Keeping all shared
logic in one place guarantees the schema stays consistent between producers
and validators.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

import json
from datetime import datetime, timezone

COURT_KEYPOINTS: List[Dict[str, Any]] = [
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


COURT_SKELETON_EDGES: List[List[int]] = [
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

    return datetime.now(timezone.utc).isoformat()


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
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
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
        if isinstance(value, Mapping) and isinstance(destination.get(key), MutableMapping):
            merge_nested_dict(destination[key], value)  # type: ignore[index]
        else:
            destination[key] = value
    return destination


def iter_games_clips(base_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield ``(game_id, game_path)`` pairs sorted by name.

    Args:
        base_dir: Root directory that potentially contains game folders.

    Yields:
        tuple[str, Path]: Game identifier and its path.
    """

    if not base_dir.exists():
        return []
    for child in sorted(base_dir.iterdir()):
        if child.is_dir() and child.name.lower().startswith("game"):
            yield child.name, child


def iter_clip_dirs(game_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield ``(clip_id, clip_path)`` pairs under ``game_dir``.

    Args:
        game_dir: Path to a specific game directory.

    Yields:
        tuple[str, Path]: Clip identifier and directory path.
    """

    for child in sorted(game_dir.iterdir()):
        if child.is_dir():
            yield child.name, child
