"""Merge ball, player, and court annotations into a unified JSON payload."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from tracknet.tools.annotation_common import (
    COURT_KEYPOINTS,
    COURT_SKELETON_EDGES,
    ensure_directory,
    iso_timestamp,
    iter_clip_dirs,
    iter_games_clips,
    merge_nested_dict,
    read_json,
    write_json_atomic,
)

SCHEMA_VERSION = "1.0.0"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

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
        help="Root directory containing per-game clip folders.",
    )
    parser.add_argument(
        "--person-tracks",
        type=Path,
        default=Path("outputs/tracking/person_tracks.json"),
        help="Path to person tracking results JSON.",
    )
    parser.add_argument(
        "--player-assignments",
        type=Path,
        default=Path("outputs/tracking/player_assignments.json"),
        help="Path to user verified player assignments JSON.",
    )
    parser.add_argument(
        "--court-annotations",
        type=Path,
        default=Path("outputs/court_annotations/games.json"),
        help="Path to court annotation JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tracknet/annotations.json"),
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--games",
        nargs="*",
        help="Subset of game identifiers to process (default: all).",
    )
    parser.add_argument(
        "--clips",
        nargs="*",
        help="Subset of clip identifiers to process (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print merged payload without writing to disk.",
    )
    return parser.parse_args(argv)


def safe_int(value: Any, default: int | None = None) -> int | None:
    """Cast ``value`` to ``int`` with graceful fallback.

    Args:
        value: Value to convert.
        default: Value returned when conversion fails.

    Returns:
        Optional[int]: Converted integer or ``default``.
    """

    try:
        if value in (None, "", "null", "None"):
            return default
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Cast ``value`` to ``float`` with graceful fallback.

    Args:
        value: Value to convert.
        default: Value returned when conversion fails.

    Returns:
        Optional[float]: Converted float or ``default``.
    """

    try:
        if value in (None, "", "null", "None"):
            return default
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_ball_csv(csv_path: Path) -> Dict[str, List[Any]]:
    """Parse ball CSV/Label file into column-aligned arrays.

    Args:
        csv_path: Path to `Label.csv`.

    Returns:
        Dict[str, List[Any]]: Column-aligned ball annotation arrays.
    """

    frames: List[int] = []
    xs: List[float | None] = []
    ys: List[float | None] = []
    visibilities: List[int] = []
    statuses: List[int | None] = []

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame = (
                safe_int(row.get("frame"))
                or safe_int(row.get("frame_index"))
                or safe_int(Path(row.get("file name", "0")).stem, default=0)
            )
            visibility = safe_int(row.get("visibility"), default=0) or 0
            status = safe_int(row.get("status"))
            x = safe_float(row.get("x")) or safe_float(row.get("x-coordinate"))
            y = safe_float(row.get("y")) or safe_float(row.get("y-coordinate"))
            frames.append(frame or 0)
            xs.append(x)
            ys.append(y)
            visibilities.append(max(0, min(3, visibility)))
            statuses.append(status if status in (0, 1, 2) else None)

    return {
        "frames": frames,
        "x": xs,
        "y": ys,
        "visibility": visibilities,
        "status": statuses,
    }


def load_person_tracks(path: Path) -> Dict[str, Any]:
    """Load person tracking data keyed by game, clip, and track.

    Args:
        path: JSON file produced by ``player_tracker --mode detect``.

    Returns:
        Dict[str, Any]: Dictionary containing ``meta`` and ``index`` maps.
    """

    payload = read_json(path, default={})
    games = payload.get("games", {}) if isinstance(payload, Mapping) else {}

    index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for game_id, game_data in games.items():
        clips = game_data.get("clips", {}) if isinstance(game_data, Mapping) else {}
        clip_index: Dict[str, Dict[str, Any]] = {}
        for clip_id, clip_data in clips.items():
            tracks = clip_data.get("tracks", {}) if isinstance(clip_data, Mapping) else {}
            clip_index[clip_id] = {str(tid): track for tid, track in tracks.items()}
        index[game_id] = clip_index
    return {
        "meta": payload,
        "index": index,
    }


def load_player_assignments(path: Path) -> Dict[str, Any]:
    """Load user player assignments keyed by game and clip.

    Args:
        path: JSON file created by the player assignment UI.

    Returns:
        Dict[str, Any]: Dictionary containing ``meta`` and ``index`` maps.
    """

    payload = read_json(path, default={})
    games = payload.get("games", {}) if isinstance(payload, Mapping) else {}
    index: Dict[str, Dict[str, Any]] = {}
    for game_id, game_data in games.items():
        clips = game_data.get("clips", {}) if isinstance(game_data, Mapping) else {}
        index[game_id] = {clip_id: clip_data for clip_id, clip_data in clips.items()}
    return {"meta": payload, "index": index}


def load_court_annotations(path: Path) -> Dict[str, Any]:
    """Load court annotations keyed by game.

    Args:
        path: JSON file generated by the court annotation manager.

    Returns:
        Dict[str, Any]: Dictionary containing ``meta`` and ``index`` maps.
    """

    payload = read_json(path, default={})
    games = payload.get("games", {}) if isinstance(payload, Mapping) else {}
    index = {game_id: game_data for game_id, game_data in games.items()}
    return {"meta": payload, "index": index}


def build_player_payload(
    assignments: Mapping[str, Any],
    track_lookup: Dict[str, Dict[str, Dict[str, Any]]],
    game_id: str,
    clip_id: str,
) -> Dict[str, Any]:
    """Build per-player payload for the unified annotation.

    Args:
        assignments: Mapping of user assignments keyed by game/clip.
        track_lookup: Tracking results keyed by game/clip/track id.
        game_id: Game identifier.
        clip_id: Clip identifier.

    Returns:
        Dict[str, Any]: Player block ready for inclusion in the unified JSON.
    """

    player_block: Dict[str, Any] = {}
    clip_assignment = assignments.get(game_id, {}).get(clip_id)
    if not clip_assignment:
        return player_block
    players = clip_assignment.get("players", {})
    for role, data in players.items():
        track_id = str(data.get("track_id"))
        tracks_for_clip = track_lookup.get(game_id, {}).get(clip_id, {})
        track = tracks_for_clip.get(track_id)
        if not track:
            continue
        detections = track.get("detections", [])
        frames: List[int] = []
        bboxes: List[List[float]] = []
        visibilities: List[float] = []
        for detection in detections:
            frame_id = safe_int(detection.get("frame"))
            bbox = detection.get("bbox")
            if frame_id is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            frames.append(frame_id)
            bboxes.append([float(component) for component in bbox])
            visibilities.append(float(detection.get("visibility", 1.0)))
        player_block[role] = {
            "track_id": track_id,
            "frames": frames,
            "bbox": bboxes,
            "visibility": visibilities,
            "state": track.get("state_sequence"),
            "metadata": {
                "summary": track.get("summary"),
                "assigned_by": data.get("assigned_by"),
            },
        }
    return player_block


def build_court_payload(court_lookup: Mapping[str, Any], game_id: str) -> Dict[str, Any]:
    """Return court payload for a game, filling defaults if missing.

    Args:
        court_lookup: Mapping of court annotations by game.
        game_id: Game identifier being processed.

    Returns:
        Dict[str, Any]: Court annotation block.
    """

    court = court_lookup.get(game_id)
    if not court:
        keypoints = [
            {
                "index": kp["index"],
                "name": kp["name"],
                "x": None,
                "y": None,
            }
            for kp in COURT_KEYPOINTS
        ]
        return {
            "keypoints": keypoints,
            "skeleton": COURT_SKELETON_EDGES,
            "metadata": {"source": "auto-generated"},
        }
    keypoints = court.get("keypoints", [])
    if len(keypoints) != len(COURT_KEYPOINTS):
        # Rebuild canonical ordering.
        kp_map = {kp["index"]: kp for kp in keypoints if isinstance(kp, Mapping)}
        keypoints = [
            {
                "index": entry["index"],
                "name": entry["name"],
                "x": kp_map.get(entry["index"], {}).get("x"),
                "y": kp_map.get(entry["index"], {}).get("y"),
                "confidence": kp_map.get(entry["index"], {}).get("confidence"),
                "visible": kp_map.get(entry["index"], {}).get("visible"),
            }
            for entry in COURT_KEYPOINTS
        ]
    return {
        "keypoints": keypoints,
        "skeleton": court.get("skeleton", COURT_SKELETON_EDGES),
        "annotator": court.get("annotator"),
        "reference_frame": court.get("reference_frame"),
        "metadata": court.get("metadata"),
    }


def convert_annotations(config: argparse.Namespace) -> Dict[str, Any]:
    """Perform the conversion and return the merged payload.

    Args:
        config: Parsed CLI arguments controlling input/output paths.

    Returns:
        Dict[str, Any]: Unified annotation payload.
    """

    person_tracks = load_person_tracks(config.person_tracks)
    player_assignments = load_player_assignments(config.player_assignments)
    court_annotations = load_court_annotations(config.court_annotations)

    games_payload: Dict[str, Any] = {}

    for game_id, game_dir in iter_games_clips(config.data_root):
        if config.games and game_id not in config.games:
            continue

        clips_payload: Dict[str, Any] = {}
        for clip_id, clip_dir in iter_clip_dirs(game_dir):
            if config.clips and clip_id not in config.clips:
                continue

            csv_candidates = [
                clip_dir / "Label.csv",
            ]
            csv_path = next((path for path in csv_candidates if path.exists()), None)
            if not csv_path:
                continue

            ball_payload = parse_ball_csv(csv_path)
            player_payload = build_player_payload(
                player_assignments["index"],
                person_tracks["index"],
                game_id,
                clip_id,
            )
            clip_payload: Dict[str, Any] = {
                "source": str(clip_dir.relative_to(config.data_root)),
                "ball": ball_payload,
            }
            if player_payload:
                clip_payload["players"] = player_payload
            clips_payload[clip_id] = clip_payload

        if not clips_payload:
            continue
        court_payload = build_court_payload(court_annotations["index"], game_id)
        games_payload[game_id] = {
            "court": court_payload,
            "clips": clips_payload,
        }

    merged_payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": iso_timestamp(),
        "games": games_payload,
    }

    existing_output = read_json(config.output, default=None)
    if isinstance(existing_output, Mapping) and not config.dry_run:
        merged_payload = merge_nested_dict(existing_output, merged_payload)  # type: ignore[arg-type]

    if config.dry_run:
        return merged_payload

    ensure_directory(config.output)
    write_json_atomic(config.output, merged_payload)
    return merged_payload


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional raw argument list.
    """

    config = parse_args(argv)
    payload = convert_annotations(config)
    if config.dry_run:
        import json

        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
