"""Utilities for running YOLOv8 tracking and assigning player identities."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

from tracknet.tools.annotation_common import (
    iso_timestamp,
    iter_clip_dirs,
    iter_games_clips,
    read_json,
    write_json_atomic,
)
from tracknet.tools.annotation_converter import (
    build_player_payload,
    load_person_tracks,
    load_player_assignments,
)
from tracknet.tools.utils.ui.player_identifier import PlayerIdentifierUI
from tracknet.tools.utils.video_generator import (
    generate_video_from_frames,
    validate_frame_sequence,
)

try:
    from tracknet.tools.utils.preprocess.tracker import Tracker as DefaultTracker
except Exception:  # pragma: no cover - optional dependency
    DefaultTracker = None  # type: ignore[assignment]

LOGGER = logging.getLogger("tracknet.player_tracker")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the player tracker.

    Args:
        argv: Optional raw argument list. ``None`` defaults to ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["detect", "assign"], required=True, help="Operating mode.")
    parser.add_argument("--data-root", type=Path, default=Path("data/tracknet"), help="Dataset root directory.")
    parser.add_argument(
        "--person-tracks",
        type=Path,
        default=Path("outputs/tracking/person_tracks.json"),
        help="Output JSON for YOLO tracking results.",
    )
    parser.add_argument(
        "--player-assignments",
        type=Path,
        default=Path("outputs/tracking/player_assignments.json"),
        help="Output JSON for manual player assignments.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/tracknet/annotations.json"),
        help="Unified annotation file updated incrementally after assignments.",
    )
    parser.add_argument("--games", nargs="*", help="Filter to specific games (default: all).")
    parser.add_argument("--clips", nargs="*", help="Filter to specific clips (default: all).")
    parser.add_argument("--force", action="store_true", help="Rerun detection even when cached results exist.")
    parser.add_argument("--dry-run", action="store_true", help="Skip writing files; useful for validation.")
    parser.add_argument(
        "--video-ext",
        nargs="*",
        default=[".mp4", ".mov", ".avi", ".mkv"],
        help="Candidate video extensions searched inside each clip directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity level for logging.",
    )
    return parser.parse_args(argv)


def default_tracker_factory() -> Any:
    """Instantiate the default YOLOv8 tracker.

    Returns:
        Any: Tracker instance implementing ``track(video_path)``.

    Raises:
        RuntimeError: If the optional tracker dependency is unavailable.
    """

    if DefaultTracker is None:
        raise RuntimeError(
            "Tracker dependency missing. Install requirements for tracknet.tools.utils.preprocess.tracker."
        )
    return DefaultTracker()


def convert_bbox_xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    """Convert ``[x1, y1, x2, y2]`` to ``[x, y, w, h]``.

    Args:
        box: Bounding box in absolute corner coordinates.

    Returns:
        List[float]: Bounding box expressed as origin plus width/height.
    """

    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def summary_from_detections(detections: List[Dict[str, Any]], total_frames: int) -> Dict[str, Any]:
    """Generate summary statistics for a track.

    Args:
        detections: Sequence of per-frame detection dictionaries.
        total_frames: Total number of frames in the clip.

    Returns:
        Dict[str, Any]: Summary metrics (frame coverage, average area, etc.).
    """

    if not detections:
        return {"frames": 0, "coverage": 0.0}
    frames_observed = len(detections)
    widths = [det["bbox"][2] for det in detections if det.get("bbox")]
    heights = [det["bbox"][3] for det in detections if det.get("bbox")]
    area = [w * h for w, h in zip(widths, heights)]
    avg_area = float(np.mean(area)) if area else 0.0
    return {
        "frames": frames_observed,
        "coverage": frames_observed / float(total_frames) if total_frames else 0.0,
        "average_area": avg_area,
    }


@dataclass
class DetectionConfig:
    """Configuration for detection runs."""

    games: Iterable[str] | None
    clips: Iterable[str] | None
    force: bool
    video_exts: Sequence[str]
    dry_run: bool


class PlayerTrackerApp:
    """Facade for detection and assignment flows."""

    def __init__(
        self,
        args: argparse.Namespace,
        tracker_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.args = args
        self.data_root = args.data_root
        self.person_tracks_path = args.person_tracks
        self.player_assignments_path = args.player_assignments
        self.annotations_path = args.annotations
        self.tracker_factory = tracker_factory or default_tracker_factory
        self._temp_videos: List[str] = []  # Track temporary videos for cleanup
        logging.basicConfig(level=getattr(logging, args.log_level))

    # --------------------------------------------------------------------- detect
    def run_detect(self) -> None:
        """Execute YOLO tracking for all requested clips.

        Raises:
            RuntimeError: When tracker dependencies are unavailable.
        """

        tracker = self.tracker_factory()
        person_tracks = load_person_tracks(self.person_tracks_path)
        meta: MutableMapping[str, Any] = (
            person_tracks["meta"]
            if isinstance(person_tracks.get("meta"), MutableMapping)
            else {"version": "1.0.0", "games": {}}
        )
        games_block: MutableMapping[str, Any] = meta.setdefault("games", {})

        config = DetectionConfig(
            games=set(self.args.games) if self.args.games else None,
            clips=set(self.args.clips) if self.args.clips else None,
            force=self.args.force,
            video_exts=tuple(ext.lower() for ext in self.args.video_ext),
            dry_run=self.args.dry_run,
        )

        processed = 0
        for game_id, game_dir in iter_games_clips(self.data_root):
            if config.games and game_id not in config.games:
                continue

            game_record: MutableMapping[str, Any] = games_block.setdefault(game_id, {"clips": {}})
            clips_block: MutableMapping[str, Any] = game_record.setdefault("clips", {})

            for clip_id, clip_dir in iter_clip_dirs(game_dir):
                if config.clips and clip_id not in config.clips:
                    continue
                if not config.force and clip_id in clips_block:
                    LOGGER.info("Skipping %s/%s (cached). Use --force to recompute.", game_id, clip_id)
                    continue

                video_path = self._find_video_path(clip_dir, config.video_exts)
                if not video_path:
                    LOGGER.warning("No video found for %s/%s", game_id, clip_id)
                    continue
                LOGGER.info("Tracking %s/%s via %s", game_id, clip_id, video_path)
                track_history = tracker.track(str(video_path))
                clips_block[clip_id] = self._build_clip_from_history(track_history, str(video_path))
                processed += 1

        if processed == 0:
            LOGGER.info("No clips processed.")
        meta["generated_at"] = iso_timestamp()
        meta["tracker"] = getattr(tracker, "__class__", type(tracker)).__name__

        if not self.args.dry_run:
            write_json_atomic(self.person_tracks_path, meta)
        
        # Clean up temporary videos
        self._cleanup_temp_videos()

    def _cleanup_temp_videos(self) -> None:
        """Clean up any temporary videos generated during processing."""
        import os
        
        for temp_video in self._temp_videos:
            try:
                if os.path.exists(temp_video):
                    os.unlink(temp_video)
                    LOGGER.debug(f"Cleaned up temporary video: {temp_video}")
            except Exception as e:
                LOGGER.warning(f"Failed to cleanup temporary video {temp_video}: {e}")
        self._temp_videos.clear()

    # --------------------------------------------------------------------- assign
    def run_assign(self) -> None:
        """Launch the Matplotlib UI for assigning player identities.

        Raises:
            RuntimeError: If tracking results are missing (detect mode not run).
        """

        person_tracks = load_person_tracks(self.person_tracks_path)
        if not person_tracks.get("index"):
            raise RuntimeError(
                "No tracking data found. Run with --mode detect first to populate person_tracks.json."
            )

        assignments = load_player_assignments(self.player_assignments_path)
        assignments_meta = assignments.get("meta") or {"version": "1.0.0", "games": {}}
        assignments_games: MutableMapping[str, Any] = assignments_meta.setdefault("games", {})

        annotation_cache = read_json(self.annotations_path, default=None)
        if isinstance(annotation_cache, MutableMapping):
            annotation_games = annotation_cache.setdefault("games", {})
        else:
            annotation_cache = {"schema_version": "1.0.0", "games": {}}
            annotation_games = annotation_cache["games"]

        filtered_clips = self._filtered_clip_keys(person_tracks["index"])
        meta_games: MutableMapping[str, Any] = {}
        if isinstance(person_tracks.get("meta"), MutableMapping):
            games_candidate = person_tracks["meta"].get("games", {}) or {}
            if isinstance(games_candidate, MutableMapping):
                meta_games = games_candidate
        ui_payload = []
        for game_id, clip_id in filtered_clips:
            track_index = person_tracks["index"].get(game_id, {}).get(clip_id, {})
            if not isinstance(track_index, MutableMapping):
                track_index = {}

            game_meta = meta_games.get(game_id, {})
            if not isinstance(game_meta, MutableMapping):
                game_meta = {}
            clips_meta = game_meta.get("clips", {})
            if not isinstance(clips_meta, MutableMapping):
                clips_meta = {}
            clip_entry: MutableMapping[str, Any] | Dict[str, Any] = clips_meta.get(clip_id, {})
            if not isinstance(clip_entry, MutableMapping):
                clip_entry = {}
            if not clip_entry.get("tracks") and track_index:
                clip_entry = {"tracks": dict(track_index)}
            clip_dir = (self.data_root / game_id / clip_id).resolve()
            source_value = clip_entry.get("source")
            source_path = None
            if source_value:
                candidate = Path(source_value)
                if not candidate.is_absolute():
                    candidate = (self.data_root / candidate).resolve()
                if candidate.exists():
                    source_path = candidate
            ui_payload.append(
                {
                    "game_id": game_id,
                    "clip_id": clip_id,
                    "clip_data": clip_entry,
                    "assignment": assignments["index"].get(game_id, {}).get(clip_id),
                    "frame_dir": str(clip_dir) if clip_dir.is_dir() else None,
                    "media_source": str(source_path) if source_path else None,
                }
            )
        if not ui_payload:
            LOGGER.warning("No clips match the requested filters.")
            return

        def on_save(game_id: str, clip_id: str, selected_tracks: List[str]) -> None:
            LOGGER.info("Saving assignment for %s/%s: %s", game_id, clip_id, selected_tracks)
            game_block: MutableMapping[str, Any] = assignments_games.setdefault(game_id, {"clips": {}})
            clips_block: MutableMapping[str, Any] = game_block.setdefault("clips", {})
            clips_block[clip_id] = {
                "source": person_tracks["index"][game_id][clip_id].get("source"),
                "players": self._build_players_from_selection(game_id, clip_id, selected_tracks, person_tracks),
                "selected_tracks": selected_tracks,
                "assigned_by": getpass.getuser(),
                "last_reviewed": iso_timestamp(),
            }
            assignments["index"].setdefault(game_id, {})[clip_id] = clips_block[clip_id]
            if not self.args.dry_run:
                write_json_atomic(self.player_assignments_path, assignments_meta)

            # Update annotation cache if available.
            game_block = annotation_games.setdefault(game_id, {"clips": {}})
            clips_block = game_block.setdefault("clips", {})
            clip_entry = clips_block.setdefault(
                clip_id,
                {
                    "source": person_tracks["index"][game_id][clip_id].get("source"),
                    "ball": {"frames": [], "x": [], "y": [], "visibility": [], "status": []},
                },
            )
            clip_entry["players"] = self._build_players_from_selection(
                game_id, clip_id, selected_tracks, person_tracks, as_sequences=True
            )
            if not self.args.dry_run:
                annotation_cache.setdefault("schema_version", "1.0.0")
                annotation_cache["generated_at"] = iso_timestamp()
                write_json_atomic(self.annotations_path, annotation_cache)

        ui = PlayerIdentifierUI(
            clips=ui_payload,
            save_callback=on_save,
            roles=["player_a", "player_b"],
        )
        ui.launch()

    # --------------------------------------------------------------------- helpers
    def _filtered_clip_keys(
        self,
        index: Mapping[str, Mapping[str, Any]],
    ) -> List[tuple[str, str]]:
        """Return filtered ``(game, clip)`` pairs.

        Args:
            index: Mapping of games to their clip dictionaries.

        Returns:
            List[Tuple[str, str]]: Filtered identifiers respecting CLI args.
        """

        result: List[tuple[str, str]] = []
        games_filter = set(self.args.games) if self.args.games else None
        clips_filter = set(self.args.clips) if self.args.clips else None
        for game_id, clips in index.items():
            if games_filter and game_id not in games_filter:
                continue
            for clip_id in clips.keys():
                if clips_filter and clip_id not in clips_filter:
                    continue
                result.append((game_id, clip_id))
        return result

    def _find_video_path(self, clip_dir: Path, extensions: Sequence[str]) -> Path | None:
        """Locate a video inside ``clip_dir`` or generate from frame sequence.

        Args:
            clip_dir: Directory containing video assets or frame sequences.
            extensions: Ordered list of candidate extensions (``.mp4`` etc.).

        Returns:
            Optional[Path]: Path to video file (existing or generated).
        """
        # First try to find existing video file
        for candidate in sorted(clip_dir.iterdir()):
            if candidate.suffix.lower() in extensions and candidate.is_file():
                return candidate
        
        # If no video found, check for frame sequence and generate video
        is_valid, issues = validate_frame_sequence(clip_dir)
        if is_valid:
            LOGGER.info("No video found in %s, generating from frame sequence", clip_dir)
            try:
                # Generate video without automatic cleanup
                with generate_video_from_frames(clip_dir, cleanup=False) as temp_video_path:
                    if temp_video_path:
                        # Store the path for cleanup later
                        self._temp_videos.append(temp_video_path)
                        return Path(temp_video_path)
                    else:
                        LOGGER.error("Video generation returned None")
                        return None
            except Exception as e:
                LOGGER.error(f"Failed to generate video from frames in {clip_dir}: {e}")
                return None
        else:
            LOGGER.debug("Frame sequence validation failed for %s: %s", clip_dir, issues)
            
        return None

    def _build_clip_from_history(self, history: List[List[Dict[str, Any]]], source: str) -> Dict[str, Any]:
        """Convert tracker history into serialisable clip payload.

        Args:
            history: Tracker output comprising per-frame detections.
            source: Original video source path.

        Returns:
            Dict[str, Any]: Clip payload ready for JSON serialisation.
        """

        tracks: Dict[str, Dict[str, Any]] = {}
        for frame_id, detections in enumerate(history):
            for det in detections:
                track_id = str(det.get("id"))
                # Handle bbox extraction properly for numpy arrays
                bbox = det.get("bbx_xyxy")
                if bbox is None:
                    bbox = det.get("bbox")
                if bbox is None:
                    continue
                xywh = convert_bbox_xyxy_to_xywh(bbox)
                entry = tracks.setdefault(
                    track_id,
                    {"label": det.get("label", "person"), "detections": [], "summary": {}},
                )
                entry["detections"].append(
                    {
                        "frame": frame_id,
                        "bbox": xywh,
                        "confidence": det.get("confidence"),
                        "visibility": float(det.get("visibility", 1.0)),
                    }
                )
        for track_id, track in tracks.items():
            track["summary"] = summary_from_detections(track["detections"], total_frames=len(history))
        return {
            "source": source,
            "frames": len(history),
            "tracks": tracks,
        }

    def _build_players_from_selection(
        self,
        game_id: str,
        clip_id: str,
        selected_tracks: List[str],
        person_tracks: Mapping[str, Any],
        *,
        as_sequences: bool = False,
    ) -> Dict[str, Any]:
        """Build player payload for assignments or annotation cache.

        Args:
            game_id: Game identifier.
            clip_id: Clip identifier.
            selected_tracks: Ordered list of chosen track identifiers.
            person_tracks: Mapping generated by ``load_person_tracks``.
            as_sequences: When ``True`` return full per-frame sequences.

        Returns:
            Dict[str, Any]: Player mapping keyed by semantic role.
        """

        selected_tracks = [track_id for track_id in selected_tracks if track_id]
        if len(selected_tracks) < 2:
            # Maintain existing ordering when insufficient selections.
            return {}
        assignments = {
            "index": {
                game_id: {
                    clip_id: {
                        "players": {
                            "player_a": {"track_id": selected_tracks[0], "assigned_by": getpass.getuser()},
                            "player_b": {"track_id": selected_tracks[1], "assigned_by": getpass.getuser()},
                        }
                    }
                }
            }
        }
        payload = build_player_payload(
            assignments["index"],
            person_tracks["index"],
            game_id,
            clip_id,
        )
        if not as_sequences:
            return {
                role: {
                    "track_id": data["track_id"],
                    "assigned_by": getpass.getuser(),
                    "frames": data["frames"],
                    "bbox": data["bbox"],
                    "visibility": data.get("visibility"),
                }
                for role, data in payload.items()
            }
        return payload


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for CLI usage.

    Args:
        argv: Optional raw argument list.
    """

    args = parse_args(argv)
    app = PlayerTrackerApp(args)
    if args.mode == "detect":
        app.run_detect()
    elif args.mode == "assign":
        app.run_assign()
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
