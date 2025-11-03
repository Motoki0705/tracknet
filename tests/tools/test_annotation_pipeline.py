import argparse
import json
from pathlib import Path

import pytest

from tracknet.tools.annotation_common import (
    COURT_KEYPOINTS,
    COURT_SKELETON_EDGES,
    read_json,
    write_json_atomic,
)
from tracknet.tools.annotation_converter import convert_annotations, parse_args as parse_converter_args
from tracknet.tools.player_tracker import PlayerTrackerApp, parse_args as parse_tracker_args


def test_yolov8_load():
    """Ensure YOLO imports correctly (skip when unavailable)."""

    ultralytics = pytest.importorskip("ultralytics")
    try:
        ultralytics.YOLO()  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - depends on environment assets
        pytest.skip(f"YOLO weights unavailable: {exc}")


class _StubTracker:
    """Minimal tracker stub returning deterministic detections."""

    def __init__(self, counter):
        self.counter = counter

    def track(self, _video_path):
        self.counter["calls"] += 1
        return [
            [
                {"id": 1, "bbx_xyxy": [0.0, 0.0, 10.0, 20.0]},
                {"id": 2, "bbx_xyxy": [100.0, 50.0, 120.0, 120.0]},
            ],
            [
                {"id": 1, "bbx_xyxy": [1.0, 2.0, 12.0, 22.0]},
            ],
        ]


@pytest.fixture()
def tmp_dataset(tmp_path):
    data_root = tmp_path / "data" / "tracknet" / "game1" / "Clip1"
    data_root.mkdir(parents=True)
    (data_root / "clip.mp4").write_bytes(b"fake-video")
    label_path = data_root / "Label.csv"
    label_path.write_text(
        "file name,visibility,x-coordinate,y-coordinate,status\n"
        "0000.jpg,1,100,200,0\n"
        "0001.jpg,2,101,205,1\n",
        encoding="utf-8",
    )
    return tmp_path


def _player_tracker_args(tmp_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        mode="detect",
        data_root=tmp_dir / "data" / "tracknet",
        person_tracks=tmp_dir / "outputs" / "tracking" / "person_tracks.json",
        player_assignments=tmp_dir / "outputs" / "tracking" / "player_assignments.json",
        annotations=tmp_dir / "data" / "tracknet" / "annotations.json",
        games=None,
        clips=None,
        force=False,
        dry_run=False,
        video_ext=[".mp4"],
        log_level="ERROR",
    )


def test_player_tracker_detect_and_cache(tmp_dataset):
    counter = {"calls": 0}
    args = _player_tracker_args(tmp_dataset)
    app = PlayerTrackerApp(args, tracker_factory=lambda: _StubTracker(counter))
    app.run_detect()

    person_tracks = read_json(args.person_tracks)
    assert person_tracks["games"]["game1"]["clips"]["Clip1"]["tracks"]["1"]["detections"]
    assert counter["calls"] == 1

    # Second run without --force should reuse cache.
    app = PlayerTrackerApp(args, tracker_factory=lambda: _StubTracker(counter))
    app.run_detect()
    assert counter["calls"] == 1  # unchanged


def test_annotation_converter_idempotent(tmp_dataset):
    counter = {"calls": 0}
    args = _player_tracker_args(tmp_dataset)
    app = PlayerTrackerApp(args, tracker_factory=lambda: _StubTracker(counter))
    app.run_detect()

    person_tracks = read_json(args.person_tracks)
    assignments = {
        "version": "1.0.0",
        "games": {
            "game1": {
                "clips": {
                    "Clip1": {
                        "source": "game1/Clip1",
                        "selected_tracks": ["1", "2"],
                        "assigned_by": "tester",
                        "last_reviewed": "1970-01-01T00:00:00Z",
                        "players": {
                            "player_a": {
                                "track_id": "1",
                                "frames": [0, 1],
                                "bbox": [[0, 0, 10, 20], [1, 2, 11, 20]],
                                "visibility": [1.0, 1.0],
                            },
                            "player_b": {
                                "track_id": "2",
                                "frames": [0],
                                "bbox": [[100, 50, 20, 70]],
                                "visibility": [1.0],
                            },
                        },
                    }
                }
            }
        },
    }
    write_json_atomic(args.player_assignments, assignments)

    court_payload = {
        "version": "1.0.0",
        "updated_at": "1970-01-01T00:00:00Z",
        "games": {
            "game1": {
                "annotator": "tester",
                "keypoints": [
                    {"index": kp["index"], "name": kp["name"], "x": 0.0, "y": 0.0} for kp in COURT_KEYPOINTS
                ],
                "skeleton": COURT_SKELETON_EDGES,
            }
        },
    }
    write_json_atomic(tmp_dataset / "outputs" / "court_annotations" / "games.json", court_payload)

    player_assignments = read_json(args.player_assignments)
    assert player_assignments["games"]["game1"]["clips"]["Clip1"]["players"]["player_a"]["track_id"] == "1"

    court_annotations = read_json(tmp_dataset / "outputs" / "court_annotations" / "games.json")
    assert len(court_annotations["games"]["game1"]["keypoints"]) == len(COURT_KEYPOINTS)
    assert court_annotations["games"]["game1"]["skeleton"] == COURT_SKELETON_EDGES

    converter_args = parse_converter_args(
        [
            "--data-root",
            str(args.data_root),
            "--person-tracks",
            str(args.person_tracks),
            "--player-assignments",
            str(args.player_assignments),
            "--court-annotations",
            str(tmp_dataset / "outputs" / "court_annotations" / "games.json"),
            "--output",
            str(args.annotations),
        ]
    )
    payload_first = convert_annotations(converter_args)
    first_written = read_json(args.annotations)
    convert_annotations(converter_args)
    second_written = read_json(args.annotations)

    assert payload_first["games"]["game1"]["clips"]["Clip1"]["ball"]["visibility"] == [1, 2]
    assert payload_first["games"]["game1"]["clips"]["Clip1"]["ball"]["status"] == [0, 1]
    assert first_written == second_written


def test_player_tracker_cli_mode_validation():
    with pytest.raises(SystemExit):
        parse_tracker_args(["--mode", "invalid"])


def test_schema_validation_helpers(tmp_dataset):
    counter = {"calls": 0}
    args = _player_tracker_args(tmp_dataset)
    app = PlayerTrackerApp(args, tracker_factory=lambda: _StubTracker(counter))
    app.run_detect()

    person_tracks = read_json(args.person_tracks)
    clip = person_tracks["games"]["game1"]["clips"]["Clip1"]
    assert clip["frames"] == 2
    assert set(clip["tracks"].keys()) == {"1", "2"}


def test_converter_multi_stage(tmp_dataset):
    counter = {"calls": 0}
    args = _player_tracker_args(tmp_dataset)
    app = PlayerTrackerApp(args, tracker_factory=lambda: _StubTracker(counter))
    app.run_detect()

    converter_args = parse_converter_args(
        [
            "--data-root",
            str(args.data_root),
            "--person-tracks",
            str(args.person_tracks),
            "--player-assignments",
            str(args.player_assignments),
            "--court-annotations",
            str(tmp_dataset / "outputs" / "court_annotations" / "games.json"),
            "--output",
            str(args.annotations),
        ]
    )

    # Stage 1: only ball data.
    payload_ball_only = convert_annotations(converter_args)
    clip_payload = payload_ball_only["games"]["game1"]["clips"]["Clip1"]
    assert "players" not in clip_payload

    # Stage 2: add player assignments.
    assignments = {
        "version": "1.0.0",
        "games": {
            "game1": {
                "clips": {
                    "Clip1": {
                        "source": "game1/Clip1",
                        "selected_tracks": ["1", "2"],
                        "assigned_by": "tester",
                        "last_reviewed": "1970-01-01T00:00:00Z",
                        "players": {
                            "player_a": {"track_id": "1"},
                            "player_b": {"track_id": "2"},
                        },
                    }
                }
            }
        },
    }
    write_json_atomic(args.player_assignments, assignments)
    payload_with_players = convert_annotations(converter_args)
    assert "players" in payload_with_players["games"]["game1"]["clips"]["Clip1"]

    # Stage 3: add court annotations.
    court_payload = {
        "version": "1.0.0",
        "updated_at": "1970-01-01T00:00:00Z",
        "games": {
            "game1": {
                "annotator": "tester",
                "keypoints": [
                    {"index": kp["index"], "name": kp["name"], "x": 0.0, "y": 0.0}
                    for kp in COURT_KEYPOINTS
                ],
                "skeleton": COURT_SKELETON_EDGES,
            }
        },
    }
    write_json_atomic(tmp_dataset / "outputs" / "court_annotations" / "games.json", court_payload)
    payload_full = convert_annotations(converter_args)
    assert payload_full["games"]["game1"]["court"]["keypoints"][0]["name"] == COURT_KEYPOINTS[0]["name"]
