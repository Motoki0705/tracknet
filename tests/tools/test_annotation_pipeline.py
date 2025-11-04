import argparse
from pathlib import Path

import cv2
import numpy as np
import pytest

from tracknet.tools.annotation_common import (
    COURT_KEYPOINTS,
    COURT_SKELETON_EDGES,
    read_json,
    write_json_atomic,
)
from tracknet.tools.annotation_converter import convert_annotations
from tracknet.tools.annotation_converter import parse_args as parse_converter_args
from tracknet.tools.player_tracker import PlayerTrackerApp
from tracknet.tools.player_tracker import parse_args as parse_tracker_args
from tracknet.tools.utils.video_generator import (
    generate_video_from_frames,
    get_frame_count,
    validate_frame_sequence,
)


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

    def track(self, video_path):
        """Return tracking results based on video length."""
        self.counter["calls"] += 1

        # Get actual video length to match frame count
        import cv2

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            # For invalid video files (like fake video in tests), default to 2 frames
            frame_count = 2

        # Generate tracking results for each frame
        results = []
        for frame_idx in range(frame_count):
            frame_detections = [
                {"id": 1, "bbx_xyxy": [0.0, 0.0, 10.0, 20.0]},
                {"id": 2, "bbx_xyxy": [100.0, 50.0, 120.0, 120.0]},
            ]
            # Vary detection positions slightly per frame
            if frame_idx > 0:
                frame_detections[0]["bbx_xyxy"][0] += frame_idx
                frame_detections[0]["bbx_xyxy"][1] += frame_idx
            results.append(frame_detections)

        return results


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
    assert person_tracks["games"]["game1"]["clips"]["Clip1"]["tracks"]["1"][
        "detections"
    ]
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

    read_json(args.person_tracks)
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
                    {"index": kp["index"], "name": kp["name"], "x": 0.0, "y": 0.0}
                    for kp in COURT_KEYPOINTS
                ],
                "skeleton": COURT_SKELETON_EDGES,
            }
        },
    }
    write_json_atomic(
        tmp_dataset / "outputs" / "court_annotations" / "games.json", court_payload
    )

    player_assignments = read_json(args.player_assignments)
    assert (
        player_assignments["games"]["game1"]["clips"]["Clip1"]["players"]["player_a"][
            "track_id"
        ]
        == "1"
    )

    court_annotations = read_json(
        tmp_dataset / "outputs" / "court_annotations" / "games.json"
    )
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

    assert payload_first["games"]["game1"]["clips"]["Clip1"]["ball"]["visibility"] == [
        1,
        2,
    ]
    assert payload_first["games"]["game1"]["clips"]["Clip1"]["ball"]["status"] == [0, 1]
    # Compare payloads excluding the generated_at timestamp
    first_copy = first_written.copy()
    second_copy = second_written.copy()
    first_copy.pop("generated_at", None)
    second_copy.pop("generated_at", None)
    assert first_copy == second_copy


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
    write_json_atomic(
        tmp_dataset / "outputs" / "court_annotations" / "games.json", court_payload
    )
    payload_full = convert_annotations(converter_args)
    assert (
        payload_full["games"]["game1"]["court"]["keypoints"][0]["name"]
        == COURT_KEYPOINTS[0]["name"]
    )


@pytest.fixture()
def tmp_frame_sequence(tmp_path):
    """Create a temporary directory with sample JPG frames."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True)

    # Create 5 test frames with different colors
    for i in range(5):
        frame = np.full((480, 640, 3), i * 50, dtype=np.uint8)
        frame_path = frames_dir / f"{i + 1:04d}.jpg"  # 0001.jpg, 0002.jpg, etc.
        cv2.imwrite(str(frame_path), frame)

    return frames_dir


def test_get_frame_count(tmp_frame_sequence):
    """Test frame counting functionality."""
    count = get_frame_count(tmp_frame_sequence)
    assert count == 5


def test_validate_frame_sequence_valid(tmp_frame_sequence):
    """Test validation of a valid frame sequence."""
    is_valid, issues = validate_frame_sequence(tmp_frame_sequence)
    assert is_valid
    assert len(issues) == 0


def test_validate_frame_sequence_empty_dir(tmp_path):
    """Test validation of an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    is_valid, issues = validate_frame_sequence(empty_dir)
    assert not is_valid
    assert "No JPG frames found" in issues


def test_validate_frame_sequence_missing_frames(tmp_path):
    """Test validation of frame sequence with gaps."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    # Create frames with gaps (0001.jpg, 0003.jpg, missing 0002.jpg)
    for i in [1, 3]:
        frame = np.full((480, 640, 3), i * 50, dtype=np.uint8)
        frame_path = frames_dir / f"{i:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)

    is_valid, issues = validate_frame_sequence(frames_dir)
    assert not is_valid
    assert any("Missing frames" in issue for issue in issues)


def test_generate_video_from_frames(tmp_frame_sequence):
    """Test video generation from frame sequence."""
    with generate_video_from_frames(tmp_frame_sequence, cleanup=True) as video_path:
        assert video_path is not None
        assert Path(video_path).exists()

        # Verify video can be opened and has correct properties
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened()

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count == 5

        fps = cap.get(cv2.CAP_PROP_FPS)
        assert fps == 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert width == 640
        assert height == 480

        cap.release()

    # Verify cleanup happened
    assert not Path(video_path).exists()


def test_generate_video_from_frames_no_frames(tmp_path):
    """Test video generation with no frames."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with generate_video_from_frames(empty_dir, cleanup=True) as video_path:
        assert video_path is None


def test_generate_video_from_frames_custom_fps(tmp_frame_sequence):
    """Test video generation with custom FPS."""
    with generate_video_from_frames(
        tmp_frame_sequence, fps=25, cleanup=True
    ) as video_path:
        assert video_path is not None

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert fps == 25.0
        cap.release()


def test_player_tracker_with_frame_sequence(tmp_path):
    """Test that player tracker can generate videos from frame sequences."""
    # Create dataset with frame sequences instead of videos
    data_root = tmp_path / "data" / "tracknet" / "game1" / "Clip1"
    data_root.mkdir(parents=True)

    # Create frame sequence
    for i in range(3):
        frame = np.full((480, 640, 3), i * 50, dtype=np.uint8)
        frame_path = data_root / f"{i + 1:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)

    # Create ball annotation CSV
    label_path = data_root / "ball.csv"
    label_path.write_text(
        "file name,visibility,x-coordinate,y-coordinate,status\n"
        "0001.jpg,1,100,200,0\n"
        "0002.jpg,2,101,205,1\n"
        "0003.jpg,1,102,210,0\n",
        encoding="utf-8",
    )

    counter = {"calls": 0}
    args = _player_tracker_args(tmp_path)
    app = PlayerTrackerApp(args, tracker_factory=lambda: _StubTracker(counter))
    app.run_detect()

    # Verify tracking was performed
    assert counter["calls"] == 1

    # Verify results were saved
    person_tracks = read_json(args.person_tracks)
    assert "game1" in person_tracks["games"]
    assert "Clip1" in person_tracks["games"]["game1"]["clips"]

    # Verify source points to generated video (should be temp file that's now cleaned up)
    clip_data = person_tracks["games"]["game1"]["clips"]["Clip1"]
    assert clip_data["frames"] == 3  # Should match number of frames
    assert len(clip_data["tracks"]) > 0
