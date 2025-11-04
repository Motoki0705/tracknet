"""Video generation utility for converting frame sequences to temporary videos.

This module provides functionality to convert JPG frame sequences to MP4 videos
for YOLOv8 processing, with automatic cleanup of temporary files.
"""

import contextlib
import logging
import tempfile
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def generate_video_from_frames(
    frames_dir: Path, fps: int = 30, cleanup: bool = True
) -> str | None:
    """Generate temporary MP4 video from JPG frame sequence.

    Args:
        frames_dir: Directory containing JPG frames with numeric names
        fps: Frames per second for output video
        cleanup: Whether to automatically delete the temporary video

    Yields:
        Path to temporary video file, or None if generation failed

    Example:
        with generate_video_from_frames(Path("data/tracknet/game1/Clip1")) as video_path:
            if video_path:
                # Use video for tracking
                results = tracker.track(video_path)
            # Video automatically cleaned up if cleanup=True
    """
    temp_video_path = None

    try:
        # Find all JPG frames in the directory
        frame_files = sorted(
            [f for f in frames_dir.glob("*.jpg") if f.is_file()],
            key=lambda x: int(x.stem),
        )

        if not frame_files:
            logger.warning(f"No JPG frames found in {frames_dir}")
            yield None
            return

        # Read first frame to get video dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            logger.error(f"Could not read frame {frame_files[0]}")
            yield None
            return

        height, width = first_frame.shape[:2]

        # Create temporary video file
        temp_fd, temp_video = tempfile.mkstemp(suffix=".mp4")
        temp_video_path = temp_video
        # Close file descriptor since OpenCV will handle the file
        import os

        os.close(temp_fd)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

        if not video_writer.isOpened():
            logger.error("Failed to initialize video writer")
            yield None
            return

        logger.info(f"Generating video from {len(frame_files)} frames in {frames_dir}")

        # Write all frames to video
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                logger.warning(f"Could not read frame {frame_file}, skipping")
                continue
            video_writer.write(frame)

        video_writer.release()

        # Verify video was created successfully
        if Path(temp_video_path).stat().st_size == 0:
            logger.error("Generated video file is empty")
            yield None
            return

        logger.info(f"Successfully generated temporary video: {temp_video_path}")
        yield temp_video_path

    except Exception as e:
        logger.error(f"Error generating video from {frames_dir}: {e}")
        yield None
    finally:
        # Clean up temporary file if requested and it exists
        if cleanup and temp_video_path and Path(temp_video_path).exists():
            try:
                Path(temp_video_path).unlink()
                logger.debug(f"Cleaned up temporary video: {temp_video_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup temporary video {temp_video_path}: {e}"
                )


def get_frame_count(frames_dir: Path) -> int:
    """Get the number of JPG frames in a directory.

    Args:
        frames_dir: Directory containing JPG frames

    Returns:
        Number of frames found
    """
    frame_files = [f for f in frames_dir.glob("*.jpg") if f.is_file()]
    return len(frame_files)


def validate_frame_sequence(frames_dir: Path) -> tuple[bool, list[str]]:
    """Validate that frame sequence is complete and properly ordered.

    Args:
        frames_dir: Directory containing JPG frames

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check if directory exists
    if not frames_dir.exists():
        issues.append(f"Directory does not exist: {frames_dir}")
        return False, issues

    # Find all JPG frames
    frame_files = [f for f in frames_dir.glob("*.jpg") if f.is_file()]

    if not frame_files:
        issues.append("No JPG frames found")
        return False, issues

    # Try to sort frames numerically
    try:
        sorted_frames = sorted(frame_files, key=lambda x: int(x.stem))
    except ValueError:
        issues.append("Frame filenames are not numeric")
        return False, issues

    # Check for gaps in sequence
    frame_numbers = [int(f.stem) for f in sorted_frames]
    expected_numbers = list(range(min(frame_numbers), max(frame_numbers) + 1))

    missing_numbers = set(expected_numbers) - set(frame_numbers)
    if missing_numbers:
        issues.append(f"Missing frames: {sorted(missing_numbers)}")

    # Try to read first and last frames
    try:
        first_frame = cv2.imread(str(sorted_frames[0]))
        last_frame = cv2.imread(str(sorted_frames[-1]))

        if first_frame is None:
            issues.append(f"Cannot read first frame: {sorted_frames[0]}")
        if last_frame is None:
            issues.append(f"Cannot read last frame: {sorted_frames[-1]}")

        # Check dimensions match
        if (first_frame is not None and last_frame is not None and first_frame.shape != last_frame.shape):
                issues.append("Frame dimensions are not consistent")

    except Exception as e:
        issues.append(f"Error reading frames: {e}")

    return len(issues) == 0, issues
