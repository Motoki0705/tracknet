#!/usr/bin/env python3
"""
Batch player tracking for all games and clips in TrackNet dataset.

This script performs player tracking on all clips in the dataset
and saves the results for later ID selection and COCO conversion.
"""

import argparse
import json
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort


class PlayerTracker:
    """Player tracking using YOLOv8 + DeepSORT."""
    
    def __init__(self, model_path: str = "yolov8x.pt"):
        """Initialize player tracker.
        
        Args:
            model_path: Path to YOLO model.
        """
        self.model = YOLO(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize DeepSORT
        self.deepsort = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.2,
            nn_budget=100,
            embedder="torchreid",
            embedder_gpu=True if torch.cuda.is_available() else False,
        )
    
    def track_video(self, video_path: Path) -> List[List[Dict]]:
        """Track players in video.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            List of tracking results per frame.
        """
        cap = cv2.VideoCapture(str(video_path))
        track_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect players
            results = self.model(frame, classes=[0])  # person class
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        if conf > 0.5:  # confidence threshold
                            detections.append(
                                ([x1, y1, x2 - x1, y2 - y1], conf, "person")
                            )
            
            # Update tracks
            tracks = self.deepsort.update_tracks(detections, frame=frame)
            
            # Format results
            frame_results = []
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                bbox = track.to_tlbr()
                track_id = track.track_id
                
                frame_results.append({
                    "id": track_id,
                    "bbx_xyxy": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                })
            
            track_history.append(frame_results)
        
        cap.release()
        return track_history
    
    def get_video_info(self, video_path: Path) -> Tuple[int, int, int]:
        """Get video information.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Tuple of (num_frames, width, height).
        """
        cap = cv2.VideoCapture(str(video_path))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return num_frames, width, height
    
    def save_tracking_results(self, track_history: List[List[Dict]], output_path: Path) -> None:
        """Save tracking results to JSON file.
        
        Args:
            track_history: Tracking results.
            output_path: Output file path.
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = []
        for frame in track_history:
            frame_data = []
            for track in frame:
                track_data = {
                    "id": track["id"],
                    "bbx_xyxy": [int(x) for x in track["bbx_xyxy"]]
                }
                frame_data.append(track_data)
            serializable_history.append(frame_data)
        
        with open(output_path, "w") as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"Tracking results saved to: {output_path}")
    
    def select_track_ids(self, track_history: List[List[Dict]]) -> List[int]:
        """Select track IDs using UI.
        
        Args:
            track_history: Tracking results.
            
        Returns:
            List of selected track IDs.
        """
        # Import UI components here to avoid circular imports
        from tracknet.tools.utils.ui.player_selector import PlayerSelectorUI
        
        ui = PlayerSelectorUI(track_history)
        selected_ids = ui.run()
        return selected_ids
    
    def extract_track_ids(
        self, 
        video_path: Path, 
        output_dir: Path, 
        launch_ui: bool = True
    ) -> List[int]:
        """Extract and select track IDs from video.
        
        Args:
            video_path: Path to video file.
            output_dir: Output directory.
            launch_ui: Whether to launch UI for selection.
            
        Returns:
            List of selected track IDs.
        """
        print(f"Extracting track IDs from: {video_path}")
        
        # Track players
        track_history = self.track_video(video_path)
        
        # Get video info
        num_frames, width, height = self.get_video_info(video_path)
        print(f"Video info: {num_frames} frames, {width}x{height}")
        
        # Count unique tracks
        unique_ids = set()
        for frame in track_history:
            for track in frame:
                unique_ids.add(track["id"])
        print(f"Found {len(unique_ids)} unique track IDs: {sorted(unique_ids)}")
        
        # Save tracking results
        output_dir.mkdir(parents=True, exist_ok=True)
        video_name = video_path.stem
        tracking_file = output_dir / f"{video_name}_tracking.json"
        self.save_tracking_results(track_history, tracking_file)
        
        # Select track IDs
        if launch_ui:
            selected_ids = self.select_track_ids(track_history)
        else:
            # Auto-select all tracks
            selected_ids = list(unique_ids)
            print(f"Auto-selected all {len(selected_ids)} track IDs")
        
        # Save selected IDs
        selected_file = output_dir / f"{video_name}_selected.json"
        with open(selected_file, "w") as f:
            json.dump(selected_ids, f, indent=2)
        
        print(f"Selected track IDs saved to: {selected_file}")
        return selected_ids


def find_all_clips(dataset_root: Path) -> List[Tuple[str, str, Path]]:
    """Find all clips in the dataset.
    
    Args:
        dataset_root: Path to the dataset root directory.
        
    Returns:
        List of (game_name, clip_name, clip_path) tuples.
    """
    clips = []
    
    # Find all game directories
    game_dirs = [d for d in dataset_root.iterdir() 
                if d.is_dir() and d.name.startswith("game")]
    
    for game_dir in sorted(game_dirs):
        game_name = game_dir.name
        
        # Find all clip directories in the game
        clip_dirs = [d for d in game_dir.iterdir() 
                    if d.is_dir() and d.name.startswith("Clip")]
        
        for clip_dir in sorted(clip_dirs):
            clip_name = clip_dir.name
            clips.append((game_name, clip_name, clip_dir))
    
    return clips


def create_video_from_images(image_files: List[Path], output_video_path: Path, fps: int = 30) -> bool:
    """Create video from image sequence.
    
    Args:
        image_files: List of image file paths (sorted).
        output_video_path: Path to save the output video.
        fps: Frames per second for the output video.
        
    Returns:
        True if successful, False otherwise.
    """
    if not image_files:
        return False
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        return False
    
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        return False
    
    # Write all frames
    for image_file in image_files:
        frame = cv2.imread(str(image_file))
        if frame is not None:
            video_writer.write(frame)
    
    video_writer.release()
    return True


def track_all_clips(
    dataset_root: Path, 
    output_dir: Path, 
    model_path: str = "yolov8x.pt",
    force: bool = False
) -> Dict[str, Dict[str, str]]:
    """Track players in all clips.
    
    Args:
        dataset_root: Path to the dataset root directory.
        output_dir: Directory to save tracking results.
        model_path: YOLO model path.
        force: Whether to overwrite existing results.
        
    Returns:
        Dictionary mapping game/clip to output file paths.
    """
    # Find all clips
    clips = find_all_clips(dataset_root)
    print(f"Found {len(clips)} clips across {len(set(c[0] for c in clips))} games")
    
    # Initialize tracker
    print(f"Initializing player tracker with model: {model_path}")
    tracker = PlayerTracker(model_path)
    
    # Results tracking
    results = {}
    processed = 0
    skipped = 0
    no_video = 0
    
    for game_name, clip_name, clip_path in clips:
        clip_output_dir = output_dir / game_name / clip_name
        tracking_file = clip_output_dir / f"{clip_name}_tracking.json"
        
        # Check if already processed
        if tracking_file.exists() and not force:
            print(f"Skipping {game_name}/{clip_name} (already exists)")
            results[f"{game_name}/{clip_name}"] = {
                "tracking_file": str(tracking_file),
                "selected_file": str(clip_output_dir / f"{clip_name}_selected.json")
            }
            skipped += 1
            continue
        
        print(f"Processing {game_name}/{clip_name} ({processed + skipped + no_video + 1}/{len(clips)})")
        
        # Find video file in clip directory
        video_files = list(clip_path.glob("*.mp4")) + list(clip_path.glob("*.avi"))
        
        if not video_files:
            # Try to create video from image sequence
            image_files = sorted(list(clip_path.glob("*.jpg")) + list(clip_path.glob("*.png")))
            if not image_files:
                print(f"Warning: No video or image files found in {clip_path}")
                no_video += 1
                continue
            
            print(f"  - Found {len(image_files)} image files, creating video sequence")
            
            # Create temporary video from images
            temp_video_path = clip_output_dir / f"{clip_name}_temp.mp4"
            clip_output_dir.mkdir(parents=True, exist_ok=True)
            
            if not create_video_from_images(image_files, temp_video_path):
                print(f"  - Failed to create video from images, skipping")
                no_video += 1
                continue
            
            video_path = temp_video_path
            is_temp_video = True
        else:
            video_path = video_files[0]
            is_temp_video = False
        
        try:
            # Track players
            track_history = tracker.track_video(video_path)
            
            # Create output directory
            clip_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save tracking results
            tracker.save_tracking_results(track_history, tracking_file)
            
            # Get video info
            num_frames, width, height = tracker.get_video_info(video_path)
            
            # Count unique tracks
            unique_ids = set()
            for frame in track_history:
                for track in frame:
                    unique_ids.add(track["id"])
            
            print(f"  - Video: {num_frames} frames, {width}x{height}")
            print(f"  - Found {len(unique_ids)} unique track IDs: {sorted(unique_ids)}")
            
            # Clean up temporary video
            if is_temp_video and temp_video_path.exists():
                temp_video_path.unlink()
            
            # Store results
            results[f"{game_name}/{clip_name}"] = {
                "tracking_file": str(tracking_file),
                "selected_file": str(clip_output_dir / f"{clip_name}_selected.json"),
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "unique_tracks": len(unique_ids),
                "track_ids": sorted(unique_ids)
            }
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {game_name}/{clip_name}: {e}")
            # Clean up temporary video on error
            if is_temp_video and temp_video_path.exists():
                temp_video_path.unlink()
            continue
    
    print(f"\nBatch tracking completed:")
    print(f"  - Processed: {processed} clips")
    print(f"  - Skipped: {skipped} clips (already exists)")
    print(f"  - No video: {no_video} clips")
    print(f"  - Total: {len(clips)} clips")
    
    return results


def extract_ids_for_all_clips(
    tracking_results: Dict[str, Dict[str, str]], 
    output_dir: Path,
    skip_ui: bool = False
) -> Dict[str, List[int]]:
    """Extract and select track IDs for all clips.
    
    Args:
        tracking_results: Results from track_all_clips.
        output_dir: Directory containing tracking results.
        skip_ui: Whether to skip UI and auto-select all tracks.
        
    Returns:
        Dictionary mapping clip to selected track IDs.
    """
    print(f"\nExtracting track IDs for {len(tracking_results)} clips")
    
    # Initialize tracker
    tracker = PlayerTracker()
    
    selected_ids = {}
    processed = 0
    
    for clip_key, clip_info in tracking_results.items():
        tracking_file = Path(clip_info["tracking_file"])
        selected_file = Path(clip_info["selected_file"])
        
        if not tracking_file.exists():
            print(f"Warning: Tracking file not found: {tracking_file}")
            continue
        
        print(f"Extracting IDs for {clip_key}")
        
        try:
            # Load tracking results
            with open(tracking_file, "r") as f:
                track_history = json.load(f)
            
            # Extract unique track IDs
            unique_ids = set()
            for frame in track_history:
                for track in frame:
                    unique_ids.add(track["id"])
            
            if skip_ui:
                # Auto-select all tracks
                clip_selected_ids = list(unique_ids)
                print(f"  - Auto-selected {len(clip_selected_ids)} tracks: {sorted(clip_selected_ids)}")
            else:
                # Launch UI for selection
                clip_selected_ids = tracker.select_track_ids(track_history)
                print(f"  - Selected {len(clip_selected_ids)} tracks: {sorted(clip_selected_ids)}")
            
            # Save selected IDs
            selected_file.parent.mkdir(parents=True, exist_ok=True)
            with open(selected_file, "w") as f:
                json.dump(clip_selected_ids, f, indent=2)
            
            selected_ids[clip_key] = clip_selected_ids
            processed += 1
            
        except Exception as e:
            print(f"Error extracting IDs for {clip_key}: {e}")
            continue
    
    print(f"\nID extraction completed for {processed} clips")
    return selected_ids


def save_batch_summary(
    tracking_results: Dict[str, Dict[str, str]], 
    selected_ids: Dict[str, List[int]],
    output_dir: Path
) -> None:
    """Save batch processing summary.
    
    Args:
        tracking_results: Tracking results dictionary.
        selected_ids: Selected IDs dictionary.
        output_dir: Output directory.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "total_clips": len(tracking_results),
        "processed_clips": len(selected_ids),
        "tracking_results": tracking_results,
        "selected_ids": selected_ids,
        "games": list(set(key.split("/")[0] for key in tracking_results.keys()))
    }
    
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Batch summary saved to: {summary_file}")


def main() -> None:
    """Main function for batch player tracking."""
    parser = argparse.ArgumentParser(
        description="Batch track players in all TrackNet dataset clips"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/tracknet",
        help="Path to the TrackNet dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/batch_tracking",
        help="Directory to save batch tracking results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8x.pt",
        help="YOLO model path (default: yolov8x.pt)",
    )
    parser.add_argument(
        "--extract-ids",
        action="store_true",
        help="Extract and select track IDs after tracking",
    )
    parser.add_argument(
        "--skip-ui",
        action="store_true",
        help="Skip player selection UI (auto-select all tracks)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing tracking results",
    )
    
    args = parser.parse_args()
    
    # Convert paths
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    
    if not dataset_root.exists():
        print(f"Error: Dataset root not found: {dataset_root}")
        return
    
    # Step 1: Track all clips
    print("=" * 60)
    print("STEP 1: Batch player tracking")
    print("=" * 60)
    
    tracking_results = track_all_clips(
        dataset_root, 
        output_dir, 
        args.model, 
        args.force
    )
    
    # Step 2: Extract track IDs (if requested)
    selected_ids = {}
    if args.extract_ids:
        print("\n" + "=" * 60)
        print("STEP 2: Track ID extraction")
        print("=" * 60)
        
        selected_ids = extract_ids_for_all_clips(
            tracking_results, 
            output_dir, 
            args.skip_ui
        )
    
    # Step 3: Save summary
    print("\n" + "=" * 60)
    print("STEP 3: Saving batch summary")
    print("=" * 60)
    
    save_batch_summary(tracking_results, selected_ids, output_dir)
    
    # Step 4: Show next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    if args.extract_ids:
        print("All clips processed with ID selection!")
        print("\nTo merge results into COCO format:")
        print("uv run python -m tracknet.tools.coco_converter \\")
        print(f"  --dataset-root {dataset_root} \\")
        print(f"  --output {dataset_root}/annotations.json \\")
        print(f"  --load-existing {dataset_root}/annotations.json")
        
        # Add command for each clip
        for clip_key in sorted(selected_ids.keys()):
            game_name, clip_name = clip_key.split("/")
            tracking_file = tracking_results[clip_key]["tracking_file"]
            selected_file = tracking_results[clip_key]["selected_file"]
            print(f"  --add-players {tracking_file} {clip_name} {selected_file} \\")
        
        print("  # (Add all clips above)")
    else:
        print("Tracking completed! To extract IDs:")
        print("uv run python -m tracknet.tools.batch_player_tracker \\")
        print(f"  --dataset-root {dataset_root} \\")
        print(f"  --output-dir {output_dir} \\")
        print("  --extract-ids [--skip-ui]")
    
    print("\nBatch processing completed successfully!")


if __name__ == "__main__":
    main()
