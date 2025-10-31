"""COCO format converter for TrackNet dataset.

This tool converts existing CSV annotations to COCO JSON format and provides
functions to add player tracking and court annotations to the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from tracknet.tools.court_annotator import CourtKeypointAnnotator


class COCOConverter:
    """Convert TrackNet CSV annotations to COCO JSON format.
    
    This class handles the conversion from the existing CSV format to
    COCO format, and provides methods to add player and court annotations.
    """
    
    # COCO category definitions
    CATEGORIES = [
        {"id": 1, "name": "ball", "keypoints": 0, "skeleton": []},
        {"id": 2, "name": "player", "keypoints": 0, "skeleton": []},
        {
            "id": 3, 
            "name": "court", 
            "keypoints": 15,
            "skeleton": [
                [1, 2],   # far doubles line
                [3, 4],   # near doubles line  
                [1, 3],   # left doubles sideline
                [2, 4],   # right doubles sideline
                [5, 6],   # left singles sideline
                [7, 8],   # right singles sideline
                [9, 10],  # far service line
                [11, 12], # near service line
                [13, 14], # service T to net center
            ]
        }
    ]
    
    # Court keypoint names (matching court_annotator.py)
    COURT_KEYPOINTS = [
        "far doubles corner left",
        "far doubles corner right", 
        "near doubles corner left",
        "near doubles corner right",
        "far singles corner left",
        "near singles corner left",
        "far singles corner right",
        "near singles corner right",
        "far service-line endpoint left",
        "far service-line endpoint right",
        "near service-line endpoint left",
        "near service-line endpoint right",
        "far service T",
        "near service T",
        "net center",
    ]
    
    def __init__(self, dataset_root: str | Path) -> None:
        """Initialize the COCO converter.
        
        Args:
            dataset_root: Root directory of the TrackNet dataset.
        """
        self.dataset_root = Path(dataset_root)
        self.coco_data = {
            "info": {
                "description": "TrackNet Tennis Dataset",
                "version": "2.0",
                "year": 2024,
                "contributor": "TrackNet Project",
                "date_created": "2024-01-01"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown License",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": self.CATEGORIES
        }
        self.image_id = 1
        self.annotation_id = 1
    
    def convert_csv_to_coco(self, output_path: str | Path) -> None:
        """Convert existing CSV annotations to COCO format.
        
        Args:
            output_path: Path to save the COCO JSON file.
        """
        output_path = Path(output_path)
        print(f"Converting CSV annotations to COCO format...")
        
        # Find all games and clips
        game_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir() and d.name.startswith("game")]
        
        for game_dir in sorted(game_dirs):
            print(f"Processing game: {game_dir.name}")
            self._process_game(game_dir)
        
        # Save COCO data
        with open(output_path, "w") as f:
            json.dump(self.coco_data, f, indent=2)
        
        print(f"COCO format saved to: {output_path}")
        print(f"Images: {len(self.coco_data['images'])}")
        print(f"Annotations: {len(self.coco_data['annotations'])}")
    
    def _process_game(self, game_dir: Path) -> None:
        """Process a single game directory.
        
        Args:
            game_dir: Path to the game directory.
        """
        clip_dirs = [d for d in game_dir.iterdir() if d.is_dir() and d.name.startswith("Clip")]
        
        for clip_dir in sorted(clip_dirs):
            self._process_clip(clip_dir)
    
    def _process_clip(self, clip_dir: Path) -> None:
        """Process a single clip directory.
        
        Args:
            clip_dir: Path to the clip directory.
        """
        csv_file = clip_dir / "Label.csv"
        if not csv_file.exists():
            print(f"Warning: CSV file not found in {clip_dir}")
            return
        
        # Read CSV annotations
        try:
            df = pd.read_csv(csv_file)
            print(f"Debug: Loaded CSV from {csv_file}, shape: {df.shape}, columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error reading CSV file {csv_file}: {e}")
            return
        
        # Process each row (frame)
        for idx, row in df.iterrows():
            if idx % 100 == 0:  # Debug print every 100 rows
                print(f"Debug: Processing row {idx} in {clip_dir.name}")
            self._process_frame(clip_dir, row)
    
    def _process_frame(self, clip_dir: Path, row: pd.Series) -> None:
        """Process a single frame annotation.
        
        Args:
            clip_dir: Path to the clip directory.
            row: DataFrame row containing frame annotation.
        """
        # Extract frame information
        frame_name = row.get('file name', '')  # Use actual column name
        if not frame_name or pd.isna(frame_name):
            # Try to find frame column name
            frame_cols = [col for col in row.index if 'frame' in col.lower() or 'file' in col.lower() or col.isdigit()]
            if frame_cols:
                frame_name = str(row.get(frame_cols[0], ''))
            else:
                if len(row) > 0:
                    frame_name = str(row.index[0])  # Use first column as fallback
                else:
                    return
        
        # Debug print first few frames
        if self.image_id <= 5:
            print(f"Debug: Frame {self.image_id} - frame_name: '{frame_name}', row keys: {list(row.keys())}")
        
        # Create image entry
        image_path = clip_dir / frame_name
        if not image_path.exists():
            # Try common image extensions
            for ext in ['.jpg', '.jpeg', '.png']:
                test_path = clip_dir / (frame_name + ext)
                if test_path.exists():
                    image_path = test_path
                    break
            else:
                if self.image_id <= 5:
                    print(f"Debug: Image file not found for frame '{frame_name}' in {clip_dir}")
                return
        
        # Get image dimensions
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            if self.image_id <= 5:
                print(f"Error reading image {image_path}: {e}")
            return
        
        # Add image to COCO data
        image_entry = {
            "id": self.image_id,
            "width": width,
            "height": height,
            "file_name": str(image_path.relative_to(self.dataset_root)),
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }
        self.coco_data["images"].append(image_entry)
        
        # Add ball annotation if available
        ball_x = row.get('x-coordinate', None)  # Use actual column name
        ball_y = row.get('y-coordinate', None)  # Use actual column name
        visibility = row.get('visibility', 0)
        
        # Try alternative column names
        if ball_x is None:
            for col in row.index:
                if 'x' in col.lower() and 'coordinate' in col.lower():
                    ball_x = row.get(col)
                    break
        if ball_y is None:
            for col in row.index:
                if 'y' in col.lower() and 'coordinate' in col.lower():
                    ball_y = row.get(col)
                    break
        
        if ball_x is not None and ball_y is not None and visibility > 0:
            ball_annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": 1,  # ball category
                "segmentation": [],
                "area": 0,  # Point annotation has no area
                "bbox": [ball_x, ball_y, 1, 1],  # Small bbox for point
                "iscrowd": 0
            }
            self.coco_data["annotations"].append(ball_annotation)
            self.annotation_id += 1
        
        self.image_id += 1
    
    def add_player_tracking(
        self, 
        tracking_file: str | Path, 
        clip_dir: str | Path,
        selected_ids: List[int]
    ) -> None:
        """Add player tracking annotations to COCO data.
        
        Args:
            tracking_file: Path to player tracking JSON file.
            clip_dir: Path to the clip directory.
            selected_ids: List of selected player track IDs.
        """
        tracking_file = Path(tracking_file)
        clip_dir = Path(clip_dir)
        
        # Load tracking data
        with open(tracking_file, "r") as f:
            tracking_data = json.load(f)
        
        # Find corresponding images in COCO data
        clip_images = [
            img for img in self.coco_data["images"]
            if str(clip_dir.name) in img["file_name"]
        ]
        
        # Add player annotations for selected tracks
        for frame_idx, frame_tracking in enumerate(tracking_data):
            if frame_idx >= len(clip_images):
                break
            
            image_id = clip_images[frame_idx]["id"]
            
            for track in frame_tracking:
                track_id = track["id"]
                if track_id in selected_ids:
                    bbox = track["bbx_xyxy"]
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    player_annotation = {
                        "id": self.annotation_id,
                        "image_id": image_id,
                        "category_id": 2,  # player category
                        "segmentation": [],
                        "area": area,
                        "bbox": [x1, y1, width, height],
                        "iscrowd": 0,
                        "track_id": track_id
                    }
                    self.coco_data["annotations"].append(player_annotation)
                    self.annotation_id += 1
        
        print(f"Added player annotations for {len(selected_ids)} tracks in {clip_dir.name}")
    
    def add_court_annotation(
        self, 
        court_file: str | Path, 
        game_dir: str | Path
    ) -> None:
        """Add court keypoint annotations to COCO data.
        
        Args:
            court_file: Path to court annotation JSON file.
            game_dir: Path to the game directory.
        """
        court_file = Path(court_file)
        game_dir = Path(game_dir)
        
        # Load court annotation
        with open(court_file, "r") as f:
            court_data = json.load(f)
        
        # Convert lists to tuples if needed
        court_annotation = {k: tuple(v) for k, v in court_data.items()}
        
        # Find all images in this game
        game_images = [
            img for img in self.coco_data["images"]
            if str(game_dir.name) in img["file_name"]
        ]
        
        # Add court annotation to all frames in the game
        for image in game_images:
            image_id = image["id"]
            
            # Create keypoints array (x, y, visibility)
            keypoints = []
            for keypoint_name in self.COURT_KEYPOINTS:
                if keypoint_name in court_annotation:
                    x, y = court_annotation[keypoint_name]
                    keypoints.extend([x, y, 2])  # 2 = visible
                else:
                    keypoints.extend([0, 0, 0])  # 0 = not annotated
            
            # Calculate bbox for court keypoints
            valid_points = [(kp[0], kp[1]) for kp in [keypoints[i:i+3] for i in range(0, len(keypoints), 3)] if kp[2] > 0]
            if valid_points:
                x_coords = [p[0] for p in valid_points]
                y_coords = [p[1] for p in valid_points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = bbox[2] * bbox[3]
            else:
                bbox = [0, 0, 0, 0]
                area = 0
            
            court_annotation_entry = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": 3,  # court category
                "segmentation": [],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": len([kp for kp in keypoints[2::3] if kp > 0])
            }
            self.coco_data["annotations"].append(court_annotation_entry)
            self.annotation_id += 1
        
        print(f"Added court annotations for {len(game_images)} frames in {game_dir.name}")
    
    def save_coco_data(self, output_path: str | Path) -> None:
        """Save COCO data to file.
        
        Args:
            output_path: Path to save the COCO JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.coco_data, f, indent=2)
        
        print(f"COCO data saved to: {output_path}")
    
    def load_existing_coco(self, coco_file: str | Path) -> None:
        """Load existing COCO format file.
        
        Args:
            coco_file: Path to existing COCO JSON file.
        """
        coco_file = Path(coco_file)
        with open(coco_file, "r") as f:
            self.coco_data = json.load(f)
        
        # Update IDs
        if self.coco_data["images"]:
            self.image_id = max(img["id"] for img in self.coco_data["images"]) + 1
        if self.coco_data["annotations"]:
            self.annotation_id = max(ann["id"] for ann in self.coco_data["annotations"]) + 1
        
        print(f"Loaded existing COCO data: {len(self.coco_data['images'])} images, {len(self.coco_data['annotations'])} annotations")


def main() -> None:
    """Main function for COCO conversion."""
    parser = argparse.ArgumentParser(
        description="Convert TrackNet CSV annotations to COCO JSON format"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory of the TrackNet dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the COCO JSON file",
    )
    parser.add_argument(
        "--add-players",
        type=str,
        nargs="+",
        help="Add player tracking: <tracking_file> <clip_dir> <selected_ids_json>",
    )
    parser.add_argument(
        "--add-court",
        type=str,
        nargs=2,
        metavar=("COURT_FILE", "GAME_DIR"),
        help="Add court annotation: <court_file> <game_dir>",
    )
    parser.add_argument(
        "--load-existing",
        type=str,
        help="Load existing COCO file before adding new annotations",
    )
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = COCOConverter(args.dataset_root)
    
    # Load existing COCO data if specified
    if args.load_existing:
        converter.load_existing_coco(args.load_existing)
    else:
        # Convert CSV to COCO
        converter.convert_csv_to_coco(args.output)
    
    # Add player tracking if specified
    if args.add_players and len(args.add_players) >= 3:
        tracking_file = args.add_players[0]
        clip_dir = args.add_players[1]
        selected_ids_file = args.add_players[2]
        
        # Load selected IDs
        with open(selected_ids_file, "r") as f:
            selected_ids = json.load(f)
        
        converter.add_player_tracking(tracking_file, clip_dir, selected_ids)
    
    # Add court annotation if specified
    if args.add_court:
        court_file, game_dir = args.add_court
        converter.add_court_annotation(court_file, game_dir)
    
    # Save final COCO data
    converter.save_coco_data(args.output)
    
    print("COCO conversion completed successfully!")


if __name__ == "__main__":
    main()
