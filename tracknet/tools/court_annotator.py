"""Court keypoint annotation tool for TrackNet dataset extension.

This tool provides functionality to annotate tennis court keypoints in videos.
Since the camera position is fixed per game, annotations are done at the game level
and can be applied to all clips within that game.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from tracknet.tools.utils.ui.court_selector import CourtAnnotationUI


class CourtKeypointAnnotator:
    """Court keypoint annotator for tennis videos.
    
    This class handles the annotation of tennis court keypoints such as
    lines, corners, and other important court features. Since camera position
    is fixed per game, annotations are collected once per game.
    """
    
    # Standard tennis court keypoints (15 points for complete court lines)
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
    
    # Court skeleton connections (keypoint index pairs)
    COURT_SKELETON = [
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
    
    def __init__(self) -> None:
        """Initialize the court keypoint annotator."""
        self.keypoints = self.COURT_KEYPOINTS
        self.keypoint_mapping = {i: name for i, name in enumerate(self.keypoints)}
        
    def annotate_game(
        self, 
        video_path: str | Path,
        existing_annotation: Optional[Dict] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Annotate court keypoints for a game video.
        
        Args:
            video_path: Path to the game video.
            existing_annotation: Optional existing annotation to edit.
            
        Returns:
            Dictionary mapping keypoint names to (x, y) coordinates.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Launch annotation UI
        ui = CourtAnnotationUI(video_path, self.keypoints, existing_annotation)
        annotation = ui.run()
        
        return annotation
    
    def save_annotation(
        self, 
        annotation: Dict[str, Tuple[float, float]], 
        output_path: str | Path
    ) -> None:
        """Save court annotation to file.
        
        Args:
            annotation: Court keypoint annotation.
            output_path: Path to save the annotation.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(annotation, f, indent=2)
        
        print(f"Court annotation saved to: {output_path}")
    
    def load_annotation(self, annotation_path: str | Path) -> Dict[str, Tuple[float, float]]:
        """Load court annotation from file.
        
        Args:
            annotation_path: Path to the annotation file.
            
        Returns:
            Dictionary mapping keypoint names to (x, y) coordinates.
        """
        annotation_path = Path(annotation_path)
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        with open(annotation_path, "r") as f:
            annotation = json.load(f)
        
        # Convert lists back to tuples
        annotation = {k: tuple(v) for k, v in annotation.items()}
        
        return annotation
    
    def validate_annotation(self, annotation: Dict[str, Tuple[float, float]]) -> bool:
        """Validate court annotation completeness.
        
        Args:
            annotation: Court keypoint annotation.
            
        Returns:
            True if annotation is valid, False otherwise.
        """
        # Check all required keypoints are present
        for keypoint in self.keypoints:
            if keypoint not in annotation:
                print(f"Missing required keypoint: {keypoint}")
                return False
            
            # Check coordinate format
            coord = annotation[keypoint]
            if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                print(f"Invalid coordinate format for {keypoint}: {coord}")
                return False
            
            # Check coordinate values are positive
            if coord[0] < 0 or coord[1] < 0:
                print(f"Invalid coordinate values for {keypoint}: {coord}")
                return False
        
        return True
    
    def visualize_annotation(
        self, 
        video_path: str | Path, 
        annotation: Dict[str, Tuple[float, float]],
        output_path: Optional[str | Path] = None
    ) -> None:
        """Visualize court annotation on video frame.
        
        Args:
            video_path: Path to the video file.
            annotation: Court keypoint annotation.
            output_path: Optional path to save visualization.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        # Get middle frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("Failed to read video frame")
        
        # Draw keypoints and connections
        frame_with_annotation = self.draw_annotation_on_frame(frame, annotation)
        
        if output_path:
            cv2.imwrite(str(output_path), frame_with_annotation)
            print(f"Annotation visualization saved to: {output_path}")
        else:
            cv2.imshow("Court Annotation", frame_with_annotation)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def draw_annotation_on_frame(
        self, 
        frame: np.ndarray, 
        annotation: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """Draw court annotation on video frame.
        
        Args:
            frame: Video frame as numpy array.
            annotation: Court keypoint annotation.
            
        Returns:
            Frame with annotation drawn on it.
        """
        result = frame.copy()
        
        # Draw skeleton connections
        for start_idx, end_idx in self.COURT_SKELETON:
            start_name = self.keypoints[start_idx]
            end_name = self.keypoints[end_idx]
            
            if start_name in annotation and end_name in annotation:
                start_pt = tuple(map(int, annotation[start_name]))
                end_pt = tuple(map(int, annotation[end_name]))
                
                cv2.line(result, start_pt, end_pt, (0, 255, 0), 2)
        
        # Draw keypoints
        for i, (name, coord) in enumerate(annotation.items()):
            center = tuple(map(int, coord))
            
            # Draw point
            cv2.circle(result, center, 5, (0, 0, 255), -1)
            
            # Draw label
            cv2.putText(
                result, f"{i}:{name[:15]}", 
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
        
        return result
    
    def apply_to_game_clips(
        self, 
        game_annotation: Dict[str, Tuple[float, float]],
        game_dir: str | Path,
        output_dir: str | Path
    ) -> None:
        """Apply game-level annotation to all clips in the game.
        
        Args:
            game_annotation: Court annotation for the game.
            game_dir: Directory containing game clips.
            output_dir: Directory to save clip-specific annotations.
        """
        game_dir = Path(game_dir)
        output_dir = Path(output_dir)
        
        # Find all clip directories
        clip_dirs = [d for d in game_dir.iterdir() if d.is_dir() and d.name.startswith("Clip")]
        
        for clip_dir in sorted(clip_dirs):
            clip_name = clip_dir.name
            
            # Create output annotation file
            output_file = output_dir / f"{clip_name}_court.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the same annotation for each clip
            self.save_annotation(game_annotation, output_file)
            print(f"Applied court annotation to {clip_name}")


def main() -> None:
    """Main function for court annotation."""
    parser = argparse.ArgumentParser(
        description="Annotate tennis court keypoints for TrackNet dataset"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the game video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the court annotation (JSON format)",
    )
    parser.add_argument(
        "--edit",
        type=str,
        help="Path to existing annotation file to edit",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show annotation visualization after saving",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate annotation before saving",
    )
    parser.add_argument(
        "--apply-to-clips",
        type=str,
        help="Apply annotation to all clips in the specified game directory",
    )
    
    args = parser.parse_args()
    
    # Initialize annotator
    annotator = CourtKeypointAnnotator()
    
    # Load existing annotation if editing
    existing_annotation = None
    if args.edit:
        print(f"Loading existing annotation: {args.edit}")
        existing_annotation = annotator.load_annotation(args.edit)
    
    # Annotate the video
    print(f"Starting court annotation for: {args.video}")
    annotation = annotator.annotate_game(args.video, existing_annotation)
    
    # Validate if requested
    if args.validate:
        print("Validating annotation...")
        if not annotator.validate_annotation(annotation):
            print("Annotation validation failed!")
            return
        print("Annotation validation passed!")
    
    # Save annotation
    annotator.save_annotation(annotation, args.output)
    
    # Apply to clips if requested
    if args.apply_to_clips:
        print(f"Applying annotation to clips in: {args.apply_to_clips}")
        output_dir = Path(args.output).parent / "clip_annotations"
        annotator.apply_to_game_clips(annotation, args.apply_to_clips, output_dir)
    
    # Visualize if requested
    if args.visualize:
        print("Showing annotation visualization...")
        annotator.visualize_annotation(args.video, annotation)
    
    print("Court annotation completed successfully!")


if __name__ == "__main__":
    main()
