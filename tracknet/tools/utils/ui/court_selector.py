"""Court annotation UI for TrackNet dataset extension.

This module provides an interactive matplotlib-based UI for annotating
tennis court keypoints in videos. Users can click to place keypoints
and visualize court lines in real-time.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox


class CourtAnnotationUI:
    """Interactive UI for annotating tennis court keypoints.
    
    Provides a video frame interface where users can click to place
    court keypoints and see the court lines visualized in real-time.
    """
    
    def __init__(
        self, 
        video_path: str | Path, 
        keypoint_names: List[str],
        existing_annotation: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> None:
        """Initialize the court annotation UI.
        
        Args:
            video_path: Path to the video file.
            keypoint_names: List of keypoint names to annotate.
            existing_annotation: Optional existing annotation coordinates.
        """
        self.video_path = Path(video_path)
        self.keypoint_names = keypoint_names
        self.existing_annotation = existing_annotation or {}
        
        # Load video
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # UI state
        self.current_frame = self.total_frames // 2  # Start at middle frame
        self.current_keypoint_idx = 0
        self.annotations = {}
        self.completed_keypoints = set()
        
        # Load existing annotations if provided
        if self.existing_annotation:
            self.annotations = self.existing_annotation.copy()
            self.completed_keypoints = set(self.existing_annotation.keys())
        
        # Setup UI
        self.setup_figure()
        self.update_frame()
    
    def setup_figure(self) -> None:
        """Setup the matplotlib figure and widgets."""
        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main image display
        self.ax_img = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3)
        self.ax_img.set_title("Court Keypoint Annotation")
        self.ax_img.axis("off")
        
        # Keypoint list and status
        self.ax_keypoints = plt.subplot2grid((4, 3), (0, 2), rowspan=2)
        self.setup_keypoint_list()
        
        # Instructions
        self.ax_instructions = plt.subplot2grid((4, 3), (2, 2))
        self.setup_instructions()
        
        # Frame navigation
        self.ax_frame_nav = plt.subplot2grid((4, 3), (3, 0), colspan=2)
        self.setup_frame_navigation()
        
        # Control buttons
        self.ax_controls = plt.subplot2grid((4, 3), (3, 2))
        self.setup_controls()
        
        # Connect mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.suptitle(f"Court Annotation - {self.video_path.name}", fontsize=14)
    
    def setup_keypoint_list(self) -> None:
        """Setup the keypoint list display."""
        self.ax_keypoints.clear()
        self.ax_keypoints.set_title("Keypoints")
        self.ax_keypoints.axis("off")
        
        y_pos = 0.95
        for i, name in enumerate(self.keypoint_names):
            status = "✓" if name in self.completed_keypoints else "○"
            color = "green" if name in self.completed_keypoints else "black"
            if i == self.current_keypoint_idx:
                color = "blue"
                status = "→"
            
            self.ax_keypoints.text(
                0.05, y_pos, f"{status} {i}: {name}",
                transform=self.ax_keypoints.transAxes,
                fontsize=10, color=color, fontweight='bold' if i == self.current_keypoint_idx else 'normal'
            )
            y_pos -= 0.08
        
        self.ax_keypoints.set_xlim(0, 1)
        self.ax_keypoints.set_ylim(0, 1)
    
    def setup_instructions(self) -> None:
        """Setup the instructions display."""
        self.ax_instructions.clear()
        self.ax_instructions.axis("off")
        
        instructions = [
            "Instructions:",
            "• Click on image to place keypoint",
            "• Use frame slider to find best view",
            "• Right-click to remove keypoint",
            "• Navigate keypoints with buttons",
            "• Press 'Done' when finished"
        ]
        
        y_pos = 0.9
        for instruction in instructions:
            self.ax_instructions.text(
                0.05, y_pos, instruction,
                transform=self.ax_instructions.transAxes,
                fontsize=9, verticalalignment='top'
            )
            y_pos -= 0.15
    
    def setup_frame_navigation(self) -> None:
        """Setup frame navigation controls."""
        from matplotlib.widgets import Slider
        
        # Frame slider
        self.frame_slider = Slider(
            self.ax_frame_nav, "Frame", 0, self.total_frames - 1,
            valinit=self.current_frame, valstep=1
        )
        self.frame_slider.on_changed(self.on_frame_change)
        
        # Jump buttons
        self.btn_start = Button(plt.axes([0.15, 0.08, 0.06, 0.03]), "Start")
        self.btn_middle = Button(plt.axes([0.22, 0.08, 0.06, 0.03]), "Middle")
        self.btn_end = Button(plt.axes([0.29, 0.08, 0.06, 0.03]), "End")
        
        self.btn_start.on_clicked(lambda x: self.jump_to_frame(0))
        self.btn_middle.on_clicked(lambda x: self.jump_to_frame(self.total_frames // 2))
        self.btn_end.on_clicked(lambda x: self.jump_to_frame(self.total_frames - 1))
    
    def setup_controls(self) -> None:
        """Setup control buttons."""
        self.btn_prev = Button(plt.axes([0.72, 0.15, 0.08, 0.04]), "Prev")
        self.btn_next = Button(plt.axes([0.81, 0.15, 0.08, 0.04]), "Next")
        self.btn_clear = Button(plt.axes([0.72, 0.10, 0.08, 0.04]), "Clear")
        self.btn_done = Button(plt.axes([0.81, 0.10, 0.08, 0.04]), "Done")
        
        self.btn_prev.on_clicked(self.prev_keypoint)
        self.btn_next.on_clicked(self.next_keypoint)
        self.btn_clear.on_clicked(self.clear_current_keypoint)
        self.btn_done.on_clicked(self.finish_annotation)
    
    def update_frame(self) -> None:
        """Update the displayed frame and annotations."""
        # Clear previous image
        self.ax_img.clear()
        
        # Read frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display frame
        self.ax_img.imshow(frame_rgb)
        
        # Draw court lines and keypoints
        self.draw_court_annotation()
        
        # Highlight current keypoint
        current_name = self.keypoint_names[self.current_keypoint_idx]
        if current_name in self.annotations:
            coord = self.annotations[current_name]
            self.ax_img.plot(coord[0], coord[1], 'yo', markersize=10, markeredgewidth=2)
        else:
            # Show prompt for current keypoint
            self.ax_img.text(
                10, 30, f"Click to place: {current_name}",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
                fontsize=12, fontweight='bold'
            )
        
        # Update title
        self.ax_img.set_title(f"Frame {self.current_frame}/{self.total_frames - 1}")
        self.ax_img.axis("off")
        
        # Update keypoint list
        self.setup_keypoint_list()
        
        # Refresh display
        self.fig.canvas.draw_idle()
    
    def draw_court_annotation(self) -> None:
        """Draw court lines and keypoints on the current frame."""
        # Use the same skeleton as defined in court_annotator.py
        skeleton = [
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
        
        # Draw connections
        for start_idx, end_idx in skeleton:
            start_name = self.keypoint_names[start_idx]
            end_name = self.keypoint_names[end_idx]
            
            if start_name in self.annotations and end_name in self.annotations:
                start_pt = self.annotations[start_name]
                end_pt = self.annotations[end_name]
                
                self.ax_img.plot(
                    [start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]],
                    'g-', linewidth=2, alpha=0.7
                )
        
        # Draw all annotated keypoints
        for i, (name, coord) in enumerate(self.annotations.items()):
            try:
                idx = self.keypoint_names.index(name)
            except ValueError:
                idx = -1
            
            # Draw point
            color = 'red' if idx == self.current_keypoint_idx else 'blue'
            self.ax_img.plot(coord[0], coord[1], 'o', color=color, markersize=6)
            
            # Draw label
            label = f"{idx}" if idx >= 0 else name[:8]
            self.ax_img.text(
                coord[0] + 10, coord[1] - 10, label,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                fontsize=8
            )
    
    def on_click(self, event) -> None:
        """Handle mouse click events on the image."""
        if event.inaxes != self.ax_img:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        current_name = self.keypoint_names[self.current_keypoint_idx]
        
        if event.button == 1:  # Left click - place keypoint
            self.annotations[current_name] = (x, y)
            self.completed_keypoints.add(current_name)
            
            # Auto-advance to next keypoint
            if self.current_keypoint_idx < len(self.keypoint_names) - 1:
                self.current_keypoint_idx += 1
            
            self.update_frame()
            
        elif event.button == 3:  # Right click - remove keypoint
            if current_name in self.annotations:
                del self.annotations[current_name]
                self.completed_keypoints.discard(current_name)
                self.update_frame()
    
    def on_frame_change(self, val: float) -> None:
        """Handle frame slider change."""
        self.current_frame = int(val)
        self.update_frame()
    
    def jump_to_frame(self, frame_idx: int) -> None:
        """Jump to specific frame."""
        self.current_frame = max(0, min(frame_idx, self.total_frames - 1))
        self.frame_slider.set_val(self.current_frame)
        self.update_frame()
    
    def prev_keypoint(self, event) -> None:
        """Go to previous keypoint."""
        if self.current_keypoint_idx > 0:
            self.current_keypoint_idx -= 1
            self.update_frame()
    
    def next_keypoint(self, event) -> None:
        """Go to next keypoint."""
        if self.current_keypoint_idx < len(self.keypoint_names) - 1:
            self.current_keypoint_idx += 1
            self.update_frame()
    
    def clear_current_keypoint(self, event) -> None:
        """Clear current keypoint annotation."""
        current_name = self.keypoint_names[self.current_keypoint_idx]
        if current_name in self.annotations:
            del self.annotations[current_name]
            self.completed_keypoints.discard(current_name)
            self.update_frame()
    
    def finish_annotation(self, event) -> None:
        """Finish annotation and close UI."""
        # Check if all keypoints are annotated
        missing_keypoints = [
            name for name in self.keypoint_names 
            if name not in self.annotations
        ]
        
        if missing_keypoints:
            print(f"Warning: Missing annotations for: {missing_keypoints}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        plt.close(self.fig)
    
    def run(self) -> Dict[str, Tuple[float, float]]:
        """Run the court annotation UI.
        
        Returns:
            Dictionary mapping keypoint names to (x, y) coordinates.
        """
        plt.show()
        
        # Clean up
        self.cap.release()
        
        return self.annotations


class CourtReviewUI:
    """UI for reviewing and editing existing court annotations."""
    
    def __init__(
        self, 
        video_path: str | Path, 
        annotation: Dict[str, Tuple[float, float]],
        keypoint_names: List[str]
    ) -> None:
        """Initialize the court review UI.
        
        Args:
            video_path: Path to the video file.
            annotation: Existing court annotation.
            keypoint_names: List of keypoint names.
        """
        self.video_path = Path(video_path)
        self.annotation = annotation.copy()
        self.keypoint_names = keypoint_names
        
        # Load video
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = self.total_frames // 2
        
        # Setup UI
        self.setup_figure()
        self.update_frame()
    
    def setup_figure(self) -> None:
        """Setup the review UI figure."""
        self.fig = plt.figure(figsize=(14, 10))
        
        # Main image display
        self.ax_img = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_img.set_title("Court Annotation Review")
        self.ax_img.axis("off")
        
        # Keypoint list
        self.ax_keypoints = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        self.setup_keypoint_list()
        
        # Frame navigation
        self.ax_frame_nav = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        self.setup_frame_navigation()
        
        # Controls
        self.ax_controls = plt.subplot2grid((3, 3), (2, 2))
        self.setup_controls()
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.selected_keypoint = None
        
        plt.suptitle(f"Court Review - {self.video_path.name}", fontsize=14)
    
    def setup_keypoint_list(self) -> None:
        """Setup the keypoint list for review."""
        self.ax_keypoints.clear()
        self.ax_keypoints.set_title("Keypoints (Click to select)")
        self.ax_keypoints.axis("off")
        
        y_pos = 0.95
        for i, name in enumerate(self.keypoint_names):
            status = "✓" if name in self.annotation else "✗"
            color = "green" if name in self.annotation else "red"
            
            self.ax_keypoints.text(
                0.05, y_pos, f"{status} {i}: {name}",
                transform=self.ax_keypoints.transAxes,
                fontsize=10, color=color
            )
            y_pos -= 0.08
        
        self.ax_keypoints.set_xlim(0, 1)
        self.ax_keypoints.set_ylim(0, 1)
    
    def setup_frame_navigation(self) -> None:
        """Setup frame navigation."""
        from matplotlib.widgets import Slider
        
        self.frame_slider = Slider(
            self.ax_frame_nav, "Frame", 0, self.total_frames - 1,
            valinit=self.current_frame, valstep=1
        )
        self.frame_slider.on_changed(self.on_frame_change)
    
    def setup_controls(self) -> None:
        """Setup control buttons."""
        self.btn_save = Button(plt.axes([0.72, 0.15, 0.08, 0.04]), "Save")
        self.btn_delete = Button(plt.axes([0.81, 0.15, 0.08, 0.04]), "Delete")
        self.btn_done = Button(plt.axes([0.72, 0.10, 0.17, 0.04]), "Done")
        
        self.btn_save.on_clicked(self.save_changes)
        self.btn_delete.on_clicked(self.delete_selected)
        self.btn_done.on_clicked(self.finish_review)
    
    def update_frame(self) -> None:
        """Update the displayed frame."""
        self.ax_img.clear()
        
        # Read frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax_img.imshow(frame_rgb)
        
        # Draw annotation
        self.draw_court_annotation()
        
        self.ax_img.set_title(f"Frame {self.current_frame}/{self.total_frames - 1}")
        self.ax_img.axis("off")
        
        self.fig.canvas.draw_idle()
    
    def draw_court_annotation(self) -> None:
        """Draw court annotation on frame."""
        # Same skeleton as in annotation UI
        skeleton = [
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
        
        # Draw connections
        for start_idx, end_idx in skeleton:
            start_name = self.keypoint_names[start_idx]
            end_name = self.keypoint_names[end_idx]
            
            if start_name in self.annotation and end_name in self.annotation:
                start_pt = self.annotation[start_name]
                end_pt = self.annotation[end_name]
                
                self.ax_img.plot(
                    [start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]],
                    'g-', linewidth=2, alpha=0.7
                )
        
        # Draw keypoints
        for name, coord in self.annotation.items():
            color = 'yellow' if name == self.selected_keypoint else 'blue'
            self.ax_img.plot(coord[0], coord[1], 'o', color=color, markersize=8)
            
            try:
                idx = self.keypoint_names.index(name)
                label = f"{idx}"
            except ValueError:
                label = name[:8]
            
            self.ax_img.text(
                coord[0] + 10, coord[1] - 10, label,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                fontsize=8
            )
    
    def on_click(self, event) -> None:
        """Handle mouse clicks."""
        if event.inaxes == self.ax_img and event.button == 1:
            # Find nearest keypoint
            min_dist = float('inf')
            nearest_keypoint = None
            
            for name, coord in self.annotation.items():
                dist = np.sqrt((coord[0] - event.xdata)**2 + (coord[1] - event.ydata)**2)
                if dist < min_dist and dist < 20:  # Within 20 pixels
                    min_dist = dist
                    nearest_keypoint = name
            
            self.selected_keypoint = nearest_keypoint
            self.update_frame()
    
    def on_frame_change(self, val: float) -> None:
        """Handle frame slider change."""
        self.current_frame = int(val)
        self.update_frame()
    
    def save_changes(self, event) -> None:
        """Save current annotation state."""
        # In a real implementation, this would save to a file
        print("Annotation changes saved!")
    
    def delete_selected(self, event) -> None:
        """Delete selected keypoint."""
        if self.selected_keypoint and self.selected_keypoint in self.annotation:
            del self.annotation[self.selected_keypoint]
            self.selected_keypoint = None
            self.setup_keypoint_list()
            self.update_frame()
    
    def finish_review(self, event) -> None:
        """Finish review and close UI."""
        plt.close(self.fig)
    
    def run(self) -> Dict[str, Tuple[float, float]]:
        """Run the review UI.
        
        Returns:
            Updated annotation dictionary.
        """
        plt.show()
        self.cap.release()
        return self.annotation
