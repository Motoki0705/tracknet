"""Player selection UI for TrackNet dataset annotation.

This module provides an interactive matplotlib-based UI for selecting
player track IDs from YOLO tracking results. Users can preview tracking
results across clips and select appropriate player IDs.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, CheckButtons


class PlayerSelectorUI:
    """Interactive UI for selecting player track IDs.
    
    Provides a video player interface with tracking overlay, allowing users
    to preview tracks and select which IDs correspond to actual players.
    """
    
    def __init__(self, video_path: str | Path, track_history: List[List[Dict]]) -> None:
        """Initialize the player selector UI.
        
        Args:
            video_path: Path to the video file.
            track_history: Tracking history from YOLO tracker.
        """
        self.video_path = Path(video_path)
        self.track_history = track_history
        
        # Load video
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # UI state
        self.current_frame = 0
        self.selected_ids = set()
        self.track_colors = {}
        
        # Generate colors for each track ID
        unique_ids = set()
        for frame in track_history:
            for track in frame:
                unique_ids.add(track["id"])
        
        colormap = plt.cm.get_cmap("tab20")
        for i, track_id in enumerate(sorted(unique_ids)):
            self.track_colors[track_id] = colormap(i % 20)
        
        # Setup UI
        self.setup_figure()
        self.update_frame()
    
    def setup_figure(self) -> None:
        """Setup the matplotlib figure and widgets."""
        # Create figure with space for controls
        self.fig = plt.figure(figsize=(14, 10))
        
        # Main image display
        self.ax_img = plt.subplot2grid((5, 3), (0, 0), colspan=2, rowspan=4)
        self.ax_img.set_title("Player Tracking Preview")
        self.ax_img.axis("off")
        
        # Track ID checkboxes
        self.ax_checkboxes = plt.subplot2grid((5, 3), (0, 2), rowspan=3)
        unique_ids = sorted(self.track_colors.keys())
        self.checkboxes = CheckButtons(
            self.ax_checkboxes, 
            [f"ID {id}" for id in unique_ids],
            [False] * len(unique_ids)
        )
        self.checkboxes.on_clicked(self.on_checkbox_click)
        self.track_id_list = unique_ids
        
        # Frame slider
        self.ax_slider = plt.subplot2grid((5, 3), (4, 0), colspan=2)
        self.frame_slider = Slider(
            self.ax_slider, "Frame", 0, self.total_frames - 1, 
            valinit=0, valstep=1
        )
        self.frame_slider.on_changed(self.on_slider_change)
        
        # Control buttons
        self.ax_buttons = plt.subplot2grid((5, 3), (3, 2))
        self.btn_play = Button(plt.axes([0.72, 0.15, 0.08, 0.04]), "Play/Pause")
        self.btn_done = Button(plt.axes([0.81, 0.15, 0.08, 0.04]), "Done")
        
        self.btn_play.on_clicked(self.toggle_playback)
        self.btn_done.on_clicked(self.finish_selection)
        
        # Playback state
        self.is_playing = False
        self.playback_timer = None
        
        plt.suptitle(f"Player Selection - {self.video_path.name}", fontsize=14)
    
    def update_frame(self) -> None:
        """Update the displayed frame and tracking overlay."""
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
        
        # Draw tracking boxes
        if self.current_frame < len(self.track_history):
            frame_tracks = self.track_history[self.current_frame]
            
            for track in frame_tracks:
                track_id = track["id"]
                bbox = track["bbx_xyxy"]
                
                # Get color (highlight if selected)
                color = self.track_colors[track_id]
                linewidth = 3 if track_id in self.selected_ids else 1
                alpha = 0.8 if track_id in self.selected_ids else 0.6
                
                # Draw bounding box
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    linewidth=linewidth, edgecolor=color, facecolor='none', alpha=alpha
                )
                self.ax_img.add_patch(rect)
                
                # Draw track ID label
                self.ax_img.text(
                    bbox[0], bbox[1] - 10, f"ID {track_id}",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=alpha),
                    color='white', fontsize=8, fontweight='bold'
                )
        
        # Update title with frame info
        self.ax_img.set_title(f"Frame {self.current_frame}/{self.total_frames - 1}")
        self.ax_img.axis("off")
        
        # Refresh display
        self.fig.canvas.draw_idle()
    
    def on_slider_change(self, val: float) -> None:
        """Handle frame slider change."""
        self.current_frame = int(val)
        self.update_frame()
    
    def on_checkbox_click(self, label: str) -> None:
        """Handle track ID checkbox click."""
        # Extract track ID from label
        track_id = int(label.split()[1])
        
        if track_id in self.selected_ids:
            self.selected_ids.remove(track_id)
        else:
            self.selected_ids.add(track_id)
        
        self.update_frame()
    
    def toggle_playback(self, event) -> None:
        """Toggle video playback."""
        if self.is_playing:
            self.is_playing = False
            if self.playback_timer:
                self.fig.canvas.stop_event_loop()
        else:
            self.is_playing = True
            self.play_video()
    
    def play_video(self) -> None:
        """Play video continuously."""
        if self.is_playing and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.set_val(self.current_frame)
            self.update_frame()
            self.fig.canvas.start_event_loop(0.03)  # ~30 FPS
            self.play_video()
        else:
            self.is_playing = False
    
    def finish_selection(self, event) -> None:
        """Finish player selection and close UI."""
        self.is_playing = False
        plt.close(self.fig)
    
    def run(self) -> List[int]:
        """Run the UI and return selected player IDs.
        
        Returns:
            List of selected player track IDs.
        """
        plt.show()
        
        # Clean up
        self.cap.release()
        
        return sorted(list(self.selected_ids))


class ClipPlayerUI:
    """UI for reviewing tracking results by clips.
    
    This provides a clip-based view where users can see tracking continuity
    across longer segments and identify ID jumps or inconsistencies.
    """
    
    def __init__(
        self, 
        video_path: str | Path, 
        track_history: List[List[Dict]],
        clip_size: int = 100
    ) -> None:
        """Initialize the clip player UI.
        
        Args:
            video_path: Path to the video file.
            track_history: Tracking history from YOLO tracker.
            clip_size: Number of frames per clip.
        """
        self.video_path = Path(video_path)
        self.track_history = track_history
        self.clip_size = clip_size
        
        # Calculate clips
        self.total_frames = len(track_history)
        self.num_clips = (self.total_frames + clip_size - 1) // clip_size
        self.current_clip = 0
        self.current_frame_in_clip = 0
        
        # Load video
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        # UI state
        self.selected_ids = set()
        self.track_colors = {}
        
        # Generate colors
        unique_ids = set()
        for frame in track_history:
            for track in frame:
                unique_ids.add(track["id"])
        
        colormap = plt.cm.get_cmap("tab20")
        for i, track_id in enumerate(sorted(unique_ids)):
            self.track_colors[track_id] = colormap(i % 20)
        
        # Setup UI
        self.setup_figure()
        self.update_clip()
    
    def setup_figure(self) -> None:
        """Setup the clip player figure."""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main video display
        self.ax_img = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3)
        self.ax_img.set_title("Clip-based Tracking Review")
        self.ax_img.axis("off")
        
        # Track timeline
        self.ax_timeline = plt.subplot2grid((4, 3), (0, 2), rowspan=1)
        self.setup_timeline()
        
        # Track checkboxes
        self.ax_checkboxes = plt.subplot2grid((4, 3), (1, 2), rowspan=2)
        unique_ids = sorted(self.track_colors.keys())
        self.checkboxes = CheckButtons(
            self.ax_checkboxes,
            [f"ID {id}" for id in unique_ids],
            [False] * len(unique_ids)
        )
        self.checkboxes.on_clicked(self.on_checkbox_click)
        self.track_id_list = unique_ids
        
        # Clip navigation
        self.ax_clip_slider = plt.subplot2grid((4, 3), (3, 0), colspan=2)
        self.clip_slider = Slider(
            self.ax_clip_slider, "Clip", 0, self.num_clips - 1,
            valinit=0, valstep=1
        )
        self.clip_slider.on_changed(self.on_clip_change)
        
        # Frame within clip
        self.ax_frame_slider = plt.subplot2grid((4, 3), (3, 2))
        self.frame_slider = Slider(
            self.ax_frame_slider, "Frame", 0, self.clip_size - 1,
            valinit=0, valstep=1
        )
        self.frame_slider.on_changed(self.on_frame_change)
        
        # Control buttons
        self.btn_prev_clip = Button(plt.axes([0.05, 0.02, 0.08, 0.03]), "Prev Clip")
        self.btn_next_clip = Button(plt.axes([0.14, 0.02, 0.08, 0.03]), "Next Clip")
        self.btn_done = Button(plt.axes([0.85, 0.02, 0.08, 0.03]), "Done")
        
        self.btn_prev_clip.on_clicked(self.prev_clip)
        self.btn_next_clip.on_clicked(self.next_clip)
        self.btn_done.on_clicked(self.finish_selection)
        
        plt.suptitle(f"Clip Player - {self.video_path.name}", fontsize=14)
    
    def setup_timeline(self) -> None:
        """Setup the track timeline visualization."""
        self.ax_timeline.clear()
        
        # Create timeline for each track ID
        for i, track_id in enumerate(sorted(self.track_colors.keys())):
            y_pos = i * 0.1
            
            # Find frames where this track appears
            track_frames = []
            for frame_idx, frame in enumerate(self.track_history):
                for track in frame:
                    if track["id"] == track_id:
                        track_frames.append(frame_idx)
                        break
            
            # Draw track segments
            if track_frames:
                segments = []
                start = track_frames[0]
                prev = track_frames[0]
                
                for frame in track_frames[1:]:
                    if frame == prev + 1:  # Continuous
                        prev = frame
                    else:  # Gap detected
                        segments.append((start, prev))
                        start = frame
                        prev = frame
                segments.append((start, prev))
                
                # Draw segments
                for start, end in segments:
                    self.ax_timeline.plot(
                        [start, end], [y_pos, y_pos],
                        color=self.track_colors[track_id], linewidth=2
                    )
                    # Mark gaps
                    if start > 0:
                        self.ax_timeline.plot(start, y_pos, 'rx', markersize=4)
            
            # Add track ID label
            self.ax_timeline.text(
                -10, y_pos, f"ID {track_id}",
                ha='right', va='center', fontsize=8
            )
        
        self.ax_timeline.set_xlim(0, self.total_frames)
        self.ax_timeline.set_ylim(-0.05, len(self.track_colors) * 0.1)
        self.ax_timeline.set_xlabel("Frame")
        self.ax_timeline.set_title("Track Timeline (gaps marked with red X)")
        self.ax_timeline.grid(True, alpha=0.3)
    
    def update_clip(self) -> None:
        """Update the current clip display."""
        # Calculate actual frame
        actual_frame = self.current_clip * self.clip_size + self.current_frame_in_clip
        if actual_frame >= self.total_frames:
            actual_frame = self.total_frames - 1
        
        # Clear and update image
        self.ax_img.clear()
        
        # Read frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Convert and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax_img.imshow(frame_rgb)
        
        # Draw tracking
        if actual_frame < len(self.track_history):
            frame_tracks = self.track_history[actual_frame]
            
            for track in frame_tracks:
                track_id = track["id"]
                bbox = track["bbx_xyxy"]
                
                color = self.track_colors[track_id]
                linewidth = 3 if track_id in self.selected_ids else 1
                alpha = 0.8 if track_id in self.selected_ids else 0.6
                
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    linewidth=linewidth, edgecolor=color, facecolor='none', alpha=alpha
                )
                self.ax_img.add_patch(rect)
                
                self.ax_img.text(
                    bbox[0], bbox[1] - 10, f"ID {track_id}",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=alpha),
                    color='white', fontsize=8, fontweight='bold'
                )
        
        # Update title
        clip_info = f"Clip {self.current_clip + 1}/{self.num_clips}, "
        clip_info += f"Frame {self.current_frame_in_clip + 1}/min({self.clip_size}, {self.total_frames - self.current_clip * self.clip_size})"
        self.ax_img.set_title(f"Frame {actual_frame}/{self.total_frames - 1} - {clip_info}")
        self.ax_img.axis("off")
        
        # Update timeline highlight
        self.ax_timeline.axvline(x=actual_frame, color='red', linestyle='--', alpha=0.7)
        
        self.fig.canvas.draw_idle()
    
    def on_clip_change(self, val: float) -> None:
        """Handle clip slider change."""
        self.current_clip = int(val)
        self.current_frame_in_clip = 0
        self.frame_slider.set_val(0)
        self.update_clip()
    
    def on_frame_change(self, val: float) -> None:
        """Handle frame slider change."""
        max_frame = min(self.clip_size - 1, self.total_frames - self.current_clip * self.clip_size - 1)
        self.current_frame_in_clip = min(int(val), max_frame)
        self.update_clip()
    
    def on_checkbox_click(self, label: str) -> None:
        """Handle track ID checkbox click."""
        track_id = int(label.split()[1])
        
        if track_id in self.selected_ids:
            self.selected_ids.remove(track_id)
        else:
            self.selected_ids.add(track_id)
        
        self.update_clip()
    
    def prev_clip(self, event) -> None:
        """Go to previous clip."""
        if self.current_clip > 0:
            self.current_clip -= 1
            self.clip_slider.set_val(self.current_clip)
            self.current_frame_in_clip = 0
            self.frame_slider.set_val(0)
            self.update_clip()
    
    def next_clip(self, event) -> None:
        """Go to next clip."""
        if self.current_clip < self.num_clips - 1:
            self.current_clip += 1
            self.clip_slider.set_val(self.current_clip)
            self.current_frame_in_clip = 0
            self.frame_slider.set_val(0)
            self.update_clip()
    
    def finish_selection(self, event) -> None:
        """Finish selection and close UI."""
        plt.close(self.fig)
    
    def run(self) -> List[int]:
        """Run the clip player UI.
        
        Returns:
            List of selected player track IDs.
        """
        plt.show()
        self.cap.release()
        return sorted(list(self.selected_ids))
