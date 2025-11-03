"""Matplotlib UI for selecting player track identifiers per clip."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger("tracknet.player_identifier_ui")


@dataclass
class ClipContext:
    """Container representing a single clip in the UI carousel."""

    game_id: str
    clip_id: str
    clip_data: Dict[str, Any]
    assignment: Dict[str, Any] | None
    frame_dir: Path | None = None
    media_source: Path | None = None
    frame_paths: Dict[int, Path] = field(default_factory=dict)
    frame_detections: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    preview_frame: Optional[int] = None
    preview_image: Any | None = None

    def __post_init__(self) -> None:
        """Normalise optional path fields after initialisation."""

        if self.frame_dir:
            frame_path = Path(self.frame_dir)
            self.frame_dir = frame_path if frame_path.is_dir() else None
        if self.media_source:
            source_path = Path(self.media_source)
            self.media_source = source_path if source_path.exists() else None


class PlayerIdentifierUI:
    """Interactive Matplotlib interface for choosing player tracks."""

    def __init__(
        self,
        clips: Sequence[Dict[str, Any]],
        save_callback: Callable[[str, str, List[str]], None],
        roles: Iterable[str] = ("player_a", "player_b"),
    ) -> None:
        self.clips: List[ClipContext] = [
            ClipContext(
                game_id=entry["game_id"],
                clip_id=entry["clip_id"],
                clip_data=entry["clip_data"],
                assignment=entry.get("assignment"),
                frame_dir=entry.get("frame_dir"),
                media_source=entry.get("media_source"),
            )
            for entry in clips
        ]
        if not self.clips:
            raise ValueError("No clips supplied to PlayerIdentifierUI.")
        self.save_callback = save_callback
        self.roles = list(roles)
        self.current_index = 0
        self.figure = None
        self.preview_axes = None
        self.track_checkboxes = None
        self.checkbox_axis = None
        self.status_label = None
        self.summary_axes = None
        self.nav_buttons: Dict[str, Any] = {}
        self._color_palette: Optional[List[Any]] = None

    # ------------------------------------------------------------------ public
    def launch(self) -> None:
        """Start the Matplotlib UI."""

        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons

        backend = matplotlib.get_backend().lower()
        if backend in {"agg", "svg", "ps"}:
            raise RuntimeError(
                f"Matplotlib backend '{backend}' is non-interactive. "
                "Configure an interactive backend (e.g., TkAgg) before launching the player UI."
            )

        plt.close("all")
        self.figure = plt.figure(figsize=(12, 7))
        self.figure.suptitle("TrackNet Player Identifier", fontsize=14)

        self.preview_axes = self.figure.add_axes([0.05, 0.25, 0.45, 0.7])
        self.preview_axes.axis("off")
        self.summary_axes = self.figure.add_axes([0.52, 0.25, 0.18, 0.7])
        self.summary_axes.axis("off")

        checkbox_axis = self.figure.add_axes([0.72, 0.25, 0.23, 0.7])
        labels, selected = self._current_labels()
        self.track_checkboxes = CheckButtons(checkbox_axis, labels, selected)
        self.checkbox_axis = checkbox_axis
        self.track_checkboxes.on_clicked(self._on_checkbox_changed)

        save_axis = self.figure.add_axes([0.72, 0.12, 0.1, 0.08])
        next_axis = self.figure.add_axes([0.84, 0.12, 0.1, 0.08])
        prev_axis = self.figure.add_axes([0.60, 0.12, 0.1, 0.08])
        quit_axis = self.figure.add_axes([0.05, 0.12, 0.1, 0.08])

        self.nav_buttons["save"] = Button(save_axis, "Save", color="#4CAF50", hovercolor="#45a049")
        self.nav_buttons["next"] = Button(next_axis, "→", color="#2196F3", hovercolor="#1e88e5")
        self.nav_buttons["prev"] = Button(prev_axis, "←", color="#2196F3", hovercolor="#1e88e5")
        self.nav_buttons["quit"] = Button(quit_axis, "Quit", color="#f44336", hovercolor="#e53935")

        self.nav_buttons["save"].on_clicked(lambda _event: self._save_current())
        self.nav_buttons["next"].on_clicked(lambda _event: self._advance(1))
        self.nav_buttons["prev"].on_clicked(lambda _event: self._advance(-1))
        self.nav_buttons["quit"].on_clicked(lambda _event: self._quit())

        status_axis = self.figure.add_axes([0.25, 0.04, 0.5, 0.06])
        status_axis.axis("off")
        self.status_label = status_axis.text(0.5, 0.5, "", ha="center", va="center", fontsize=10)

        self._render_current_clip()
        plt.show()

    # ------------------------------------------------------------------ events
    def _on_checkbox_changed(self, _label: str) -> None:
        """Refresh status when a checkbox is toggled."""

        self._update_status("Selection updated. Remember to save.", level="info")
        self._refresh_visuals()

    def _refresh_visuals(self) -> None:
        """Redraw preview and summary panels for the active clip."""

        context = self.clips[self.current_index]
        selected_ids = set(self._current_selection())
        self._render_preview(context, selected_ids)
        self._render_summary(context, selected_ids)
        if self.figure:
            self.figure.canvas.draw_idle()

    def _advance(self, step: int) -> None:
        """Move to the next/previous clip in the carousel.

        Args:
            step: Positive for forward navigation, negative for backward.
        """

        total = len(self.clips)
        self.current_index = (self.current_index + step) % total
        self._render_current_clip()
        self._update_status(f"Moved to clip {self.current_index + 1}/{total}.", level="info")

    def _save_current(self) -> None:
        """Persist the current selection via the save callback."""

        context = self.clips[self.current_index]
        selections = self._current_selection()
        if len(selections) < len(self.roles):
            self._update_status(
                f"Select at least {len(self.roles)} tracks before saving.", level="warning"
            )
            return
        try:
            self.save_callback(context.game_id, context.clip_id, selections)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to persist selection: %s", exc)
            self._update_status(f"Save failed: {exc}", level="error")
            return
        context.assignment = context.assignment or {}
        context.assignment["selected_tracks"] = list(selections)
        self._update_status("Selection saved.", level="success")
        # Refresh checkboxes to reflect persisted order.
        self._render_current_clip()

    def _quit(self) -> None:
        import matplotlib.pyplot as plt

        """Close the UI window and terminate the Matplotlib session."""

        self._update_status("Closing UI.", level="info")
        plt.close(self.figure)

    # ------------------------------------------------------------------ helpers
    def _current_labels(self) -> tuple[List[str], List[bool]]:
        """Return checkbox labels and initial selection states."""

        context = self.clips[self.current_index]
        clip_tracks = context.clip_data.get("tracks", {})

        labels = [track_id for track_id in self._sorted_track_ids(clip_tracks.keys())]
        existing_selection = set(self._existing_selection(context))
        if not existing_selection and labels:
            default_count = min(len(self.roles), len(labels))
            existing_selection = set(labels[:default_count])
        selected = [track_id in existing_selection for track_id in labels]
        return labels, selected

    def _current_selection(self) -> List[str]:
        """Return the currently selected track identifiers."""

        if not self.track_checkboxes:
            return []
        return [
            label.get_text()
            for label, state in zip(self.track_checkboxes.labels, self.track_checkboxes.get_status())
            if state
        ]

    def _existing_selection(self, context: ClipContext) -> List[str]:
        """Resolve previously saved selections for ``context``."""

        if context.assignment:
            if "selected_tracks" in context.assignment:
                return list(context.assignment["selected_tracks"])
            players = context.assignment.get("players", {})
            return [player.get("track_id") for player in players.values() if player.get("track_id")]
        return []

    def _render_current_clip(self) -> None:
        """Render the active clip across preview, summary, and checkbox panels."""

        labels, selected_states = self._current_labels()
        if self.checkbox_axis:
            self.checkbox_axis.clear()
            self.checkbox_axis.set_title("Tracks")
            from matplotlib.widgets import CheckButtons

            self.track_checkboxes = CheckButtons(self.checkbox_axis, labels, selected_states)
            self.track_checkboxes.on_clicked(self._on_checkbox_changed)

        selected_ids = {label for label, state in zip(labels, selected_states) if state}
        context = self.clips[self.current_index]
        self._render_preview(context, selected_ids)
        self._render_summary(context, selected_ids)
        if self.figure:
            self.figure.canvas.draw_idle()

    def _render_preview(self, context: ClipContext, selected_ids: Iterable[str]) -> None:
        """Render an image preview with track overlays for ``context``.

        Args:
            context: Active clip container to visualise.
            selected_ids: Track identifiers currently selected in the UI.
        """

        if not self.preview_axes:
            return
        frame_index = self._resolve_preview_frame(context)
        selected_set = set(selected_ids)
        image = self._load_preview_image(context, frame_index)
        self.preview_axes.clear()
        self.preview_axes.axis("off")
        if image is None:
            self.preview_axes.text(
                0.5,
                0.5,
                "Preview unavailable",
                ha="center",
                va="center",
                fontsize=11,
                color="#555555",
            )
            return

        self.preview_axes.imshow(image)
        if frame_index is not None:
            self.preview_axes.set_title(f"{context.game_id}/{context.clip_id} – frame {frame_index}")
        detections = self._detections_for_frame(context, frame_index)
        if detections:
            from matplotlib import patches

            for track_id in self._sorted_track_ids(detections.keys()):
                det = detections[track_id]
                bbox = det.get("bbox")
                if not bbox:
                    continue
                x, y, width, height = bbox
                color = self._color_for_track(track_id)
                linewidth = 3 if track_id in selected_set else 1.5
                rectangle = patches.Rectangle(
                    (x, y),
                    width,
                    height,
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor="none",
                )
                self.preview_axes.add_patch(rectangle)
                self.preview_axes.text(
                    x,
                    max(0, y - 8),
                    f"ID {track_id}",
                    color=color,
                    fontsize=8,
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.0, "edgecolor": "none"},
                )
        height, width = image.shape[0], image.shape[1]
        self.preview_axes.set_xlim(0, width)
        self.preview_axes.set_ylim(height, 0)

    def _render_summary(self, context: ClipContext, selected_ids: Iterable[str]) -> None:
        """Render textual summary metrics for ``context``.

        Args:
            context: Active clip container to summarise.
            selected_ids: Track identifiers currently selected in the UI.
        """

        if not self.summary_axes:
            return
        self.summary_axes.clear()
        self.summary_axes.axis("off")
        tracks = context.clip_data.get("tracks", {})
        selected_set = set(selected_ids)
        lines = [
            f"Game: {context.game_id}",
            f"Clip: {context.clip_id}",
            f"Tracks detected: {len(tracks)}",
            "",
            "✓ ID | Frames | Coverage | Avg area",
        ]
        for track_id in self._sorted_track_ids(tracks.keys()):
            track = tracks[track_id]
            summary = track.get("summary", {})
            mark = "✓" if track_id in selected_set else " "
            lines.append(
                f"{mark} {track_id:>2} | {summary.get('frames', 0):>6} | "
                f"{summary.get('coverage', 0.0):>6.2f} | {summary.get('average_area', 0.0):>8.2f}"
            )
        if not tracks:
            lines.append("No track data available.")
        self.summary_axes.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )

    def _update_status(self, message: str, level: str = "info") -> None:
        """Update the status banner with ``message``.

        Args:
            message: Text displayed to the user.
            level: Semantic level controlling banner colour.
        """

        if not self.status_label:
            return
        colors = {
            "info": "#1976d2",
            "success": "#2e7d32",
            "warning": "#f9a825",
            "error": "#c62828",
        }
        self.status_label.set_text(message)
        self.status_label.set_color(colors.get(level, "#1976d2"))
        if self.figure:
            self.figure.canvas.draw_idle()

    def _sorted_track_ids(self, track_ids: Iterable[Any]) -> List[str]:
        """Return identifiers sorted numerically when possible.

        Args:
            track_ids: Raw identifiers sourced from the track dictionary.

        Returns:
            List[str]: Sorted track identifiers as strings.
        """

        def sort_key(value: str) -> Tuple[int, Any]:
            try:
                return (0, int(value))
            except ValueError:
                return (1, value)

        return sorted([str(track_id) for track_id in track_ids], key=sort_key)

    def _detections_for_frame(
        self,
        context: ClipContext,
        frame_index: Optional[int],
    ) -> Mapping[str, Mapping[str, Any]]:
        """Return detections present on ``frame_index`` for ``context``.

        Args:
            context: Clip container containing detection metadata.
            frame_index: Frame number of interest.

        Returns:
            Mapping[str, Mapping[str, Any]]: Per-track detection data for the frame.
        """

        if frame_index is None:
            return {}
        detections = self._ensure_frame_detections(context)
        return detections.get(int(frame_index), {})

    def _resolve_preview_frame(self, context: ClipContext) -> Optional[int]:
        """Choose a representative frame number for ``context``.

        Args:
            context: Clip container containing detection metadata.

        Returns:
            Optional[int]: Frame number showing the most simultaneous tracks.
        """

        if context.preview_frame is not None:
            return context.preview_frame
        frame_detections = self._ensure_frame_detections(context)
        if not frame_detections:
            context.preview_frame = None
            return None
        best_frame, _ = max(frame_detections.items(), key=lambda item: (len(item[1]), -item[0]))
        context.preview_frame = best_frame
        return best_frame

    def _ensure_frame_detections(self, context: ClipContext) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """Index detections by frame for faster lookup.

        Args:
            context: Clip container to index.

        Returns:
            Dict[int, Dict[str, Dict[str, Any]]]: Mapping of frame numbers to per-track detections.
        """

        if context.frame_detections:
            return context.frame_detections
        frame_index: Dict[int, Dict[str, Dict[str, Any]]] = {}
        tracks = context.clip_data.get("tracks", {})
        for raw_track_id, track in tracks.items():
            track_id = str(raw_track_id)
            for detection in track.get("detections", []):
                frame_value = detection.get("frame")
                if frame_value is None:
                    continue
                try:
                    frame_number = int(frame_value)
                except (TypeError, ValueError):
                    continue
                frame_detections = frame_index.setdefault(frame_number, {})
                frame_detections[track_id] = detection
        context.frame_detections = frame_index
        return frame_index

    def _ensure_frame_map(self, context: ClipContext) -> Dict[int, Path]:
        """Cache a mapping of frame numbers to image paths for ``context``.

        Args:
            context: Clip container with optional frame directory.

        Returns:
            Dict[int, Path]: Mapping of frame indices to image files.
        """

        if context.frame_paths:
            return context.frame_paths
        if not context.frame_dir or not context.frame_dir.is_dir():
            return {}

        def sort_key(path: Path) -> Tuple[int, str]:
            try:
                return (0, int(path.stem))
            except ValueError:
                return (1, path.stem)

        for entry in sorted(context.frame_dir.iterdir(), key=sort_key):
            if entry.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                frame_number = int(entry.stem)
            except ValueError:
                continue
            context.frame_paths[frame_number] = entry
        return context.frame_paths

    def _resolve_frame_path(self, context: ClipContext, frame_index: Optional[int]) -> Optional[Path]:
        """Resolve the best available frame path for the requested frame.

        Args:
            context: Clip container with cached frame paths.
            frame_index: Desired frame number.

        Returns:
            Optional[Path]: Path to the closest matching frame image.
        """

        if frame_index is None:
            return None
        frame_map = self._ensure_frame_map(context)
        if not frame_map:
            return None
        if frame_index in frame_map:
            return frame_map[frame_index]
        nearest = min(frame_map.keys(), key=lambda candidate: abs(candidate - frame_index))
        return frame_map.get(nearest)

    def _load_preview_image(self, context: ClipContext, frame_index: Optional[int]) -> Any | None:
        """Load and cache the preview image for ``frame_index``.

        Args:
            context: Clip container storing cache metadata.
            frame_index: Frame number earmarked for preview.

        Returns:
            Optional[Any]: Loaded image (RGB array) or ``None`` when unavailable.
        """

        if frame_index is None:
            return None
        if context.preview_image is not None and context.preview_frame == frame_index:
            return context.preview_image

        image = None
        frame_path = self._resolve_frame_path(context, frame_index)
        if frame_path and frame_path.exists():
            from matplotlib import image as mpimg

            try:
                image = mpimg.imread(frame_path)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to load frame image %s: %s", frame_path, exc)
                image = None
        elif context.media_source:
            image = self._read_frame_from_video(context.media_source, frame_index)

        if image is not None:
            context.preview_frame = frame_index
            context.preview_image = image
        return image

    def _read_frame_from_video(self, source: Path, frame_index: int) -> Any | None:
        """Extract a single frame from ``source`` using OpenCV.

        Args:
            source: Path to a video file.
            frame_index: Frame number to capture.

        Returns:
            Optional[Any]: RGB image array or ``None`` when extraction fails.
        """

        try:
            import cv2
        except ImportError:  # pragma: no cover - optional dependency
            LOGGER.debug("OpenCV not available; cannot render video preview for %s", source)
            return None

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            LOGGER.warning("Unable to open video source %s for preview.", source)
            return None

        try:
            if frame_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
            success, frame = capture.read()
            if not success or frame is None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = capture.read()
            if not success or frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()

    def _color_for_track(self, track_id: str) -> Any:
        """Return a stable colour for ``track_id``.

        Args:
            track_id: Identifier of the track to colourise.

        Returns:
            Any: Matplotlib-compatible colour specification.
        """

        if self._color_palette is None:
            from matplotlib import cm

            cmap = cm.get_cmap("tab10", 10)
            self._color_palette = [cmap(index) for index in range(cmap.N)]
        palette = self._color_palette or ["#ff9800"]
        return palette[abs(hash(track_id)) % len(palette)]
