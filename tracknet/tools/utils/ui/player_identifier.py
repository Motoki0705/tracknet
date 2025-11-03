"""Matplotlib UI for selecting player track identifiers per clip."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence

LOGGER = logging.getLogger("tracknet.player_identifier_ui")


@dataclass
class ClipContext:
    """Container representing a single clip in the UI carousel."""

    game_id: str
    clip_id: str
    clip_data: Dict[str, Any]
    assignment: Dict[str, Any] | None


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
            )
            for entry in clips
        ]
        if not self.clips:
            raise ValueError("No clips supplied to PlayerIdentifierUI.")
        self.save_callback = save_callback
        self.roles = list(roles)
        self.current_index = 0
        self.figure = None
        self.track_checkboxes = None
        self.checkbox_axis = None
        self.status_label = None
        self.summary_axes = None
        self.nav_buttons: Dict[str, Any] = {}

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
        self.figure = plt.figure(figsize=(10, 6))
        self.figure.suptitle("TrackNet Player Identifier", fontsize=14)

        self.summary_axes = self.figure.add_axes([0.05, 0.25, 0.4, 0.7])
        self.summary_axes.axis("off")

        checkbox_axis = self.figure.add_axes([0.5, 0.25, 0.2, 0.7])
        labels, selected = self._current_labels()
        self.track_checkboxes = CheckButtons(checkbox_axis, labels, selected)
        self.checkbox_axis = checkbox_axis
        self.track_checkboxes.on_clicked(self._on_checkbox_changed)

        save_axis = self.figure.add_axes([0.75, 0.15, 0.1, 0.08])
        next_axis = self.figure.add_axes([0.86, 0.15, 0.08, 0.08])
        prev_axis = self.figure.add_axes([0.64, 0.15, 0.1, 0.08])
        quit_axis = self.figure.add_axes([0.05, 0.15, 0.1, 0.08])

        self.nav_buttons["save"] = Button(save_axis, "Save", color="#4CAF50", hovercolor="#45a049")
        self.nav_buttons["next"] = Button(next_axis, "→", color="#2196F3", hovercolor="#1e88e5")
        self.nav_buttons["prev"] = Button(prev_axis, "←", color="#2196F3", hovercolor="#1e88e5")
        self.nav_buttons["quit"] = Button(quit_axis, "Quit", color="#f44336", hovercolor="#e53935")

        self.nav_buttons["save"].on_clicked(lambda _event: self._save_current())
        self.nav_buttons["next"].on_clicked(lambda _event: self._advance(1))
        self.nav_buttons["prev"].on_clicked(lambda _event: self._advance(-1))
        self.nav_buttons["quit"].on_clicked(lambda _event: self._quit())

        status_axis = self.figure.add_axes([0.2, 0.05, 0.6, 0.07])
        status_axis.axis("off")
        self.status_label = status_axis.text(0.5, 0.5, "", ha="center", va="center", fontsize=10)

        self._render_current_clip()
        plt.show()

    # ------------------------------------------------------------------ events
    def _on_checkbox_changed(self, _label: str) -> None:
        """Refresh status when a checkbox is toggled."""

        self._update_status("Selection updated. Remember to save.", level="info")

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
        def _sort_key(value: str) -> Any:
            try:
                return int(value)
            except ValueError:
                return value

        labels = [track_id for track_id in sorted(clip_tracks.keys(), key=_sort_key)]
        existing_selection = set(self._existing_selection(context))
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
        """Render summary text and refresh the checkbox widget."""

        labels, selected = self._current_labels()
        if self.checkbox_axis:
            self.checkbox_axis.clear()
            self.checkbox_axis.set_title("Tracks")
            from matplotlib.widgets import CheckButtons

            self.track_checkboxes = CheckButtons(self.checkbox_axis, labels, selected)
            self.track_checkboxes.on_clicked(self._on_checkbox_changed)

        if self.summary_axes:
            self.summary_axes.clear()
            self.summary_axes.axis("off")
            context = self.clips[self.current_index]
            tracks = context.clip_data.get("tracks", {})
            lines = [
                f"Game: {context.game_id}",
                f"Clip: {context.clip_id}",
                f"Tracks detected: {len(tracks)}",
                "",
                "ID | Frames | Coverage | Avg area",
            ]
            for track_id, track in sorted(tracks.items(), key=lambda item: int(item[0])):
                summary = track.get("summary", {})
                lines.append(
                    f"{track_id:>2} | {summary.get('frames', 0):>6} | "
                    f"{summary.get('coverage', 0.0):>6.2f} | {summary.get('average_area', 0.0):>8.2f}"
                )
            self.summary_axes.text(
                0.0,
                1.0,
                "\n".join(lines),
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
            )
        if self.figure:
            self.figure.canvas.draw_idle()

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
