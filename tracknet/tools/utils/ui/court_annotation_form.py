"""CLI-assisted form for annotating tennis court keypoints per game."""

from __future__ import annotations

import getpass
from collections.abc import Iterable, Mapping
from typing import Any

from tracknet.tools.annotation_common import (
    COURT_KEYPOINTS,
    COURT_SKELETON_EDGES,
    iso_timestamp,
)


class CourtAnnotationForm:
    """Assist users in entering court keypoints via textual prompts."""

    def __init__(self, enable_plot: bool = True) -> None:
        """Initialise the form.

        Args:
            enable_plot: Whether to display a Matplotlib preview while editing.
        """

        self.enable_plot = enable_plot
        self._figure = None
        self._axes = None

    def collect(
        self,
        game_id: str,
        existing: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Collect court annotations for ``game_id``.

        Args:
            game_id: Identifier of the game being annotated.
            existing: Previous annotation used to pre-fill values.

        Returns:
            Dict[str, Any]: Court annotation payload containing keypoints and metadata.
        """

        existing_keypoints = {
            kp["index"]: kp
            for kp in (existing or {}).get("keypoints", [])
            if isinstance(kp, Mapping)
        }
        keypoints: list[dict[str, Any]] = []

        print(f"\nAnnotating court for {game_id}")
        print(
            "Enter coordinates as `x,y` in pixel space. Leave empty to reuse existing values."
        )
        for entry in COURT_KEYPOINTS:
            idx = entry["index"]
            label = entry["name"]
            current = existing_keypoints.get(idx)
            default_display = (
                f"{current.get('x'):.2f},{current.get('y'):.2f}"
                if current
                and current.get("x") is not None
                and current.get("y") is not None
                else ""
            )
            point = self._prompt_point(label, default_display)
            keypoints.append(
                {
                    "index": idx,
                    "name": label,
                    "x": point[0],
                    "y": point[1],
                    "confidence": current.get("confidence") if current else None,
                    "visible": current.get("visible") if current else None,
                }
            )
            self._update_plot(keypoints)

        reference_frame = self._prompt_reference_frame(
            existing.get("reference_frame") if existing else None
        )
        metadata = dict(existing.get("metadata", {})) if existing else {}
        metadata["last_updated"] = iso_timestamp()

        return {
            "annotator": getpass.getuser(),
            "keypoints": keypoints,
            "skeleton": COURT_SKELETON_EDGES,
            "reference_frame": reference_frame,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------ helpers
    def _prompt_point(
        self, label: str, default_display: str
    ) -> tuple[float | None, float | None]:
        """Prompt the user for a single keypoint coordinate."""

        while True:
            prompt = (
                f"{label} [{default_display}]: "
                if default_display
                else f"{label} [x,y]: "
            )
            raw = input(prompt).strip()
            if raw == "" and default_display:
                x_str, y_str = default_display.split(",")
                return float(x_str), float(y_str)
            if raw == "":
                return None, None
            try:
                x_str, y_str = [token.strip() for token in raw.split(",")]
                return float(x_str), float(y_str)
            except (ValueError, IndexError):
                print("Invalid format. Provide coordinates as 'x,y'.")

    def _prompt_reference_frame(self, current: Any) -> int | None:
        """Prompt for a reference frame index."""

        while True:
            raw = input(
                f"Reference frame [{current if current is not None else ''}]: "
            ).strip()
            if raw == "" and current is not None:
                return int(current)
            if raw == "":
                return None
            try:
                return int(raw)
            except ValueError:
                print("Enter an integer frame index or leave empty.")

    def _update_plot(self, keypoints: Iterable[Mapping[str, Any]]) -> None:
        """Refresh the Matplotlib preview with current keypoints."""

        if not self.enable_plot:
            return
        import matplotlib.pyplot as plt

        if self._figure is None or self._axes is None:
            self._figure, self._axes = plt.subplots(figsize=(6, 6))
            self._axes.set_title("Court keypoints (preview)")
            self._axes.set_xlabel("X")
            self._axes.set_ylabel("Y")
            self._axes.invert_yaxis()
        self._axes.clear()
        xs = [kp.get("x") for kp in keypoints if kp.get("x") is not None]
        ys = [kp.get("y") for kp in keypoints if kp.get("y") is not None]
        labels = [kp.get("name") for kp in keypoints if kp.get("x") is not None]
        self._axes.scatter(xs, ys, c="tab:orange")
        for x, y, label in zip(xs, ys, labels, strict=False):
            self._axes.text(x, y, label, fontsize=8, color="black")
        self._figure.canvas.draw_idle()
        # Ensure the UI updates while prompting.
        plt.pause(0.001)
