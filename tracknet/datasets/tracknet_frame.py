"""TrackNet frame-level dataset.

Parses the TrackNet dataset organized as:
``data/tracknet/game*/Clip*/{0000..}.jpg`` with corresponding ``Label.csv``
files containing columns: ``file name,visibility,x-coordinate,y-coordinate,status``.

Each dataset item corresponds to a single frame with a target coordinate and a
visibility flag. Heatmap generation is deferred to collate utilities.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tracknet.datasets.base.image_dataset import BaseImageDataset, PreprocessConfig


@dataclass
class TrackNetFrameDatasetConfig:
    """Configuration for TrackNet frame dataset.

    Attributes:
        root: Root directory of TrackNet data (e.g., ``data/tracknet``).
        games: Iterable of game directory names to include (e.g., ``["game1"]``).
        preprocess: Preprocessing/augmentation settings.
    """

    root: str
    games: Iterable[str]
    preprocess: PreprocessConfig | None = None


class TrackNetFrameDataset(BaseImageDataset):
    """Dataset for single frames from TrackNet with coordinates/visibility."""

    def __init__(self, cfg: TrackNetFrameDatasetConfig) -> None:
        super().__init__(cfg.preprocess)
        self.root = Path(cfg.root)
        self.games = list(cfg.games)
        self.records: list[dict[str, Any]] = []
        self._build_index()

    # ----- Indexing -----
    def _build_index(self) -> None:
        for game in self.games:
            game_dir = self.root / game
            if not game_dir.exists():
                continue
            for clip_dir in sorted(
                p
                for p in game_dir.iterdir()
                if p.is_dir() and p.name.startswith("Clip")
            ):
                label_csv = clip_dir / "Label.csv"
                if not label_csv.exists():
                    # Skip if missing labels
                    continue
                # Build a map from filename -> (vis, x, y)
                labels: dict[str, tuple[int, float, float]] = {}

                def _safe_float(v: str | float | int, default: float = 0.0) -> float:
                    try:
                        s = str(v).strip()
                        if s == "" or s.lower() == "nan":
                            return float(default)
                        return float(s)
                    except Exception:
                        return float(default)

                with open(label_csv, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fname = row["file name"].strip()
                        vis = int(row.get("visibility", 1))
                        x = _safe_float(row.get("x-coordinate", 0.0))
                        y = _safe_float(row.get("y-coordinate", 0.0))
                        labels[fname] = (vis, x, y)

                for img_path in sorted(
                    p for p in clip_dir.iterdir() if p.suffix.lower() == ".jpg"
                ):
                    key = img_path.name
                    if key not in labels:
                        continue
                    vis, x, y = labels[key]
                    self.records.append(
                        {
                            "path": str(img_path),
                            "coord": (x, y),
                            "visibility": vis,
                            "game": game,
                            "clip": clip_dir.name,
                        }
                    )

    # ----- Base class hooks -----
    def _get_record(self, index: int) -> dict[str, Any]:
        return self.records[index]

    def __len__(self) -> int:
        return len(self.records)
