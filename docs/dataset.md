# TrackNet Dataset

This document summarizes the processed tennis tracking dataset under `data/tracknet/` (renamed from the previous `data/processed/` path), with an emphasis on how to work with it for model training.

## Directory Layout
- `data/tracknet/game*` &mdash; 10 distinct matches (treated as high-level sessions).
- Each game folder contains multiple `Clip*` subfolders (95 clips total).
- A clip bundles sequential frames (`0000.jpg`, `0001.jpg`, …) plus a `Label.csv`.

Recommended training granularity:
- Treat games as independent domains to avoid leakage when making train/val/test splits.
- Clips provide ready-to-use short sequences; maintain clip boundaries if temporal context matters.

## Media Assets
- Frames are JPEG images encoded at 1280×720 px (confirmed via `file data/tracknet/game1/Clip1/0000.jpg`).
- Filenames are zero-padded frame indices matching the entries in `Label.csv`.

## Annotation Schema (`Label.csv`)
Header columns: `file name, visibility, x-coordinate, y-coordinate, status`

| Field         | Type    | Description | Notes for Training |
|---------------|---------|-------------|--------------------|
| `file name`   | string  | Frame filename (e.g., `0000.jpg`). | Join with image; sorted chronologically. |
| `visibility`  | integer | Ball visibility class: `0` (not visible), `1` (visible), `2` (partially occluded), `3` (high occlusion/ambiguous) | Imbalanced: most frames are `1`. Decide whether to mask `0` entries or treat as negative samples when supervising coordinates. |
| `x-coordinate`| string/integer | Pixel x-position of the ball. Blank when `visibility=0`. | Cast to int; guard against missing values. |
| `y-coordinate`| string/integer | Pixel y-position of the ball. Blank when `visibility=0`. | Same handling as x. |
| `status`      | integer | Rally state metadata: `0` (in-play), `1` (serve), `2` (bounce), blank (~729 rows) | Can enable multi-task training; blanks align with `visibility=0` frames. |

There are 19,835 labeled frames overall. Distribution highlights:
- `visibility`: `1` (17,632), `2` (1,392), `0` (729), `3` (82)
- `status`: `0` (18,067), `1` (516), `2` (523), blank (729)

## Suggested Training Workflow
- **Parsing:** Read `Label.csv` with `csv.DictReader` or pandas; coerce numeric columns and drop rows with empty coordinates as needed.
- **Masking:** When supervising ball coordinates, mask loss where `visibility=0` or coordinates are missing. Consider weighting `visibility` classes to offset imbalance.
- **Augmentation:** Standard spatial augmentations must update x/y labels. Ensure aspect ratio of 1280×720 is preserved or coordinates are rescaled.
- **Temporal Models:** Clips already provide contiguous frames; sample sliding windows within clips to preserve order.
- **Evaluation Splits:** Suggested approach is to reserve full games or clips for validation/test sets to prevent near-duplicate frames leaking into training.

## Quick Reference Scripts
Example Python snippet to summarize a clip:

```python
import csv
from pathlib import Path

clip = Path("data/tracknet/game1/Clip1")
with open(clip / "Label.csv") as f:
    rows = list(csv.DictReader(f))

print(f"{clip}: {len(rows)} frames")
print("Visibility counts:", Counter(row["visibility"] for row in rows))
```

Adapt this pattern to build dataset loaders or sanity-check label integrity before training.
