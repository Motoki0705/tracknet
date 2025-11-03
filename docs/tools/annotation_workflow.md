# Annotation Workflow

TrackNet’s unified annotation pipeline combines automated tracking with two interactive
UIs. Follow this guide to produce `data/tracknet/annotations.json`.

## 1. Person Tracking (YOLOv8)

```bash
uv run python tracknet/tools/player_tracker.py --mode detect
```

Key flags:

- `--games game1 game2` – restrict processing to given games.
- `--clips Clip1 Clip2` – restrict within a game.
- `--force` – overwrite cached results in `outputs/tracking/person_tracks.json`.
- `--dry-run` – execute without writing output (useful for validation).

### Verification

1. (Automated) `pytest` contains `test_player_tracker_detect_and_cache` ensuring clip data
   is captured and caching works.
2. (Manual) Inspect `outputs/tracking/person_tracks.json`:
   - Confirm `games.<game>.clips.<clip>.tracks` contains track IDs.
   - Ensure each detection has `frame`, `bbox`, and `visibility`.

## 2. Player Assignment UI

```bash
uv run python tracknet/tools/player_tracker.py --mode assign
```

The Matplotlib UI provides:

- Track list with multi-select checkboxes.
- Navigation buttons (`←`, `→`) to move between clips.
- `Save` to persist selections to `outputs/tracking/player_assignments.json`.
- `Quit` to exit.

Selections update `data/tracknet/annotations.json` incrementally when available.

### Manual UI Test Checklist (required)

Perform once per environment to verify interactivity:

1. Launch the UI (`--mode assign`) with a dataset containing person tracks.
2. Confirm the track summary table shows per-track statistics.
3. Toggle multiple checkboxes; the status banner should display “Selection updated”.
4. Click `Save` and verify the banner reports success; `player_assignments.json`
   should update the `selected_tracks` array.
5. Use `→` and `←` to navigate between clips; the active clip indicator should update.
6. Re-launch the UI and ensure previous selections are restored (resume support).

## 3. Court Annotation Manager

```bash
uv run python tracknet/tools/court_annotation_manager.py
```

Default behaviour iterates detected games; use `--games` to target specific matches.
Set `--force` to overwrite existing annotations and `--no-plot` in headless environments.

The CLI prompts for 15 keypoints plus an optional reference frame. If plotting is enabled,
coordinates are previewed live.

Result saved to `outputs/court_annotations/games.json`.

## 4. Conversion to Unified JSON

```bash
uv run python tracknet/tools/annotation_converter.py
```

Important flags:

- `--dry-run` – display merged payload without writing.
- `--games / --clips` – subset of data to convert.

The converter reads:

- `data/tracknet/<game>/<clip>/Label.csv` (ball annotations; `ball.csv` also supported)
- `outputs/tracking/person_tracks.json`
- `outputs/tracking/player_assignments.json`
- `outputs/court_annotations/games.json`

Output schema is documented in `openspec/changes/add-unified-annotation-pipeline/design.md`.

## 5. Automated Validation

```bash
uv run pytest tests/tools/test_annotation_pipeline.py
```

Coverage:

- YOLO import (skipped when unavailable).
- Tracker integration & caching.
- JSON schema checks for tracking, player assignments, court annotations.
- Converter multi-stage and idempotency validation.

Running the full suite after each major change ensures contract compatibility.
