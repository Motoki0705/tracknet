## Overview
We are introducing a tooling suite that unifies ball, player, and court annotations into a single JSON artifact. The work spans three functional areas:

1. **Data aggregation (`tracknet/tools/annotation_converter.py`)** — merge disparate annotation sources into `data/tracknet/annotations.json`.
2. **Player tracking & selection (`tracknet/tools/player_tracker.py`, `tracknet/tools/utils/ui/player_identifier.py`)** — batch run person tracking and collect user decisions on which track IDs represent players using a Matplotlib UI.
3. **Court annotation (`tracknet/tools/court_annotation_manager.py`, `tracknet/tools/utils/ui/court_annotation_form.py`)** — capture static per-game court geometry via UI and feed it into the unified artifact.
4. **Video generation (`tracknet/tools/utils/video_generator.py`)** — convert frame sequences to temporary videos for YOLOv8 processing.

All tooling should be idempotent and safe to re-run, enabling iterative annotation without overwriting user-confirmed data unintentionally.

## Video Generation Architecture

Since the dataset stores frames as individual JPG images (`data/tracknet/<game>/<clip>/XXXX.jpg`), we need on-the-fly video generation for YOLOv8 tracking:

### Frame-to-Video Conversion
- **Input**: Sequential JPG frames named with 4-digit zero-padding (`0001.jpg`, `0002.jpg`, etc.)
- **Processing**: Sort frames numerically, generate MP4 at 30 FPS using OpenCV VideoWriter
- **Output**: Temporary video files in system temp directory, automatically cleaned up
- **Fallback**: If video generation fails, log warning and skip the clip

### Integration Points
- `player_tracker.py --mode detect` calls video generator before YOLOv8 tracking
- Temporary videos are created per-clip, used immediately, then deleted
- Error handling ensures missing/corrupt frames don't crash the entire pipeline
- Frame ordering preserved based on numeric filename parsing

## Data Model

```jsonc
{
  "schema_version": "1.0.0",
  "generated_at": "2024-05-01T12:34:56Z",
  "games": {
    "game_id": {
      "court": {
        "keypoints": [
          {"index": 0, "name": "far doubles corner left", "x": 0.0, "y": 0.0},
          {"index": 1, "name": "far doubles corner right", "x": 0.0, "y": 0.0},
          {"index": 2, "name": "near doubles corner left", "x": 0.0, "y": 0.0},
          {"index": 3, "name": "near doubles corner right", "x": 0.0, "y": 0.0},
          {"index": 4, "name": "far singles corner left", "x": 0.0, "y": 0.0},
          {"index": 5, "name": "near singles corner left", "x": 0.0, "y": 0.0},
          {"index": 6, "name": "far singles corner right", "x": 0.0, "y": 0.0},
          {"index": 7, "name": "near singles corner right", "x": 0.0, "y": 0.0},
          {"index": 8, "name": "far service-line endpoint left", "x": 0.0, "y": 0.0},
          {"index": 9, "name": "far service-line endpoint right", "x": 0.0, "y": 0.0},
          {"index": 10, "name": "near service-line endpoint left", "x": 0.0, "y": 0.0},
          {"index": 11, "name": "near service-line endpoint right", "x": 0.0, "y": 0.0},
          {"index": 12, "name": "far service T", "x": 0.0, "y": 0.0},
          {"index": 13, "name": "near service T", "x": 0.0, "y": 0.0},
          {"index": 14, "name": "net center", "x": 0.0, "y": 0.0}
        ],
        "skeleton": [
          [1, 2],
          [3, 4],
          [1, 3],
          [2, 4],
          [5, 6],
          [7, 8],
          [9, 10],
          [11, 12],
          [13, 14]
        ]
      },
      "clips": {
        "clip_id": {
          "source": "relative/path/to/clip.mp4",
          "ball": {
            "frames": [0, 1, 2, ...],
            "x": [123.0, 124.2, ...],
            "y": [456.0, 455.1, ...],
            "visibility": [1, 1, 2, ...],
            "status": [0, 1, null, ...]
          },
          "players": {
            "player_a": {
              "track_id": "5",
              "frames": [0, 1, 2, ...],
              "bbox": [[10.0, 20.0, 50.0, 120.0], ...],
              "visibility": [1.0, 0.9, ...]
            },
            "player_b": {
              "track_id": "12",
              "frames": [0, 1, 2, ...],
              "bbox": [[600.0, 18.0, 48.0, 118.0], ...],
              "visibility": [0.95, 0.92, ...]
            }
          }
        }
      }
    }
  }
}
```

- **Ball annotations** retain frame-level positions sourced from CSVs, augmented with values defined in `docs/data/tracknet.md`:
  - `visibility` integers (`0`: not visible, `1`: visible, `2`: partially occluded, `3`: highly occluded/ambiguous).
  - `status` integers (`0`: in-screen rally, `1`: shot event, `2`: net interaction, `null`: unspecified/missing).
- **Players** map semantic roles (e.g., `player_a`, `player_b`) to tracker IDs and record per-frame bounding boxes (plus optional metadata) to support pose/trajectory analyses.
- **Court** holds per-game geometry shared across clips, represented by 15 keypoints and 9 skeleton edges:
  - Keypoints indexed 0–14 map to labeled court landmarks (`far doubles corner left`, …, `net center`).
  - Skeleton entries define line segments between keypoint indices to reconstruct key court lines.
The array-of-values layout for ball and player time series keeps related data contiguous for cache-friendly loading while allowing new channels (e.g., spin, velocity) to be appended without restructuring existing entries.

### File Schemas

#### `data/tracknet/annotations.json`

- **Type**: object
- **Required keys**:
  - `schema_version`: string (semantic version for the annotation format, e.g., `"1.0.0"`)
  - `generated_at`: ISO 8601 timestamp string
  - `games`: object keyed by `game_id`
- **Game entry** (`games[game_id]`):
  - `court`: object
    - `keypoints`: array of 15 objects; each object requires `index` (int 0–14), `name` (string matching keypoint definition), `x` (float), `y` (float)
    - `skeleton`: array of 9 two-element integer arrays referencing keypoint indices
  - `clips`: object keyed by `clip_id`
- **Clip entry** (`clips[clip_id]`):
  - `source`: relative path string to the clip media
  - `ball`: object with required arrays `frames`, `x`, `y`, `visibility`, `status`; arrays must be equal length and ordered by frame index.
    - `visibility`: integer array constrained to `{0,1,2,3}` per `docs/data/tracknet.md`.
    - `status`: integer array permitting `{0,1,2}` (per `docs/data/tracknet.md`) with `null` for missing values.
  - `players`: object keyed by semantic roles (`player_a`, `player_b`, optional extras). Each role object requires:
    - `track_id`: string (matching tracker identifiers)
    - `frames`: array of integers aligned with `bbox`
    - `bbox`: array of `[x, y, width, height]` floats (pixel coordinates)
    - `visibility`: optional float array (0–1) aligned with frames
    - `state`: optional string array (e.g., `ready`, `swing`, `out_of_frame`)
    - `metadata`: optional object for derived metrics (speed, team, etc.)
This structure minimizes per-frame object overhead, enabling efficient vectorized parsing and straightforward extension (new per-frame channels append as additional arrays).

#### `outputs/tracking/person_tracks.json`

- **Type**: object
- **Required keys**:
  - `version`: string (e.g., `"1.0.0"`)
  - `generated_at`: ISO 8601 timestamp string
  - `tracker`: object describing model info (e.g., `{"name": "yolov8", "weights": "yolov8x.pt"}`)
  - `games`: object keyed by `game_id`
- **Game entry** (`games[game_id]`):
  - `clips`: object keyed by `clip_id`
- **Clip entry** (`clips[clip_id]`):
  - `source`: relative path string to the clip media
  - `frames`: integer count of frames processed
  - `tracks`: object keyed by stringified `track_id`
- **Track entry** (`tracks[track_id]`):
  - `label`: string (default `"person"`)
  - `detections`: array of objects with required fields
    - `frame`: int frame index
    - `bbox`: array `[x_min, y_min, width, height]` (float pixels)
    - `confidence`: float
    - `keypoints`: optional array if the tracker outputs pose information
  - `summary`: object with derived metrics (e.g., `{"frames": 123, "avg_confidence": 0.87}`)

#### `outputs/tracking/player_assignments.json`

- **Type**: object
- **Required keys**:
  - `version`: string
  - `updated_at`: ISO 8601 timestamp string
  - `games`: object keyed by `game_id`
- **Game entry** (`games[game_id]`):
  - `clips`: object keyed by `clip_id`
- **Clip entry** (`clips[clip_id]`):
  - `source`: relative path string to the clip media
  - `players`: object mapping semantic roles to assignment data
    - Each role object requires `track_id` (string) and `assigned_by` (string username or `"unknown"`), optional `notes`, optional `confidence`
  - `last_reviewed`: ISO 8601 timestamp string

#### `outputs/court_annotations/games.json`

- **Type**: object
- **Required keys**:
  - `version`: string (e.g., `"1.0.0"`)
  - `updated_at`: ISO 8601 timestamp string
  - `games`: object keyed by `game_id` (e.g., `"game_01"`)
- **Game entry** (`games[game_id]`):
  - `annotator`: string identifier or `"unknown"`
  - `keypoints`: array of exactly 15 objects. Each keypoint object requires:
    - `index`: int (0–14)
    - `name`: string (must match canonical labels table)
    - `x`: float (pixel or normalized coordinate)
    - `y`: float
    - `confidence`: optional float 0–1
    - `visible`: optional boolean
  - `skeleton`: array of 9 two-element arrays referencing keypoint indices (`[[1,2], ...]`)
  - `reference_frame`: optional integer frame number used while annotating
  - `metadata`: optional object (e.g., court surface, tournament info, comments)
- Skeleton arrays must exactly match the defined topology to guarantee downstream consumers can draw or infer lines without recomputation.
- The manager should prevent save operations unless all 15 keypoints are present and indices/names align with the canonical ordering.

All schemas should tolerate forward-compatible optional keys but must fail validation when required fields are missing or malformed. Tooling should bump the `version` field when incompatible changes occur.

## Tool Responsibilities

### `tracknet/tools/player_tracker.py`
- Offer a CLI flag or subcommand (e.g., `--mode detect` / `--mode assign`) to choose between running detection and launching the assignment UI.
- **Detect mode**: enumerate clip directories under `data/tracknet/`, generate temporary videos from frame sequences, invoke `tracker.py` / YOLOv8 tracking for each clip, and aggregate results into `outputs/tracking/person_tracks.json` keyed by game, clip, and track ID metadata. Implement caching and a `--force` option to recompute.
- **Video generation**: before tracking, convert JPG frame sequences to temporary MP4 files using OpenCV VideoWriter at 30 FPS, ensuring proper frame ordering based on numeric filenames. Clean up temporary files after processing.
- **Assign mode**: load existing `person_tracks.json`, invoke the Matplotlib UI helper, and ensure the resulting selections are persisted to `outputs/tracking/player_assignments.json`.
- Both modes continuously sync JSON files so reruns pick up prior state without reprocessing finished clips.
- Provide helper functions to:
  - Load the latest `data/tracknet/annotations.json` (if present) so ball data and prior player annotations can pre-populate the UI state.
  - Merge updated player assignments back into an in-memory annotation structure and flush to disk atomically (write temp file + move) to avoid corruption.

### `tracknet/tools/utils/ui/player_identifier.py`
- Provide a Matplotlib-based interactive interface driven by `player_tracker.py` (not standalone executable).
- Render the current clip frame or summary alongside a multi-select checklist of track IDs with metadata (e.g., detection confidence, frame counts). Users can select/deselect IDs corresponding to each player role.
- Include navigation controls (←/→ buttons or keyboard bindings) to change clips and a save action that writes `outputs/tracking/player_assignments.json`.
- Keep UI state synchronized with both `person_tracks.json` and `player_assignments.json` so users can resume annotation mid-way.
- **Synchronization mechanics**:
  - On launch, hydrate UI state from `person_tracks.json` (for available tracks) and `player_assignments.json` (for existing selections). If `annotations.json` already contains player sequences, use them as authoritative and reflect them in the UI.
  - After each save, emit updated assignments via callbacks to `player_tracker.py`, which will redraw UI state (ensuring checklists stay current) and optionally update a shared in-memory cache of `annotations.json`.
  - Provide filesystem watchers or timestamp polling (configurable interval) so external edits to JSON files trigger a refresh prompt, preventing divergent copies.

### `tracknet/tools/court_annotation_manager.py`
- Provide a lightweight orchestration layer to launch the court UI and manage saved state.
- Maintain storage in `outputs/court_annotations/games.json`, ensuring all 10 games have entries.
- Offer validation helpers (e.g., check for four corner points) and allow re-entry for corrections.

### `tracknet/tools/utils/ui/court_annotation_form.py`
- Present the user with a visual or textual interface to enter court geometry (e.g., four corner coordinates, optional metadata like net height).
- If graphical interactions are out of scope, provide an iterative prompt asking for numeric inputs with live verification (e.g., preview computed bounds).
- Return a structured dictionary ready for serialization by `court_annotater.py`.

### `tracknet/tools/annotation_converter.py`
- Consume:
  - Ball CSVs per clip (`data/tracknet/<game>/<clip>/ball.csv`).
  - Player assignments (`outputs/tracking/player_assignments.json`).
  - Court annotations (`outputs/court_annotations/games.json`).
- Validate that required inputs exist; emit warnings for missing components but continue processing unaffected parts.
- Assemble the unified JSON using the data model above and write `data/tracknet/annotations.json`.
- Support optional CLI flags:
  - `--games` / `--clips` for filtering.
  - `--overwrite` to control behavior when the target JSON already exists.

## Execution Flow
1. **Tracking** — Run `uv run python tracknet/tools/player_tracker.py --mode detect`. The script walks all clips and updates `person_tracks.json`.
2. **Player Selection** — Run `uv run python tracknet/tools/player_tracker.py --mode assign`, which opens the Matplotlib UI for choosing player IDs per clip.
3. **Court Annotation** — Run `uv run python tracknet/tools/court_annotation_manager.py`, which iterates games and invokes the UI module to gather geometry.
4. **Conversion** — Run `uv run python tracknet/tools/annotation_converter.py` to emit the unified annotation file.

Each step can be repeated as data evolves. Downstream jobs (e.g., demos, training scripts) will rely on `annotations.json`.

## Error Handling & Logging
- Use structured logging (e.g., `logging` module) with INFO for progress, WARNING for missing data, ERROR for failures.
- Validate inputs early; for malformed CSV rows or JSON entries, log the context and skip rather than aborting the entire run.
- Include dry-run modes (`--dry-run`) where useful to inspect planned outputs without writing files.

## Testing Strategy
- **YOLOv8 load test**: verify the detector weights initialize successfully; mark as skipped with clear messaging when assets unavailable in CI.
- **Tracker integration test**: run `player_tracker.py --mode detect` against a fixture clip (or mocked inference) to confirm output structure matches expectations.
- **JSON schema tests**:
  - Validate `outputs/tracking/person_tracks.json` and `outputs/tracking/player_assignments.json` adhere to defined schemas (games → clips → tracks/assignments).
  - Validate `outputs/court_annotations/games.json` contains 15 keypoints (named + indexed) and the nine skeleton edges.
  - Validate `data/tracknet/annotations.json` across three pipeline stages: after ball import only, after adding player assignments, and after adding court data, ensuring ball `visibility` values are confined to `{0,1,2,3}` and `status` values to `{0,1,2,null}`.
- **CLI contract tests**: assert that `player_tracker.py` enforces valid modes, supports `--force`/`--dry-run` flags, and surfaces helpful errors for invalid arguments.
- **Caching/resume tests**: simulate partial detection runs to ensure subsequent executions skip completed clips yet allow regeneration when forced.
- **Converter idempotency**: confirm repeated runs with identical inputs leave `annotations.json` unchanged (byte-for-byte or via semantic comparison), protecting downstream reproducibility.
- **UI verification**: document a manual test plan ensuring the Matplotlib interface launches, navigates clips, saves assignments, and resumes correctly (not automated in pytest).

## Dependencies & Tooling
- Reuse existing project dependencies where possible. If `tracker.py` introduces third-party requirements, add them via `uv add`.
- Consider lightweight UI frameworks already present; if none, the UI modules can remain CLI-based prompts.
