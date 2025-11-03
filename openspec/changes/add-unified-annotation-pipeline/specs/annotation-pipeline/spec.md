## ADDED Requirements
### Requirement: Consolidate Clip Annotations
The system MUST convert per-clip ball CSV annotations into a single `data/tracknet/annotations.json` file that captures ball visibility/status and player bounding boxes per frame.

#### Scenario: Convert Ball CSVs
- **GIVEN** `data/tracknet/<game>/<clip>/Label.csv` annotations exist
- **WHEN** I run `uv run python tracknet/tools/annotation_converter.py`
- **THEN** the tool reads every clip CSV and writes ball coordinates (with integer `visibility` and `status` metadata per `docs/data/tracknet.md`) into `data/tracknet/annotations.json` grouped by game and clip.
- **AND** the resulting structure organizes per-frame data for efficient loading while remaining extensible.

### Requirement: Generate Videos from Frame Sequences
The system MUST convert frame sequences to temporary video files for person tracking processing.

#### Scenario: Create Video from Frames
- **GIVEN** frame images exist in `data/tracknet/<game>/<clip>/` as sequential JPG files
- **WHEN** the player tracker needs video input for YOLOv8 processing
- **THEN** the tool generates a temporary MP4 video from the frame sequence at appropriate FPS
- **AND** the temporary video is used for tracking and then cleaned up after processing
- **AND** frame order and timing are preserved based on numeric filename sequencing

### Requirement: Capture Player Track Assignments
The system MUST let users detect candidate person tracks and select player track IDs per clip, persisting them for integration.

#### Scenario: Generate Person Tracks
- **GIVEN** frame sequences exist under `data/tracknet/<game>/<clip>/` as JPG images
- **WHEN** I run `uv run python tracknet/tools/player_tracker.py --mode detect`
- **THEN** the tool generates temporary videos from frame sequences for YOLOv8 tracking
- **AND** YOLOv8-based tracking results for every clip are saved to `outputs/tracking/person_tracks.json`.

#### Scenario: Persist Player IDs
- **GIVEN** `outputs/tracking/person_tracks.json` is available
- **WHEN** I launch `uv run python tracknet/tools/player_tracker.py --mode assign`
- **THEN** an interactive Matplotlib UI (backed by `tracknet/tools/utils/ui/player_identifier.py`) lets me pick player IDs per clip with multi-select checklists, save progress, and navigate clips
- AND selections are written to `outputs/tracking/player_assignments.json`, remaining in sync so that re-opening the UI resumes prior work
- AND the chosen IDs are included in `data/tracknet/annotations.json` as per-frame player bounding boxes (plus optional visibility/status channels) linked to their track identifiers.

### Requirement: Record Court Geometry Per Game
The system MUST collect static court annotations per game and merge them into the unified annotation payload.

#### Scenario: Add Court Annotation
- **GIVEN** a user annotates court geometry for game `N` with `tracknet/tools/utils/ui/court_annotation_form.py`
- **WHEN** the annotation is saved
- **THEN** `outputs/court_annotations/games.json` stores the geometry for game `N` with version metadata, the 15 labeled keypoints, and the predefined skeleton edges
- **AND** running `tracknet/tools/court_annotation_manager.py` followed by `tracknet/tools/annotation_converter.py` integrates the geometry into `data/tracknet/annotations.json`.
