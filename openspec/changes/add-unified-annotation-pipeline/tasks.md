## 1. Research
- [ ] 1.1 Review existing dataset structure under `data/tracknet/` (clips, games, CSV layouts).
- [ ] 1.2 Confirm dependencies for tracking (`tracker.py`) and UI stacks already available in the project.

## 2. Tooling Implementation
- [ ] 2.1 Implement `tracknet/tools/annotation_converter.py` to merge ball CSVs, player selections, and court annotations into `data/tracknet/annotations.json`.
- [ ] 2.2 Implement `tracknet/tools/player_tracker.py` with CLI modes for YOLOv8 detection (`--mode detect`) and player assignment UI (`--mode assign`), persisting `outputs/tracking/person_tracks.json` and `outputs/tracking/player_assignments.json`.
- [ ] 2.3 Implement `tracknet/tools/utils/ui/player_identifier.py` as a Matplotlib UI component used by `player_tracker.py` for multi-select player ID assignment with clip navigation and save/resume support.
- [ ] 2.4 Implement `tracknet/tools/court_annotation_manager.py` to manage loading/saving per-game court annotations via UI input.
- [ ] 2.5 Implement `tracknet/tools/utils/ui/court_annotation_form.py` to collect per-game court geometry and save to `outputs/court_annotations/games.json`.

## 3. Validation
- [ ] 3.1 Add automated tests confirming YOLOv8 model loading succeeds (skip or mark xfail when weights unavailable).
- [ ] 3.2 Add integration test covering `tracker.py` invocation to verify expected tracking structure.
- [ ] 3.3 Add schema validation tests for `outputs/tracking/person_tracks.json` and `outputs/tracking/player_assignments.json`.
- [ ] 3.4 Add schema validation test for `outputs/court_annotations/games.json`, including required keypoints and skeleton.
- [ ] 3.5 Add multi-stage validation test for `data/tracknet/annotations.json` covering ball-only, ball+player, and full (ball+player+court) scenarios, asserting ball `visibility` ∈ {0,1,2,3} and `status` ∈ {0,1,2,null}.
- [ ] 3.6 Document manual UI verification steps ensuring the Matplotlib interface launches and receives interactions correctly.
- [ ] 3.7 Add CLI-level tests to confirm `tracknet/tools/player_tracker.py` handles `--mode`, filtering, `--force`, and dry-run switches as expected.
- [ ] 3.8 Add regression test ensuring detection caching prevents unnecessary recomputation and resumes correctly after partial runs.
- [ ] 3.9 Add idempotency test guaranteeing re-running `tracknet/tools/annotation_converter.py` with unchanged inputs does not alter existing output (apart from timestamp/log metadata).

## 4. Docs & Cleanup
- [ ] 4.1 Update documentation (`docs/dataset.md` or new doc) with instructions for the new annotation workflow.
- [ ] 4.2 Ensure all new files include Google-style docstrings and follow coding standards.
- [ ] 4.3 Run `uv run pytest` (or relevant tests) and formatters if available.
