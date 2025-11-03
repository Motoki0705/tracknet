## Why
- Current ball annotations live in clip-scoped CSV files, making downstream training cumbersome.
- Player and court annotations do not exist, yet modeling work needs unified metadata for ball, player identities, and court geometry.
- Manual aggregation is error-prone; we need a reproducible pipeline and interactive tooling.

## What Changes
- Build tooling to ingest per-clip CSV ball annotations and emit a consolidated `data/tracknet/annotation.json`.
- Add automated person tracking plus a lightweight UI so users can label player track IDs per clip.
- Add video generation capability to convert frame sequences to temporary videos for YOLOv8 processing.
- Add court annotation UI at the game level so users can supply static court geometry per game.
- Integrate these sources into a single JSON payload via a converter tool.

## Impact
- Enables consistent annotations for model training and evaluation.
- Introduces new tooling; no breaking change to existing demos, but requires new dependencies validated during implementation.
- Manual labeling steps (player IDs, court geometry) become guided, lowering chance of inconsistent data.

## Open Questions
- Do we need to support partial annotation runs (e.g., skip already processed clips/games)?
- Is there an existing schema consumers expect for `annotation.json`, or can we define a new one?
