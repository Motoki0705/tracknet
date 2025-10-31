# TrackNet Tools

Tools for extending the TrackNet dataset with player tracking and court keypoint annotations.

## Documentation

For detailed specifications and usage instructions, see:
- **[docs/tools/index.md](../../docs/tools/index.md)** - Main documentation
- **[docs/tools/visualize_dataset.md](../../docs/tools/visualize_dataset.md)** - Dataset visualization tool
- **[docs/tools/coco_converter.md](../../docs/tools/coco_converter.md)** - COCO format converter
- **[docs/tools/batch_player_tracker.md](../../docs/tools/batch_player_tracker.md)** - Batch player tracking tool
- **[docs/tools/court_annotator.md](../../docs/tools/court_annotator.md)** - Court annotation tool

## Quick Start

```bash
# Install dependencies
uv add ultralytics opencv-python matplotlib pandas pillow

# Visualize existing dataset
uv run python -m tracknet.tools.visualize_dataset --config configs/dataset/tracknet.yaml

# Convert CSV to COCO format
uv run python -m tracknet.tools.coco_converter --dataset-root data/tracknet --output annotations.json

# Batch track all players and extract IDs
uv run python -m tracknet.tools.batch_player_tracker --dataset-root data/tracknet --output-dir outputs/batch_tracking --extract-ids --skip-ui

# Annotate court keypoints
uv run python -m tracknet.tools.court_annotator --video path/to/game_video.mp4 --output court.json
```

## File Structure

```
tracknet/tools/
├── README.md                     # This file
├── visualize_dataset.py          # Dataset visualization tool
├── coco_converter.py             # COCO format converter
├── batch_player_tracker.py       # Batch player tracking tool
├── court_annotator.py            # Court annotation tool
├── tracknet_frame_config.yaml    # Configuration file
└── utils/
    └── ui/
        ├── __init__.py
        ├── player_selector.py    # Player selection UI
        └── court_selector.py     # Court annotation UI
```
