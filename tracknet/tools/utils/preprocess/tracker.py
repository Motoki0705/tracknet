from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


def get_video_lwh(video_path: str) -> tuple[int, int, int]:
    """Get video length, width, and height.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (length_frames, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return length, width, height


def frame_id_to_mask(frame_ids: torch.Tensor, total_frames: int) -> torch.Tensor:
    """Convert frame IDs to binary mask.

    Args:
        frame_ids: Tensor of frame indices
        total_frames: Total number of frames

    Returns:
        Binary mask tensor of shape (total_frames,)
    """
    mask = torch.zeros(total_frames, dtype=torch.bool)
    mask[frame_ids] = True
    return mask


def get_frame_id_list_from_mask(mask: torch.Tensor) -> list[list[int]]:
    """Get consecutive frame ID lists from mask.

    Args:
        mask: Binary mask where False indicates missing frames

    Returns:
        List of lists containing consecutive frame IDs
    """
    missing_indices = torch.where(~mask)[0].tolist()

    if not missing_indices:
        return []

    # Group consecutive indices
    result = []
    current_group = [missing_indices[0]]

    for idx in missing_indices[1:]:
        if idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            result.append(current_group)
            current_group = [idx]

    result.append(current_group)
    return result


def rearrange_by_mask(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Rearrange data according to mask, filling missing with zeros.

    Args:
        data: Input data tensor
        mask: Binary mask for arrangement

    Returns:
        Rearranged tensor with zeros for missing positions
    """
    result = torch.zeros(mask.shape[0], data.shape[1])
    result[mask] = data
    return result


def linear_interpolate_frame_ids(
    data: torch.Tensor, missing_frame_groups: list[list[int]]
) -> torch.Tensor:
    """Linear interpolate missing frames.

    Args:
        data: Tensor with some zero-filled frames
        missing_frame_groups: Groups of consecutive missing frame indices

    Returns:
        Interpolated tensor
    """
    for group in missing_frame_groups:
        for frame_idx in group:
            if frame_idx == 0 or frame_idx == len(data) - 1:
                continue

            # Find previous and next non-zero frames
            prev_idx = frame_idx - 1
            while prev_idx >= 0 and torch.all(data[prev_idx] == 0):
                prev_idx -= 1

            next_idx = frame_idx + 1
            while next_idx < len(data) and torch.all(data[next_idx] == 0):
                next_idx += 1

            if prev_idx >= 0 and next_idx < len(data):
                # Linear interpolation
                alpha = (frame_idx - prev_idx) / (next_idx - prev_idx)
                data[frame_idx] = (1 - alpha) * data[prev_idx] + alpha * data[next_idx]

    return data


def moving_average_smooth(
    data: torch.Tensor, window_size: int, dim: int = 0
) -> torch.Tensor:
    """Apply moving average smoothing.

    Args:
        data: Input tensor
        window_size: Size of moving window
        dim: Dimension along which to smooth

    Returns:
        Smoothed tensor
    """
    if window_size >= data.shape[dim]:
        return data

    kernel = torch.ones(window_size) / window_size
    if dim == 0:
        smoothed = torch.zeros_like(data)
        for i in range(data.shape[1]):
            smoothed[:, i] = torch.conv1d(
                data[:, i].unsqueeze(0).unsqueeze(0).float(),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=window_size // 2,
            ).squeeze()
        return smoothed
    else:
        raise NotImplementedError(f"Smoothing along dim {dim} not implemented")


class Tracker:
    def __init__(self) -> None:
        # https://docs.ultralytics.com/modes/predict/
        model_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "checkpoints"
            / "yolo"
            / "yolov8x.pt"
        )
        self.yolo = YOLO(str(model_path))

    def track(self, video_path):
        track_history = []
        cfg = {
            "device": "cuda",
            "conf": 0.5,  # default 0.25, wham 0.5
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
        }
        results = self.yolo.track(video_path, **cfg)
        # frame-by-frame tracking
        track_history = []
        for result in tqdm(
            results, total=get_video_lwh(video_path)[0], desc="YoloV8 Tracking"
        ):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()  # (N)
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                result_frame = [
                    {"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]}
                    for i in range(len(track_ids))
                ]
            else:
                result_frame = []
            track_history.append(result_frame)

        return track_history

    @staticmethod
    def sort_track_length(track_history, video_path):
        """This handles the track history from YOLO tracker."""
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        # parse to {det_id : [frame_id]}
        for frame_id, frame in enumerate(track_history):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
        for k, v in id_to_bbx_xyxys.items():
            id_to_bbx_xyxys[k] = np.array(v)

        # Sort by length of each track (max to min)
        id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
        dict(sorted(id_length.items(), key=lambda item: item[1], reverse=True))

        # Sort by area sum (max to min)
        id_area_sum = {}
        video_length, video_width, video_height = get_video_lwh(video_path)
        for k, v in id_to_bbx_xyxys.items():
            bbx_wh = v[:, 2:] - v[:, :2]
            id_area_sum[k] = (bbx_wh[:, 0] * bbx_wh[:, 1] / video_width / video_height).sum()
        id2area_sum = dict(
            sorted(id_area_sum.items(), key=lambda item: item[1], reverse=True)
        )
        id_sorted = list(id2area_sum.keys())

        return id_to_frame_ids, id_to_bbx_xyxys, id_sorted

    def get_one_track(self, video_path):
        # track
        track_history = self.track(video_path)

        # parse track_history & use top1 track
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(
            track_history, video_path
        )
        track_id = id_sorted[0]
        frame_ids = torch.tensor(id_to_frame_ids[track_id])  # (N,)
        bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])  # (N, 4)

        # interpolate missing frames
        mask = frame_id_to_mask(frame_ids, get_video_lwh(video_path)[0])
        bbx_xyxy_one_track = rearrange_by_mask(
            bbx_xyxys, mask
        )  # (F, 4), missing filled with 0
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)  # list of list
        bbx_xyxy_one_track = linear_interpolate_frame_ids(
            bbx_xyxy_one_track, missing_frame_id_list
        )
        assert (bbx_xyxy_one_track.sum(1) != 0).all()

        bbx_xyxy_one_track = moving_average_smooth(
            bbx_xyxy_one_track, window_size=5, dim=0
        )
        bbx_xyxy_one_track = moving_average_smooth(
            bbx_xyxy_one_track, window_size=5, dim=0
        )

        return bbx_xyxy_one_track
