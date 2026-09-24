#!/usr/bin/env python3
"""Yangluo visualization entrypoint using its dataset-level calibration."""

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


# Executing this file directly places yangluo_adapter, rather than the repository
# root, first on sys.path.  Add the root so the existing visualization module is
# imported without copying or modifying it.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import visualize as base_visualize  # noqa: E402


ORIGINAL_LIDAR_BOX_CLASS = base_visualize.LiDARInstance3DBoxes


def load_json_with_comments(path: Path):
    text = path.read_text(encoding="utf-8-sig")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"(^|\s)//.*?$", r"\1", text, flags=re.MULTILINE)
    return json.loads(text)


@lru_cache(maxsize=8)
def load_calibration_file(path: str):
    raw = load_json_with_comments(Path(path))
    return {str(item["desc"]): item for item in raw["camera_params"]}


def load_yangluo_raw_camera_model(image_path):
    """Return raw K+D from the Yangluo dataset-level calibration JSON."""
    image_path = Path(image_path).resolve()
    camera_name = image_path.parent.name
    if not camera_name.startswith("cam"):
        raise ValueError(f"cannot infer Yangluo camera from {image_path}")
    camera_map = {"cam5": "0", "cam10": "10"}
    if camera_name not in camera_map:
        raise ValueError(f"Unsupported Yangluo camera directory: {camera_name}")
    camera_desc = camera_map[camera_name]

    calibration_path = None
    for parent in image_path.parents:
        candidate = parent / "config" / "camera_calibration.json"
        if candidate.is_file():
            calibration_path = candidate
            break
    if calibration_path is None:
        raise FileNotFoundError(
            "cannot find config/camera_calibration.json above "
            f"Yangluo image: {image_path}"
        )

    by_desc = load_calibration_file(str(calibration_path))
    if camera_desc not in by_desc:
        raise KeyError(
            f"camera desc={camera_desc} is absent from {calibration_path}"
        )
    camera = by_desc[camera_desc]
    intrinsic = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera["distortion"], dtype=np.float64).reshape(-1)
    if distortion.size not in (4, 5, 8):
        raise ValueError(
            f"camera {camera_name} must have 4, 5, or 8 distortion values, got "
            f"{distortion.shape}"
        )
    return intrinsic, distortion


def build_yangluo_lidar_boxes(tensor, *args, **kwargs):
    """Undo tools/visualize.py's legacy center-to-bottom Z subtraction.

    Yangluo PKLs are loaded by NuScenesDataset with origin=(0.5, 0.5, 0), so
    ``gt_bboxes_3d.tensor[..., 2]`` is already the bottom-face Z.  The shared
    visualizer subtracts height/2 before constructing LiDAR boxes, which would
    move these boxes down by half their height.  Add that amount back only in
    this adapter entrypoint; the shared visualizer remains unchanged.
    """
    if hasattr(tensor, "clone"):
        corrected = tensor.clone()
    else:
        corrected = np.array(tensor, copy=True)
    corrected[..., 2] += corrected[..., 5] * 0.5
    return ORIGINAL_LIDAR_BOX_CLASS(corrected, *args, **kwargs)


def main():
    base_visualize.load_raw_camera_model = load_yangluo_raw_camera_model
    base_visualize.LiDARInstance3DBoxes = build_yangluo_lidar_boxes
    base_visualize.main()


if __name__ == "__main__":
    main()
