#!/usr/bin/env python3
"""Yangluo-only prediction visualizer without changing tools/visualize.py.

The shared passenger-car visualizer expects one calibration JSON beside every
clip.  Yangluo stores all camera parameters in one dataset-level
``config/camera_calibration.json``.  This entrypoint replaces only that lookup
for the current process and keeps the original passenger-car entrypoint intact.
"""

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import visualize as base_visualize  # noqa: E402


ORIGINAL_LIDAR_BOX_CLASS = base_visualize.LiDARInstance3DBoxes
EXPLICIT_CALIBRATION = None
REPORTED_CALIBRATION = set()


def pop_wrapper_arguments(argv):
    """Remove the Yangluo-only option before the base parser sees argv."""
    global EXPLICIT_CALIBRATION
    retained = [argv[0]]
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == "--camera-calibration":
            if index + 1 >= len(argv):
                raise ValueError("--camera-calibration requires a JSON path")
            EXPLICIT_CALIBRATION = Path(argv[index + 1]).expanduser().resolve()
            index += 2
            continue
        if argument.startswith("--camera-calibration="):
            EXPLICIT_CALIBRATION = Path(argument.split("=", 1)[1]).expanduser().resolve()
            index += 1
            continue
        retained.append(argument)
        index += 1
    argv[:] = retained


def load_json_with_comments(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"(^|\s)//.*?$", r"\1", text, flags=re.MULTILINE)
    return json.loads(text)


@lru_cache(maxsize=8)
def load_calibration_file(path):
    calibration_path = Path(path)
    raw = load_json_with_comments(calibration_path)
    if "camera_params" not in raw:
        raise KeyError(f"camera_params is absent from {calibration_path}")
    return {str(item["desc"]): item for item in raw["camera_params"]}


def find_calibration(image_path):
    if EXPLICIT_CALIBRATION is not None:
        if not EXPLICIT_CALIBRATION.is_file():
            raise FileNotFoundError(EXPLICIT_CALIBRATION)
        return EXPLICIT_CALIBRATION

    for parent in image_path.parents:
        candidate = parent / "config" / "camera_calibration.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot find config/camera_calibration.json above Yangluo image "
        f"{image_path}. Pass --camera-calibration /absolute/path/camera_calibration.json."
    )


CAMERA_CALIBRATION_MAP = {"cam5": "0", "cam10": "10"}


def load_yangluo_raw_camera_model(image_path):
    """Return raw K and OpenCV D for mapped cam5/cam10 Yangluo images."""
    image_path = Path(image_path).expanduser().resolve()
    camera_name = image_path.parent.name
    if not camera_name.startswith("cam"):
        raise ValueError(f"Cannot infer Yangluo camera name from {image_path}")
    if camera_name not in CAMERA_CALIBRATION_MAP:
        raise ValueError(f"Unsupported Yangluo camera directory: {camera_name}")
    camera_desc = CAMERA_CALIBRATION_MAP[camera_name]

    calibration_path = find_calibration(image_path)
    by_desc = load_calibration_file(str(calibration_path))
    if camera_desc not in by_desc:
        raise KeyError(f"camera desc={camera_desc} is absent from {calibration_path}")

    camera = by_desc[camera_desc]
    intrinsic = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera["distortion"], dtype=np.float64).reshape(-1)
    if distortion.size not in (4, 5, 8):
        raise ValueError(
            f"camera {camera_name} must have 4, 5, or 8 distortion values, got {distortion.shape}"
        )

    report_key = (str(calibration_path), camera_desc)
    if report_key not in REPORTED_CALIBRATION:
        print(
            "YANGLUO_VIS_CALIBRATION_OK "
            f"camera={camera_name} file={calibration_path}",
            flush=True,
        )
        REPORTED_CALIBRATION.add(report_key)
    return intrinsic, distortion


def build_yangluo_lidar_boxes(tensor, *args, **kwargs):
    """Keep Yangluo bottom-face Z after the base visualizer's legacy shift.

    Yangluo PKLs are loaded with origin=(0.5, 0.5, 0), so tensor[..., 2] is
    already bottom Z.  The shared visualizer subtracts half the height before
    constructing LiDAR boxes.  Adding it here cancels only that legacy shift.
    """
    if hasattr(tensor, "clone"):
        corrected = tensor.clone()
    else:
        corrected = np.array(tensor, copy=True)
    corrected[..., 2] += corrected[..., 5] * 0.5
    return ORIGINAL_LIDAR_BOX_CLASS(corrected, *args, **kwargs)


def main():
    pop_wrapper_arguments(sys.argv)
    base_visualize.load_raw_camera_model = load_yangluo_raw_camera_model
    base_visualize.LiDARInstance3DBoxes = build_yangluo_lidar_boxes
    base_visualize.main()


if __name__ == "__main__":
    main()
