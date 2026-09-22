#!/usr/bin/env python3
"""Static contract checks that do not require CUDA, ROS, or PyTorch."""

import argparse
import json
import re
import sys
from pathlib import Path


EXPECTED_CLASSES = [
    "truck", "car", "traffic_cone", "crane_tyre", "quay_crane", "agv",
    "crane_spreader", "tricyclist", "fence", "forklift", "pedestrian",
]


def load_json_comments(path):
    return json.loads(re.sub(r"/\*.*?\*/", "", Path(path).read_text(encoding="utf-8"), flags=re.S))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-calibration", default="yangluogang/camera_calibration.json")
    parser.add_argument("--runtime-calibration", default="yangluo_deploy/configs/calibration_runtime.json")
    parser.add_argument("--classes", default="yangluo_deploy/configs/classes.txt")
    parser.add_argument("--model-root")
    parser.add_argument("--allow-class-change", action="store_true")
    args = parser.parse_args()

    errors = []
    raw = load_json_comments(args.camera_calibration)
    cameras = {str(item["desc"]): item for item in raw["camera_params"]}
    for camera_id in ("0", "5"):
        if camera_id not in cameras:
            errors.append(f"missing camera calibration {camera_id}")
        elif len(cameras[camera_id].get("distortion", [])) != 8:
            errors.append(f"camera {camera_id} does not have 8 distortion coefficients")

    classes = [line.strip() for line in Path(args.classes).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes or len(classes) != len(set(classes)):
        errors.append("class list must be non-empty and contain unique names")
    if not args.allow_class_change and classes != EXPECTED_CLASSES:
        errors.append(f"class order mismatch: {classes}")

    runtime_path = Path(args.runtime_calibration)
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if list(runtime.get("cameras", {}).keys()) != ["front", "rear"]:
            errors.append("runtime calibration camera order is not front,rear")

    if args.model_root:
        required = ["camera.backbone.onnx", "camera.vtransform.onnx", "lidar.backbone.xyz.onnx", "fuser.onnx", "head.bbox.onnx"]
        for name in required:
            path = Path(args.model_root) / name
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty model artifact: {path}")

    if errors:
        print("FAIL")
        for error in errors:
            print("-", error)
        return 1
    print(f"PASS cameras=0,5 classes={len(classes)} coordinate_frame=base_link point_layout=xyzi0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
