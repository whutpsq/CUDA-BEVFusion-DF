"""Validate converted Yangluo BEVFusion info files and geometry."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import pickle

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="data_yangluo/yangluo_bevfusion"
    )
    parser.add_argument("--info-prefix", default="yangluo")
    parser.add_argument(
        "--containment-samples",
        type=int,
        default=6,
        help="Number of leading samples checked against points-in-box counts.",
    )
    return parser.parse_args()


def load_info(path: Path) -> dict:
    with path.open("rb") as stream:
        return pickle.load(stream)


def points_in_boxes_count(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    counts = []
    for box in boxes:
        delta = points[:, :3] - box[:3]
        cosine, sine = np.cos(box[6]), np.sin(box[6])
        local_x = delta[:, 0] * cosine - delta[:, 1] * sine
        local_y = delta[:, 0] * sine + delta[:, 1] * cosine
        inside = (
            (np.abs(local_x) <= box[3] * 0.5)
            & (np.abs(local_y) <= box[4] * 0.5)
            & (delta[:, 2] >= 0.0)
            & (delta[:, 2] <= box[5])
        )
        counts.append(int(inside.sum()))
    return np.asarray(counts, dtype=np.int64)


def validate_sample(info: dict) -> None:
    point_path = Path(info["lidar_path"])
    if not point_path.is_file():
        raise FileNotFoundError(point_path)
    if point_path.stat().st_size % (5 * 4) != 0:
        raise ValueError(f"Invalid float32 Nx5 point file size: {point_path}")
    boxes = np.asarray(info["gt_boxes"])
    names = np.asarray(info["gt_names"])
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError(f"gt_boxes must be Nx7: {info['token']} {boxes.shape}")
    if not (len(boxes) == len(names) == len(info["valid_flag"])):
        raise ValueError(f"Annotation array length mismatch: {info['token']}")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
        raise ValueError(f"Invalid box values: {info['token']}")
    if list(info["cams"]) != ["CAM_FRONT", "CAM_BACK"]:
        raise ValueError(f"Unexpected camera order: {list(info['cams'])}")
    for camera in info["cams"].values():
        if not Path(camera["data_path"]).is_file():
            raise FileNotFoundError(camera["data_path"])
        if np.asarray(camera["cam_intrinsic"]).shape != (3, 3):
            raise ValueError(f"Invalid intrinsic: {info['token']}")
        if np.asarray(camera["distortion"]).size not in (4, 5, 8):
            raise ValueError(f"Invalid distortion: {info['token']}")


def validate_camera_projection(info: dict) -> dict:
    """Check the exact camera-to-model matrices consumed by the dataset."""
    points = np.fromfile(info["lidar_path"], dtype=np.float32).reshape(-1, 5)
    homogeneous = np.column_stack([points[:, :3], np.ones(len(points))])
    counts = {}
    for name, camera in info["cams"].items():
        camera_to_model = np.eye(4, dtype=np.float64)
        camera_to_model[:3, :3] = camera["sensor2lidar_rotation"]
        camera_to_model[:3, 3] = camera["sensor2lidar_translation"]
        model_to_camera = np.linalg.inv(camera_to_model)
        camera_points = (model_to_camera @ homogeneous.T).T[:, :3]
        projected = (np.asarray(camera["cam_intrinsic"]) @ camera_points.T).T
        pixels = projected[:, :2] / projected[:, 2:3]
        visible = (
            (camera_points[:, 2] > 0)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < camera["width"])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < camera["height"])
        )
        counts[name] = int(visible.sum())
        if counts[name] < 100:
            raise ValueError(
                f"Camera projection closure failed for {name}: {counts[name]} points"
            )
    return counts


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    datasets = {}
    for split in ("train", "val"):
        path = data_dir / f"{args.info_prefix}_infos_{split}.pkl"
        datasets[split] = load_info(path)
        if not datasets[split]["infos"]:
            raise ValueError(f"Empty {split} split: {path}")
        for info in datasets[split]["infos"]:
            validate_sample(info)

    train_clips = {x["location"] for x in datasets["train"]["infos"]}
    val_clips = {x["location"] for x in datasets["val"]["infos"]}
    if datasets["train"]["metadata"].get("split_by") == "clip":
        overlap = train_clips & val_clips
        if overlap:
            raise ValueError(f"Clip leakage between train and val: {sorted(overlap)}")

    checked = 0
    observed_total = 0
    expected_total = 0
    for info in datasets["train"]["infos"] + datasets["val"]["infos"]:
        if checked >= args.containment_samples:
            break
        points = np.fromfile(info["lidar_path"], dtype=np.float32).reshape(-1, 5)
        observed = points_in_boxes_count(points, np.asarray(info["gt_boxes"]))
        expected = np.asarray(info["num_lidar_pts"], dtype=np.int64)
        observed_total += int(observed.sum())
        expected_total += int(expected.sum())
        checked += 1
    if checked and expected_total > 0:
        ratio = observed_total / expected_total
        if not 0.85 <= ratio <= 1.15:
            raise ValueError(
                "Point/box containment disagrees with source pointsNum: "
                f"observed={observed_total} expected={expected_total} ratio={ratio:.4f}"
            )
    else:
        ratio = float("nan")

    label_counts = Counter(
        str(name)
        for split in datasets.values()
        for info in split["infos"]
        for name in info["gt_names"]
    )
    projection_counts = validate_camera_projection(datasets["train"]["infos"][0])
    print(
        f"PASS train={len(datasets['train']['infos'])} "
        f"val={len(datasets['val']['infos'])}"
    )
    print(f"train_clips={sorted(train_clips)}")
    print(f"val_clips={sorted(val_clips)}")
    print(f"labels={dict(sorted(label_counts.items()))}")
    print(
        f"containment_samples={checked} observed={observed_total} "
        f"expected={expected_total} ratio={ratio:.4f}"
    )
    print(f"projection_in_image={projection_counts}")


if __name__ == "__main__":
    main()
