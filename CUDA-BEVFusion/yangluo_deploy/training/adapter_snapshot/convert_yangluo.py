"""Convert the Yangluo port-truck dataset to BEVFusion info files.

The source point clouds and 3D labels use RFU (x right, y front, z up).
This adapter writes FLU point clouds (x front, y left, z up), matching the
LiDAR coordinate convention used by this BEVFusion checkout.  It does not
modify or import the existing passenger-car converter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pickle
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np


CLASS_MAP = {
    "truck": "truck",
    "car": "car",
    "trafficCone": "traffic_cone",
    "Crane_Tyre": "crane_tyre",
    "Crane_quay": "quay_crane",
    "AGV": "agv",
    "Crane_spreader": "crane_spreader",
    "tricyclist": "tricyclist",
    "fence": "fence",
    "forkLift": "forklift",
    "person": "pedestrian",
}

OBJECT_CLASSES = tuple(dict.fromkeys(CLASS_MAP.values()))
CAMERA_MAP = (("CAM_FRONT", "cam5", "0"), ("CAM_BACK", "cam10", "10"))

# Column-vector transforms.  The supplied camera extrinsics map RFU vehicle
# coordinates to OpenCV camera coordinates.
RFU_TO_FLU = np.array(
    [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
FLU_TO_RFU = np.linalg.inv(RFU_TO_FLU)
IDENTITY = np.eye(4, dtype=np.float64)


def portable_path(path: Path) -> str:
    """Prefer a repo-relative POSIX path so PKLs work in Windows and Linux."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-path",
        default="data_yangluo/baidu_dy_1_20260723_355p",
        help="Directory containing data/, annotation/, and config/.",
    )
    parser.add_argument(
        "--out-dir",
        default="data_yangluo/yangluo_bevfusion",
        help="Output directory for FLU .bin files and info .pkl files.",
    )
    parser.add_argument("--info-prefix", default="yangluo")
    parser.add_argument("--train-ratio", type=float, default=0.67)
    parser.add_argument(
        "--split-by",
        choices=("clip", "sample"),
        default="clip",
        help="Clip split avoids adjacent frames leaking into validation.",
    )
    parser.add_argument(
        "--skip-points",
        action="store_true",
        help="Reuse already converted .bin files after an interrupted run.",
    )
    return parser.parse_args()


def load_json_with_comments(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return json.loads(text)


def matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    """Return a normalized [w, x, y, z] quaternion."""
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s,
                      (r[0, 2] - r[2, 0]) / s,
                      (r[1, 0] - r[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s,
                          (r[0, 1] + r[1, 0]) / s,
                          (r[0, 2] + r[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array([(r[0, 2] - r[2, 0]) / s,
                          (r[0, 1] + r[1, 0]) / s, 0.25 * s,
                          (r[1, 2] + r[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array([(r[1, 0] - r[0, 1]) / s,
                          (r[0, 2] + r[2, 0]) / s,
                          (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q.tolist()


def load_calibration(root: Path) -> Dict[str, dict]:
    raw = load_json_with_comments(root / "config" / "camera_calibration.json")
    by_desc = {str(item["desc"]): item for item in raw["camera_params"]}
    calibration = {}
    for bev_name, source_name, desc in CAMERA_MAP:
        item = by_desc[desc]
        rfu_to_camera = np.asarray(item["extrinsics"], dtype=np.float64).reshape(4, 4)
        flu_to_camera = rfu_to_camera @ FLU_TO_RFU
        camera_to_flu = np.linalg.inv(flu_to_camera)
        calibration[source_name] = {
            "bev_name": bev_name,
            "intrinsic": np.asarray(item["intrinsics"], dtype=np.float32).reshape(3, 3),
            "distortion": np.asarray(item["distortion"], dtype=np.float32),
            "width": int(item["width"]),
            "height": int(item["height"]),
            "flu_to_camera": flu_to_camera,
            "camera_to_flu": camera_to_flu,
        }
    return calibration


def read_ascii_pcd_xyzi(path: Path) -> np.ndarray:
    header = {}
    header_lines = 0
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD has no DATA line: {path}")
            header_lines += 1
            decoded = line.decode("ascii").strip()
            if decoded and not decoded.startswith("#"):
                key, *values = decoded.split()
                header[key] = values
            if decoded.startswith("DATA"):
                break
    if header.get("DATA") != ["ascii"]:
        raise ValueError(f"Only DATA ascii is supported, got {header.get('DATA')}: {path}")
    fields = header["FIELDS"]
    required = ("x", "y", "z", "intensity")
    usecols = tuple(fields.index(name) for name in required)
    values = np.loadtxt(path, skiprows=header_lines, usecols=usecols, dtype=np.float32)
    values = np.atleast_2d(values)
    points = np.zeros((len(values), 5), dtype=np.float32)
    # RFU (right, front, up) -> FLU (front, left, up).
    points[:, 0] = values[:, 1]
    points[:, 1] = -values[:, 0]
    points[:, 2] = values[:, 2]
    points[:, 3] = values[:, 3]
    points[:, 4] = 0.0
    return points[np.isfinite(points).all(axis=1)]


def convert_points(pcd_path: Path, output_path: Path, skip: bool) -> None:
    if skip and output_path.is_file() and output_path.stat().st_size > 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = read_ascii_pcd_xyzi(pcd_path)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}")
    points.tofile(temporary)
    os.replace(temporary, output_path)


def convert_annotations(annotation: dict) -> Tuple[np.ndarray, ...]:
    boxes, names, velocities, point_counts = [], [], [], []
    unknown = set()
    for obj in annotation.get("dataList", []):
        raw_name = obj.get("label")
        name = CLASS_MAP.get(raw_name)
        if name is None:
            unknown.add(raw_name)
            continue
        center = obj["center"]
        dims = obj["dimensions"]
        length = float(dims["length"])
        width = float(dims["width"])
        height = float(dims["height"])
        raw_yaw = float(obj.get("rotation", {}).get("z", 0.0))
        # RFU center -> FLU center. The source z is the gravity center, whereas
        # this checkout constructs LiDAR boxes with a bottom-center origin.
        x_flu = float(center["y"])
        y_flu = -float(center["x"])
        z_bottom = float(center["z"]) - height * 0.5
        # NuScenes-style info uses [width, length, height].  Combined with the
        # clockwise LiDARInstance3DBoxes yaw convention, RFU->FLU reduces to
        # internal_yaw = -raw_yaw (equivalent modulo pi for an unoriented box).
        boxes.append([x_flu, y_flu, z_bottom, width, length, height, -raw_yaw])
        names.append(name)
        velocities.append([0.0, 0.0])
        point_counts.append(max(0, int(obj.get("pointsNum", 0))))
    if unknown:
        raise ValueError(f"Unmapped Yangluo labels: {sorted(unknown)}")
    counts = np.asarray(point_counts, dtype=np.int32)
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
        np.asarray(names),
        np.asarray(velocities, dtype=np.float32).reshape(-1, 2),
        counts,
        counts > 0,
    )


def build_cameras(
    clip_dir: Path, sample: dict, calibration: Dict[str, dict], token: str
) -> Dict[str, dict]:
    cameras = {}
    for bev_name, source_name, _ in CAMERA_MAP:
        calib = calibration[source_name]
        image_token = str(sample[source_name])
        image_path = (clip_dir / "img" / source_name / f"{image_token}.jpg").resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        cam2flu = calib["camera_to_flu"]
        cameras[bev_name] = {
            "data_path": portable_path(image_path),
            "type": bev_name,
            "sample_data_token": f"{token}:{source_name}",
            "sensor2ego_translation": cam2flu[:3, 3].tolist(),
            "sensor2ego_rotation": matrix_to_quaternion(cam2flu),
            "ego2global_translation": [0.0, 0.0, 0.0],
            "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
            "timestamp": int(image_token),
            "cam_intrinsic": calib["intrinsic"],
            "distortion": calib["distortion"],
            "width": calib["width"],
            "height": calib["height"],
            "sensor2lidar_rotation": cam2flu[:3, :3].astype(np.float32),
            "sensor2lidar_translation": cam2flu[:3, 3].astype(np.float32),
        }
    return cameras


def build_clip_infos(
    root: Path,
    clip_name: str,
    calibration: Dict[str, dict],
    points_root: Path,
    skip_points: bool,
) -> List[dict]:
    clip_dir = root / "data" / clip_name
    annotation_dir = root / "annotation" / clip_name
    data_info = json.loads((clip_dir / "data_info.json").read_text(encoding="utf-8-sig"))
    infos = []
    for frame_index, sample in enumerate(data_info["data"]):
        pcd_token = str(sample["pcd"])
        annotation_path = annotation_dir / f"{pcd_token}.json"
        pcd_path = clip_dir / "pcl" / "pcd" / f"{pcd_token}.pcd"
        if not annotation_path.is_file() or not pcd_path.is_file():
            raise FileNotFoundError(annotation_path if not annotation_path.is_file() else pcd_path)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        if annotation.get("info", {}).get("coord") != "RFU":
            raise ValueError(f"Expected RFU annotation coordinates: {annotation_path}")
        if int(annotation.get("isEffective", 1)) != 1:
            continue
        token = f"{clip_name}:{pcd_token}"
        output_path = (points_root / clip_name / f"{pcd_token}.bin").resolve()
        convert_points(pcd_path, output_path, skip_points)
        gt_boxes, gt_names, gt_velocity, num_lidar_pts, valid_flag = (
            convert_annotations(annotation)
        )
        infos.append({
            "lidar_path": portable_path(output_path),
            "token": token,
            "sweeps": [],
            "cams": build_cameras(clip_dir, sample, calibration, token),
            "lidar2ego_translation": [0.0, 0.0, 0.0],
            "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "ego2global_translation": [0.0, 0.0, 0.0],
            "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
            "timestamp": int(pcd_token),
            "prev_token": "" if frame_index == 0 else
                f"{clip_name}:{data_info['data'][frame_index - 1]['pcd']}",
            "next_token": "" if frame_index + 1 == len(data_info["data"]) else
                f"{clip_name}:{data_info['data'][frame_index + 1]['pcd']}",
            "location": clip_name,
            "ann_path": portable_path(annotation_path),
            "gt_boxes": gt_boxes,
            "gt_names": gt_names,
            "gt_velocity": gt_velocity,
            "num_lidar_pts": num_lidar_pts,
            "num_radar_pts": np.zeros(len(gt_boxes), dtype=np.int32),
            "valid_flag": valid_flag,
            "source_coordinate_frame": "RFU",
            "model_reference_frame": "FLU_vehicle",
        })
    return infos


def split_infos(
    clip_infos: List[Tuple[str, List[dict]]], split_by: str, ratio: float
) -> Tuple[List[dict], List[dict]]:
    if not 0.0 < ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if split_by == "clip" and len(clip_infos) > 1:
        split = min(max(int(round(len(clip_infos) * ratio)), 1), len(clip_infos) - 1)
        return (
            [x for _, infos in clip_infos[:split] for x in infos],
            [x for _, infos in clip_infos[split:] for x in infos],
        )
    all_infos = [x for _, infos in clip_infos for x in infos]
    split = min(max(int(round(len(all_infos) * ratio)), 1), len(all_infos) - 1)
    return all_infos[:split], all_infos[split:]


def dump_pickle(path: Path, infos: Iterable[dict], metadata: dict) -> None:
    with path.open("wb") as stream:
        pickle.dump({"infos": list(infos), "metadata": metadata}, stream)


def main() -> None:
    args = parse_args()
    root = Path(args.root_path).resolve()
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration(root)
    data_clips = {p.name for p in (root / "data").iterdir() if p.is_dir()}
    annotation_clips = {
        p.name for p in (root / "annotation").iterdir() if p.is_dir()
    }
    clip_names = sorted(data_clips & annotation_clips)
    if not clip_names:
        raise RuntimeError(f"No matching data/annotation clips found under {root}")
    clip_infos = []
    for index, clip_name in enumerate(clip_names, 1):
        print(f"[{index}/{len(clip_names)}] converting {clip_name}", flush=True)
        infos = build_clip_infos(
            root, clip_name, calibration, output / "points", args.skip_points
        )
        clip_infos.append((clip_name, infos))
    train_infos, val_infos = split_infos(clip_infos, args.split_by, args.train_ratio)
    metadata = {
        "version": "yangluo-v1.0",
        "classes": list(OBJECT_CLASSES),
        "source_coordinate_frame": "RFU",
        "model_reference_frame": "FLU_vehicle",
        "split_by": args.split_by,
        "train_clips": sorted({x["location"] for x in train_infos}),
        "val_clips": sorted({x["location"] for x in val_infos}),
    }
    train_path = output / f"{args.info_prefix}_infos_train.pkl"
    val_path = output / f"{args.info_prefix}_infos_val.pkl"
    dump_pickle(train_path, train_infos, metadata)
    dump_pickle(val_path, val_infos, metadata)
    print(f"train={len(train_infos)} val={len(val_infos)}")
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")


if __name__ == "__main__":
    main()
