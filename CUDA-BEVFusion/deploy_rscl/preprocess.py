from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from PIL import Image

from .config import RsclAdapterConfig
from .sync import SyncedFrame


def load_calibration(path: str | None, camera_order: Sequence[str]) -> Dict[str, Any]:
    if not path:
        return _identity_calibration(camera_order)
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8")
    if path_obj.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    return data


def build_model_input(
    frame: SyncedFrame,
    cfg: RsclAdapterConfig,
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    image_tensors, img_aug_matrix = _build_images(frame, cfg, calibration)
    points = _build_points(frame, cfg)
    matrices = _build_matrices(calibration, cfg.camera_order, cfg.calibration_extrinsic_direction)
    lidar_aug_matrix = np.eye(4, dtype=np.float32)[None]

    metas = [
        {
            "timestamp": frame.timestamp_us,
            "token": str(frame.timestamp_us),
            "lidar_path": "rscl://{}".format(cfg.lidar_topic),
            "filename": ["rscl://{}".format(t) for t in cfg.camera_topics],
            "img_shape": [tuple(cfg.image_size)] * len(cfg.camera_topics),
            "ori_shape": [tuple(cfg.image_size)] * len(cfg.camera_topics),
            "pad_shape": [tuple(cfg.image_size)] * len(cfg.camera_topics),
            "lidar2image": matrices["lidar2image"].tolist(),
        }
    ]

    return {
        "img": image_tensors,
        "points": [points],
        "camera2ego": matrices["camera2ego"][None],
        "lidar2ego": matrices["lidar2ego"][None],
        "lidar2camera": matrices["lidar2camera"][None],
        "lidar2image": matrices["lidar2image"][None],
        "camera_intrinsics": matrices["camera_intrinsics"][None],
        "camera2lidar": matrices["camera2lidar"][None],
        "img_aug_matrix": img_aug_matrix[None],
        "lidar_aug_matrix": lidar_aug_matrix,
        "metas": metas,
    }


def _build_images(
    frame: SyncedFrame,
    cfg: RsclAdapterConfig,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(cfg.image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(cfg.image_std, dtype=np.float32).reshape(1, 1, 3)
    tensors: List[np.ndarray] = []
    aug_mats: List[np.ndarray] = []
    for name in cfg.camera_order:
        image = frame.cameras[name]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Camera payload for {name} must be PIL.Image or numpy array")
        image = image.convert("RGB")
        if cfg.undistort_images:
            image = _undistort_image(image, calibration, name)
        image, aug = _resize_crop_image(image, cfg)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        tensors.append(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        aug_mats.append(aug)
    return np.stack(tensors, axis=0)[None].astype(np.float32, copy=False), np.stack(aug_mats, axis=0)


def _resize_crop_image(image: Image.Image, cfg: RsclAdapterConfig) -> tuple[Image.Image, np.ndarray]:
    width, height = image.size
    final_h, final_w = int(cfg.image_size[0]), int(cfg.image_size[1])
    resize = float(getattr(cfg, "image_resize", 0.48))
    resize_w = max(int(width * resize), final_w)
    resize_h = max(int(height * resize), final_h)
    crop_w = int(max(0, resize_w - final_w) / 2)
    crop_h = int(max(0, resize_h - final_h))
    image = image.resize((resize_w, resize_h), Image.BILINEAR)
    image = image.crop((crop_w, crop_h, crop_w + final_w, crop_h + final_h))

    aug = np.eye(4, dtype=np.float32)
    aug[0, 0] = resize
    aug[1, 1] = resize
    aug[0, 3] = -float(crop_w)
    aug[1, 3] = -float(crop_h)
    return image, aug


def _build_points(frame: SyncedFrame, cfg: RsclAdapterConfig) -> np.ndarray:
    points = np.asarray(frame.lidar, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, cfg.point_dim)
    elif points.ndim != 2:
        raise ValueError(f"Expected lidar points to be 1D or 2D, got shape {points.shape}")
    pcd_range = np.asarray(cfg.point_cloud_range, dtype=np.float32)
    mask = (
        (points[:, 0] > pcd_range[0])
        & (points[:, 0] < pcd_range[3])
        & (points[:, 1] > pcd_range[1])
        & (points[:, 1] < pcd_range[4])
        & (points[:, 2] > pcd_range[2])
        & (points[:, 2] < pcd_range[5])
    )
    points = points[mask]
    if points.shape[1] < 5:
        pad = np.zeros((points.shape[0], 5 - points.shape[1]), dtype=np.float32)
        points = np.concatenate([points, pad], axis=1)
    return np.ascontiguousarray(points.astype(np.float32, copy=False))


def _build_matrices(
    calibration: Mapping[str, Any],
    camera_order: Sequence[str],
    extrinsic_direction: str = "lidar2camera",
) -> Dict[str, np.ndarray]:
    lidar2ego = _matrix(calibration.get("lidar2ego", np.eye(4, dtype=np.float32)))
    if "cameras" in calibration:
        camera2ego = _stack_camera_matrix(calibration, camera_order, "camera2ego")
        camera_intrinsics = _stack_camera_matrix(calibration, camera_order, "camera_intrinsics")
        ego2lidar = np.linalg.inv(lidar2ego)
        camera2lidar = np.matmul(ego2lidar[None], camera2ego)
        lidar2camera = np.linalg.inv(camera2lidar)
    else:
        camera_intrinsics = _stack_top_level_camera_matrix(calibration, camera_order, "cam_intrinsic")
        extrinsic = _stack_top_level_camera_matrix(calibration, camera_order, "extrinsic")
        if extrinsic_direction == "lidar2camera":
            lidar2camera = extrinsic
            camera2lidar = np.linalg.inv(lidar2camera)
        elif extrinsic_direction == "camera2lidar":
            camera2lidar = extrinsic
            lidar2camera = np.linalg.inv(camera2lidar)
        else:
            raise ValueError(
                "calibration_extrinsic_direction must be 'lidar2camera' or 'camera2lidar', "
                f"got {extrinsic_direction!r}"
            )
        camera2ego = np.matmul(lidar2ego[None], camera2lidar)
    lidar2image = np.matmul(camera_intrinsics, lidar2camera)
    return {
        "camera2ego": np.ascontiguousarray(camera2ego),
        "lidar2ego": np.ascontiguousarray(lidar2ego),
        "camera_intrinsics": np.ascontiguousarray(camera_intrinsics),
        "camera2lidar": np.ascontiguousarray(camera2lidar),
        "lidar2camera": np.ascontiguousarray(lidar2camera),
        "lidar2image": np.ascontiguousarray(lidar2image),
    }


def _stack_camera_matrix(
    calibration: Mapping[str, Any], camera_order: Sequence[str], key: str
) -> np.ndarray:
    cameras = calibration.get("cameras", {})
    values = []
    for name in camera_order:
        cam = cameras.get(name, {})
        if key in cam:
            values.append(_matrix(cam[key]))
        else:
            values.append(np.eye(4, dtype=np.float32))
    return np.stack(values, axis=0)


def _stack_top_level_camera_matrix(
    calibration: Mapping[str, Any], camera_order: Sequence[str], key: str
) -> np.ndarray:
    values = []
    for name in camera_order:
        cam = _camera_entry(calibration, name)
        if key not in cam:
            raise KeyError(f"Calibration for camera {name!r} misses {key!r}")
        values.append(_matrix(cam[key]))
    return np.stack(values, axis=0)


def _camera_entry(calibration: Mapping[str, Any], camera_name: str) -> Mapping[str, Any]:
    if "cameras" in calibration:
        return calibration.get("cameras", {}).get(camera_name, {})
    return calibration.get(camera_name, {})


def _undistort_image(image: Image.Image, calibration: Mapping[str, Any], camera_name: str) -> Image.Image:
    cam = _camera_entry(calibration, camera_name)
    dist = cam.get("cam_dist") if isinstance(cam, Mapping) else None
    if not dist:
        return image
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("undistort_images is enabled, but OpenCV is not installed") from exc

    intrinsics = np.asarray(cam.get("cam_intrinsic"), dtype=np.float32)
    coeff = np.asarray(
        [
            float(dist.get("k1", 0.0)),
            float(dist.get("k2", 0.0)),
            float(dist.get("p1", 0.0)),
            float(dist.get("p2", 0.0)),
            float(dist.get("k3", 0.0)),
            float(dist.get("k4", 0.0)),
            float(dist.get("k5", 0.0)),
            float(dist.get("k6", 0.0)),
        ],
        dtype=np.float32,
    )
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    undistorted = cv2.undistort(arr, intrinsics, coeff)
    return Image.fromarray(undistorted).convert("RGB")


def _matrix(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (3, 3):
        padded = np.eye(4, dtype=np.float32)
        padded[:3, :3] = arr
        arr = padded
    if arr.shape == (3, 4):
        padded = np.eye(4, dtype=np.float32)
        padded[:3, :4] = arr
        arr = padded
    if arr.shape != (4, 4):
        raise ValueError(f"Expected matrix shape (4, 4) or (3, 3), got {arr.shape}")
    return np.ascontiguousarray(arr)


def _identity_calibration(camera_order: Iterable[str]) -> Dict[str, Any]:
    return {
        "lidar2ego": np.eye(4, dtype=np.float32).tolist(),
        "cameras": {
            name: {
                "camera2ego": np.eye(4, dtype=np.float32).tolist(),
                "camera_intrinsics": np.eye(4, dtype=np.float32).tolist(),
            }
            for name in camera_order
        },
    }
