from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import RsclAdapterConfig


class BevFusionRunner:
    """CUDA-BEVFusion inference backend for the RSCL adapter.

    The runner intentionally keeps one long-lived libpybev Core instance. RSCL
    callbacks only update per-frame matrices and run forward, which is the
    behavior needed for continuous online inference.
    """

    def __init__(self, cfg: RsclAdapterConfig) -> None:
        self.cfg = cfg
        self.repo_root = Path(__file__).resolve().parents[1]
        self._prepare_python_path()
        self.core = self._load_core()

    def infer(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        images = _as_contiguous_numpy(data["img"], np.float16)
        points = _as_contiguous_numpy(_first_points(data["points"]), np.float16)

        self.core.update(
            _matrix_array(data["camera2lidar"]),
            _matrix_array(data["camera_intrinsics"]),
            _matrix_array(data["lidar2image"]),
            _matrix_array(data["img_aug_matrix"]),
        )
        if self.cfg.enable_object_detection:
            boxes = self.core.forward(
                images,
                points,
                with_normalization=False,
                with_dlpack=False,
            )
            output = _format_cuda_boxes(boxes)
        else:
            output = _format_cuda_boxes(np.empty((0, 11), dtype=np.float32))
        if self.cfg.enable_map_segmentation:
            output["map_probs"] = np.ascontiguousarray(self.core.forward_map(images, points), dtype=np.float32)
            output["map_classes"] = list(self.cfg.map_classes)
            output["map_score_threshold"] = float(self.cfg.map_score_threshold)
        return [output]

    def _prepare_python_path(self) -> None:
        candidates = [
            self.repo_root / "tool",
            Path(self.cfg.cuda_build_dir),
            self.repo_root / "build",
        ]
        for path in candidates:
            if path.exists():
                text = str(path.resolve())
                if text not in sys.path:
                    sys.path.insert(0, text)

    def _load_core(self):
        libpybev = self._import_libpybev()

        model_root = Path(self.cfg.cuda_model_root)
        camera = Path(self.cfg.camera_plan) if self.cfg.camera_plan else model_root / "build" / "camera.backbone.plan"
        vtransform = (
            Path(self.cfg.vtransform_plan)
            if self.cfg.vtransform_plan
            else model_root / "build" / "camera.vtransform.plan"
        )
        lidar = Path(self.cfg.lidar_onnx) if self.cfg.lidar_onnx else model_root / "lidar.backbone.xyz.onnx"
        fuser = Path(self.cfg.fuser_plan) if self.cfg.fuser_plan else model_root / "build" / "fuser.plan"
        head = ""
        if self.cfg.enable_object_detection:
            head_path = Path(self.cfg.head_plan) if self.cfg.head_plan else model_root / "build" / "head.bbox.plan"
            if not head_path.exists():
                raise RuntimeError(
                    "Object detection is enabled, but the TensorRT bbox head plan does not exist: "
                    f"{head_path}. Build head.bbox.plan first, set head_plan, or set enable_object_detection: false."
                )
            head = str(head_path)
        mapseg = ""
        if self.cfg.enable_map_segmentation:
            mapseg_path = Path(self.cfg.map_plan) if self.cfg.map_plan else model_root / "build" / "head.map.plan"
            if not mapseg_path.exists():
                raise RuntimeError(
                    "Map segmentation is enabled, but the TensorRT map head plan does not exist: "
                    f"{mapseg_path}. Export/build head.map.plan first or set map_plan in the adapter config."
                )
            mapseg = str(mapseg_path)

        core = libpybev.load_bevfusion(
            str(camera),
            str(vtransform),
            str(lidar),
            str(fuser),
            str(head),
            self.cfg.precision,
            self.cfg.profile,
            mapseg,
            self.cfg.map_input_binding,
            self.cfg.map_output_binding,
        )
        if core is None:
            raise RuntimeError(
                "Failed to create CUDA-BEVFusion core. "
                f"camera={camera} vtransform={vtransform} lidar={lidar} fuser={fuser} head={head}"
            )
        if self.cfg.print_model_info:
            core.print()
        return core

    def _import_libpybev(self):
        try:
            import libpybev

            return libpybev
        except ModuleNotFoundError:
            pass

        candidates = [
            Path(self.cfg.cuda_build_dir) / "libpybev.so",
            self.repo_root / "build" / "libpybev.so",
            self.repo_root / "libpybev.so",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            spec = importlib.util.spec_from_file_location("libpybev", str(candidate.resolve()))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules["libpybev"] = module
            return module

        searched = "\n  ".join(str(path.resolve()) for path in candidates)
        raise RuntimeError(
            "Cannot find libpybev.so for CUDA-BEVFusion RSCL inference.\n"
            "Build the Python binding first with USE_Python=ON, or set cuda_build_dir in "
            "deploy_rscl/configs/bevfusion_rscl.yaml.\n"
            f"Searched:\n  {searched}"
        )


def _first_points(points: Any) -> Any:
    if isinstance(points, list):
        if not points:
            return np.empty((0, 5), dtype=np.float32)
        return points[0]
    return points


def _as_contiguous_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    arr = np.asarray(value, dtype=dtype)
    return np.ascontiguousarray(arr)


def _matrix_array(value: Any) -> np.ndarray:
    arr = _as_contiguous_numpy(value, np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return np.ascontiguousarray(arr)


def _format_cuda_boxes(boxes: Any) -> Dict[str, np.ndarray]:
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return {
            "boxes_3d": np.empty((0, 9), dtype=np.float32),
            "scores_3d": np.empty((0,), dtype=np.float32),
            "labels_3d": np.empty((0,), dtype=np.int64),
        }
    arr = arr.reshape(-1, arr.shape[-1])
    return {
        "boxes_3d": np.ascontiguousarray(arr[:, :9], dtype=np.float32),
        "scores_3d": np.ascontiguousarray(arr[:, 10], dtype=np.float32),
        "labels_3d": np.ascontiguousarray(arr[:, 9].astype(np.int64)),
    }
