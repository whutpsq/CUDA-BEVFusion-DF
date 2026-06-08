from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_CAMERA_TOPICS = [
    "/sensor/camera/center_camera_fov120/encode",
    "/sensor/camera/left_front_camera/encode",
    "/sensor/camera/right_front_camera/encode",
    "/sensor/camera/rear_camera/encode",
    "/sensor/camera/left_rear_camera/encode",
    "/sensor/camera/right_rear_camera/encode",
]


@dataclass
class RsclAdapterConfig:
    config_path: str = ""
    checkpoint_path: str = ""
    output_topic: str = "/perception/bevfusion/objects"
    lidar_topic: str = "/perception/lidar/preproc_points_cloud"
    camera_topics: List[str] = field(default_factory=lambda: list(DEFAULT_CAMERA_TOPICS))
    camera_order: List[str] = field(
        default_factory=lambda: [
            "front",
            "front_left",
            "front_right",
            "rear",
            "rear_left",
            "rear_right",
        ]
    )
    node_name: str = "bevfusion_rscl"
    module_name: str = "bevfusion_rscl"
    device: str = "cuda:0"
    model: str = "bevfusion_df"
    precision: str = "fp16"
    profile: str = "bevfusion_df"
    cuda_model_root: str = "model/bevfusion_df"
    cuda_build_dir: str = "build"
    camera_plan: str | None = None
    vtransform_plan: str | None = None
    lidar_onnx: str | None = None
    fuser_plan: str | None = None
    head_plan: str | None = None
    enable_object_detection: bool = True
    enable_map_segmentation: bool = False
    map_plan: str | None = None
    map_input_binding: str = "middle"
    map_output_binding: str = "map"
    map_score_threshold: float = 0.5
    map_classes: List[str] = field(
        default_factory=lambda: [
            "drivable_area",
            "ped_crossing",
            "walkway",
            "stop_line",
            "carpark_area",
            "divider",
        ]
    )
    print_model_info: bool = False
    score_threshold: float = 0.2
    sync_tolerance_ms: float = 50.0
    image_size: Sequence[int] = (128, 352)
    image_resize: float = 0.48
    image_mean: Sequence[float] = (0.485, 0.456, 0.406)
    image_std: Sequence[float] = (0.229, 0.224, 0.225)
    undistort_images: bool = False
    point_dim: int = 5
    point_cloud_range: Sequence[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    calibration_file: str | None = None
    calibration_extrinsic_direction: str = "lidar2camera"
    allow_identity_calibration: bool = False
    input_message_type: str = "RawMessage"
    output_message_type: str = "RawMessage"
    publish_empty_frame: bool = True
    bag_path: str | None = None
    max_frames: int | None = None
    output_file: str | None = None

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "RsclAdapterConfig":
        return cls(
            config_path=str(data.get("config_path", "")),
            checkpoint_path=str(data.get("checkpoint_path", "")),
            output_topic=str(data.get("output_topic", "/perception/bevfusion/objects")),
            lidar_topic=str(data.get("lidar_topic", "/perception/lidar/preproc_points_cloud")),
            camera_topics=list(data.get("camera_topics", DEFAULT_CAMERA_TOPICS)),
            camera_order=list(
                data.get(
                    "camera_order",
                    ["front", "front_left", "front_right", "rear", "rear_left", "rear_right"],
                )
            ),
            node_name=str(data.get("node_name", "bevfusion_rscl")),
            module_name=str(data.get("module_name", "bevfusion_rscl")),
            device=str(data.get("device", "cuda:0")),
            model=str(data.get("model", "bevfusion_df")),
            precision=str(data.get("precision", "fp16")),
            profile=str(data.get("profile", "bevfusion_df")),
            cuda_model_root=str(data.get("cuda_model_root", "model/bevfusion_df")),
            cuda_build_dir=str(data.get("cuda_build_dir", "build")),
            camera_plan=data.get("camera_plan"),
            vtransform_plan=data.get("vtransform_plan"),
            lidar_onnx=data.get("lidar_onnx"),
            fuser_plan=data.get("fuser_plan"),
            head_plan=data.get("head_plan"),
            enable_object_detection=bool(data.get("enable_object_detection", True)),
            enable_map_segmentation=bool(data.get("enable_map_segmentation", False)),
            map_plan=data.get("map_plan"),
            map_input_binding=str(data.get("map_input_binding", "middle")),
            map_output_binding=str(data.get("map_output_binding", "map")),
            map_score_threshold=float(data.get("map_score_threshold", 0.5)),
            map_classes=list(
                data.get(
                    "map_classes",
                    ["drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area", "divider"],
                )
            ),
            print_model_info=bool(data.get("print_model_info", False)),
            score_threshold=float(data.get("score_threshold", 0.2)),
            sync_tolerance_ms=float(data.get("sync_tolerance_ms", 50.0)),
            image_size=tuple(data.get("image_size", (128, 352))),
            image_resize=float(data.get("image_resize", 0.48)),
            image_mean=tuple(data.get("image_mean", (0.485, 0.456, 0.406))),
            image_std=tuple(data.get("image_std", (0.229, 0.224, 0.225))),
            undistort_images=bool(data.get("undistort_images", False)),
            point_dim=int(data.get("point_dim", 5)),
            point_cloud_range=tuple(
                data.get("point_cloud_range", (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0))
            ),
            calibration_file=data.get("calibration_file"),
            calibration_extrinsic_direction=str(data.get("calibration_extrinsic_direction", "lidar2camera")),
            allow_identity_calibration=bool(data.get("allow_identity_calibration", False)),
            input_message_type=str(data.get("input_message_type", "RawMessage")),
            output_message_type=str(data.get("output_message_type", "RawMessage")),
            publish_empty_frame=bool(data.get("publish_empty_frame", True)),
            bag_path=data.get("bag_path"),
            max_frames=int(data["max_frames"]) if data.get("max_frames") is not None else None,
            output_file=data.get("output_file"),
        )


def load_adapter_config(path: str | Path) -> RsclAdapterConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    cfg = RsclAdapterConfig.from_mapping(data)
    base_dir = path.resolve().parent
    cfg.config_path = _resolve_optional_path(cfg.config_path, base_dir)
    cfg.checkpoint_path = _resolve_optional_path(cfg.checkpoint_path, base_dir)
    cfg.cuda_model_root = _resolve_path(cfg.cuda_model_root, base_dir)
    cfg.cuda_build_dir = _resolve_path(cfg.cuda_build_dir, base_dir)
    cfg.camera_plan = _resolve_optional_path(cfg.camera_plan, base_dir)
    cfg.vtransform_plan = _resolve_optional_path(cfg.vtransform_plan, base_dir)
    cfg.lidar_onnx = _resolve_optional_path(cfg.lidar_onnx, base_dir)
    cfg.fuser_plan = _resolve_optional_path(cfg.fuser_plan, base_dir)
    cfg.head_plan = _resolve_optional_path(cfg.head_plan, base_dir)
    cfg.map_plan = _resolve_optional_path(cfg.map_plan, base_dir)
    if cfg.calibration_file:
        cfg.calibration_file = _resolve_path(cfg.calibration_file, base_dir)
    if cfg.bag_path:
        cfg.bag_path = _resolve_path(cfg.bag_path, base_dir)
    if cfg.output_file:
        cfg.output_file = _resolve_path(cfg.output_file, base_dir)
    return cfg


def _resolve_optional_path(value: str | None, base_dir: Path) -> str | None:
    if not value:
        return None if value is None else ""
    return _resolve_path(value, base_dir)


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())
