#!/usr/bin/env python3
"""Audit the local Yangluo source/config handoff without runtime artifacts."""

from pathlib import Path
import ast
import json
import sys


REQUIRED_PATHS = (
    "CMakeLists.txt",
    "deploy_rscl/config.py",
    "deploy_rscl/preprocess.py",
    "deploy_rscl/cpp/include/rscl_adapter/config.hpp",
    "deploy_rscl/cpp/src/config.cpp",
    "deploy_rscl/cpp/src/preprocess.cpp",
    "deploy_rscl/cpp/src/pipeline.cpp",
    "yangluogang/camera_calibration.json",
    "yangluogang/lidar_calibration.json",
    "yangluo_deploy/README.md",
    "yangluo_deploy/RESNET50_TRAIN_EXPORT_RUNBOOK_CN.md",
    "yangluo_deploy/cpp/native_undistort.hpp",
    "yangluo_deploy/cpp/native_undistort.cpp",
    "yangluo_deploy/configs/classes.txt",
    "yangluo_deploy/configs/runtime.yaml",
    "yangluo_deploy/configs/runtime_vehicle.yaml",
    "yangluo_deploy/configs/runtime_vehicle_48ms.yaml",
    "yangluo_deploy/configs/calibration_runtime.json",
    "yangluo_deploy/configs/calibration_vehicle_0_10.json",
    "yangluo_deploy/export/export_fp16.py",
    "yangluo_deploy/export/export_fuser_head.py",
    "yangluo_deploy/export/export_lidar.py",
    "yangluo_deploy/tensorrt/CMakeLists.txt",
    "yangluo_deploy/tensorrt/build_engine.cpp",
    "yangluo_deploy/tensorrt/build_fp16_engines.sh",
    "yangluo_deploy/ros_ws/src/yangluo_bevfusion/CMakeLists.txt",
    "yangluo_deploy/ros_ws/src/yangluo_bevfusion/src/yangluo_bevfusion_node.cpp",
    "yangluo_deploy/ros_ws/src/yangluo_bevfusion_msgs/CMakeLists.txt",
    "yangluo_deploy/ros_ws/src/yangluo_bevfusion_msgs/msg/DetectedObject.msg",
    "yangluo_deploy/ros_ws/src/yangluo_bevfusion_msgs/msg/DetectedObjectArray.msg",
    "yangluo_deploy/tools/build_arm64.sh",
    "yangluo_deploy/tools/build_thor_fp16.sh",
    "yangluo_deploy/tools/run_thor_fp16.sh",
    "yangluo_deploy/tools/run_vehicle_fp16.sh",
    "yangluo_deploy/tools/convert_calibration.py",
    "yangluo_deploy/tools/validate_native_undistort.py",
    "yangluo_deploy/training/make_resnet50_config.py",
    "yangluo_deploy/training/adapter_snapshot/README.md",
    "yangluo_deploy/training/adapter_snapshot/convert_yangluo.py",
    "yangluo_deploy/training/adapter_snapshot/validate_yangluo.py",
    "yangluo_deploy/training/adapter_snapshot/demo_overfit.yaml",
    "yangluo_deploy/training/adapter_snapshot/train.yaml",
)

TEXT_CONTRACTS = {
    "yangluo_deploy/training/adapter_snapshot/convert_yangluo.py": (
        '("CAM_FRONT", "cam5", "0")',
        '("CAM_BACK", "cam10", "10")',
        '"source_coordinate_frame": "RFU"',
        '"model_reference_frame": "FLU_vehicle"',
    ),
    "deploy_rscl/cpp/include/rscl_adapter/config.hpp": (
        'input_point_coordinate_frame = "FLU"',
    ),
    "deploy_rscl/cpp/src/preprocess.cpp": ("RFU (right, front, up) -> FLU",),
    "yangluo_deploy/configs/runtime_vehicle_48ms.yaml": (
        'input_point_coordinate_frame: "RFU"',
        'output_topic: "/perception/bevfusion/objects"',
        "camera_time_offsets_ms: [48.0, 48.0]",
        'calibration_file: "calibration_vehicle_0_10.json"',
    ),
    "yangluo_deploy/ros_ws/src/yangluo_bevfusion_msgs/msg/DetectedObject.msg": (
        "geometry_msgs/Pose pose",
        "geometry_msgs/Vector3 dimensions",
        "string class_name",
        "float32 score",
    ),
    "CMakeLists.txt": ("RSCL_ENABLE_NATIVE_UNDISTORT",),
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    missing = [relative for relative in REQUIRED_PATHS if not (repo / relative).is_file()]
    if missing:
        fail("missing required files:\n  " + "\n  ".join(missing))

    for relative, markers in TEXT_CONTRACTS.items():
        text = (repo / relative).read_text(encoding="utf-8")
        absent = [marker for marker in markers if marker not in text]
        if absent:
            fail(f"contract marker missing from {relative}: {absent}")

    python_files = sorted((repo / "yangluo_deploy").rglob("*.py"))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            fail(f"Python syntax check failed for {path.relative_to(repo)}: {exc}")

    calibration_payloads = {}
    for relative in (
        "yangluo_deploy/configs/calibration_runtime.json",
        "yangluo_deploy/configs/calibration_vehicle_0_10.json",
    ):
        try:
            calibration_payloads[relative] = json.loads(
                (repo / relative).read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            fail(f"JSON check failed for {relative}: {exc}")

    vehicle_calibration = calibration_payloads[
        "yangluo_deploy/configs/calibration_vehicle_0_10.json"
    ]
    vehicle_cameras = vehicle_calibration.get("cameras", {})
    expected_sources = {"front": "0", "rear": "10"}
    for name, source_id in expected_sources.items():
        camera = vehicle_cameras.get(name)
        if not isinstance(camera, dict):
            fail(f"vehicle calibration is missing camera {name}")
        if str(camera.get("source_camera_id")) != source_id:
            fail(
                f"vehicle camera {name} must use source calibration {source_id}, "
                f"got {camera.get('source_camera_id')}"
            )
        distortion = camera.get("distortion", [])
        if len(distortion) != 8:
            fail(
                f"vehicle camera {name} runtime distortion must contain 8 values, "
                f"got {len(distortion)}"
            )

    classes = [
        line.strip()
        for line in (repo / "yangluo_deploy/configs/classes.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(classes) != 11 or len(set(classes)) != len(classes):
        fail(f"expected 11 unique class names, got {len(classes)}: {classes}")

    print(f"required_files={len(REQUIRED_PATHS)}")
    print(f"python_syntax_files={len(python_files)}")
    print(f"classes={len(classes)}")
    print("camera_contract=cam5_to_calibration0,cam10_to_calibration10")
    print("point_contract=RFU_to_FLU")
    print("runtime_artifacts_required=false")
    print("YANGLUO_LOCAL_SOURCE_COMPLETE")


if __name__ == "__main__":
    main()
