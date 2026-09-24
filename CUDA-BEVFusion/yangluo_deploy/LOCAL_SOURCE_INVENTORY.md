# Local source and configuration inventory

This inventory defines the source-only Yangluo handoff stored in this
repository.  It intentionally excludes generated or hardware-specific
artifacts such as AArch64 executables, shared libraries, TensorRT `.plan`
files, ONNX files, checkpoints, bags, build directories, and logs.

## Included source/configuration

- `yangluo_deploy/training/adapter_snapshot/`: corrected, local copy of the
  dataset converter, validation scripts, split tool, visualizer, and original
  training YAML files.
- `yangluo_deploy/training/`: ResNet50 config generation and optional
  compatible-checkpoint utilities.
- `yangluo_deploy/export/`: FP16 ONNX exporters and sparse export helpers.
- `yangluo_deploy/tensorrt/`: TensorRT builder source and target-side engine
  build script.  No generated plans are stored.
- `yangluo_deploy/ros_ws/src/`: ROS1 nodes, launch files, and custom message
  definitions for `/perception/bevfusion/objects`.
- `yangluo_deploy/cpp/`: OpenCV-free native image undistortion implementation.
- `yangluo_deploy/configs/`: class order, bag configuration, current
  legacy-model vehicle configuration, and calibration JSON files.
- `yangluo_deploy/tools/`: ARM64 build scripts, Thor wrappers, calibration
  conversion, contract checks, bag validation, and diagnostics.
- `deploy_rscl/`: shared preprocessing/pipeline source extended with the
  configurable RFU-to-FLU point conversion.  The default remains FLU, so the
  passenger-car path is unchanged.
- root `CMakeLists.txt`: isolated native-undistortion build option.
- `yangluogang/camera_calibration.json` and
  `yangluogang/lidar_calibration.json`: raw calibration inputs.

## Contract status

- Live PointCloud2 payload: RFU.  Vehicle YAML selects
  `input_point_coordinate_frame: "RFU"`; preprocessing converts to FLU.
- Correct camera/calibration mapping for training and live vehicle runtime:
  `cam5 -> 0`, `cam10 -> 10`.
- `calibration_vehicle_0_10.json` is generated from
  `yangluogang/camera_calibration.json`. Standard 5-coefficient OpenCV
  distortion is padded with zero `k4,k5,k6` values for the fixed 8-value C++
  runtime representation.
- Current output contract: ROS1
  `yangluo_bevfusion_msgs/DetectedObjectArray` on
  `/perception/bevfusion/objects`, including position, dimensions, yaw,
  velocity, class, label, and score.

Run the source-only audit from the repository root:

```bash
python yangluo_deploy/tools/check_local_source_complete.py
```

The expected success marker is `YANGLUO_LOCAL_SOURCE_COMPLETE`.
