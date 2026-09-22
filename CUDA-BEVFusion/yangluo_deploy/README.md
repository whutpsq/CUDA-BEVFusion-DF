# Yangluo port-truck BEVFusion deployment

This directory is an isolated ROS1/FP16 deployment path for the Yangluo
port-truck model.  It does not replace the passenger-car `src/main.cpp`, QAT
scripts, or the RSCL deployment path.

For the complete Chinese ResNet50 training, FP16 ONNX export and repeatable
deployment procedure, see `RESNET50_TRAIN_EXPORT_RUNBOOK_CN.md`.

## Verified input contract

- ROS1 Noetic.
- Front camera: `/df/camera/0/image/compressed`, `sensor_msgs/CompressedImage`.
- Rear camera: `/df/camera/5/image/compressed`, `sensor_msgs/CompressedImage`.
- LiDAR: `/perception/lidar/concated_points_cloud`, `sensor_msgs/PointCloud2`.
- LiDAR fields: `x`, `y`, `z`, `intensity`, `timestamp`, `ring`, `label`.
- Model point layout: `[x, y, z, intensity, 0]`.  The fifth training feature
  was zero; it is deliberately not populated with the per-point timestamp.
- Output: `/perception/bevfusion/objects`,
  `yangluo_bevfusion_msgs/DetectedObjectArray`, frame `base_link`.
- Camera order: front then rear.  Runtime camera IDs are 0 and 5; the training
  source used IDs 5 and 10.
- Images are rectified with the rational eight-coefficient distortion model,
  standardized to 1920x1080, then resized/cropped to 704x256.
- Initial synchronization tolerance: 50 ms.

## Validation boundary

The available ARM64 container has CUDA 13.0 and TensorRT 10.13.3 headers and
libraries but no GPU.  It can compile AArch64 artifacts.  TensorRT plans must
be generated and load-tested on the real DRIVE AGX Thor.  Passing an AArch64
build or ELF check is not a Thor runtime result.

## Directory layout

- `training/`: create an isolated ResNet50 training config without changing
  `H:/df_code/bevfusion/yangluo_adapter`.  Full-model warm-start conversion is
  optional and is not used by the default Yangluo ResNet50 training path.
- `export/`: export the two-camera FP16 ONNX components on the x86 GPU server.
- `tools/`: calibration conversion, bag-contract validation, and preflight.
- `configs/`: ROS/runtime configuration and class names.
- `ros_ws/src/`: isolated ROS message and node packages.
- `tensorrt/`: target-native ONNX-to-plan builder for real Thor.

The existing sibling dependency
`libraries/3DSparseConvolution/libspconv/lib/aarch64_cuda13.0/libspconv.so`
has been checked as a 64-bit AArch64 ELF (32,575,856 bytes, SHA-256
`675a7c6647912cd40fbd7830901b29f175936c6efcf43b5208f91b85a6c73f80`) and
contains Blackwell/SM110 support markers.  Copy the complete sibling
`libraries/3DSparseConvolution/libspconv` directory with this repository.

To avoid transferring bags, old outputs and cached SDK files, create a small
source/dependency bundle from Windows PowerShell:

```powershell
Set-Location H:\github\Lidar_AI_Solution\CUDA-BEVFusion
powershell -ExecutionPolicy Bypass -File .\yangluo_deploy\tools\package_arm64_source.ps1
```

The archive contains the CUDA-BEVFusion sources plus only the required
`dependencies`, `cuOSD/src`, and AArch64 CUDA 13 spconv files.  The script
prints its path, size and SHA-256 for verification after upload.

## 1. Prepare the ResNet50 training inputs

Download `resnet50-0676ba61.pth` on a networked machine and copy it to the
training checkout as `pretrained/resnet50-0676ba61.pth`.

From the CUDA-BEVFusion repository, with the original training environment
active:

```bash
python yangluo_deploy/training/make_resnet50_config.py \
  --input /path/to/bevfusion/yangluo_adapter/demo_overfit.yaml \
  --output yangluo_deploy/generated/demo_overfit_resnet50.yaml \
  --pretrained /path/to/bevfusion/pretrained/resnet50-0676ba61.pth
```

The generated config explicitly sets `load_from: null` and `resume_from: null`.
Only the ResNet50 camera backbone loads ImageNet pretrained weights; the camera
neck, LiDAR branch, fusion layers, decoder, and 11-class detection head start
from their normal random initialization.  The old Swin `epoch_100.pth` is not
read by this training path.

The ROS output reads class names at runtime, so a later dataset may use a
different class count without changing the C++ node.  Retrain/re-export the
matching head, replace `configs/classes.txt` in the exact model-label order,
and run preflight with `--allow-class-change`.  Never change only the text
file while reusing an old detection head.

Train using the same command style as the original adapter:

```bash
cd /path/to/bevfusion
torchpack dist-run -np 1 python tools/train.py \
  /path/to/CUDA-BEVFusion/yangluo_deploy/generated/demo_overfit_resnet50.yaml \
  --run-dir yangluo_runs/yangluo_resnet50_demo
```

## 2. Install offline export dependencies

Do not upgrade the working environment's PyTorch, CUDA, MMCV, TorchPack,
NumPy, or Protobuf packages.  For the verified Python 3.8 x86_64 environment,
install only the matching ONNX wheel without dependencies into an isolated
directory.  ONNX Simplifier and ONNX Runtime are optional and are not required
for the first FP16 export:

```bash
python -m pip install --no-index --no-deps \
  --target yangluo_deploy/python_packages/onnx_1_16_2 \
  /path/to/onnx-1.16.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
export PYTHONPATH="$PWD/yangluo_deploy/python_packages/onnx_1_16_2${PYTHONPATH:+:$PYTHONPATH}"
```

`pytorch-quantization` is not required for this FP16 phase.  The isolated
export implementation and its sparse-convolution helpers live entirely under
`yangluo_deploy/export/`; it does not modify or invoke the passenger-car
`qat/` entrypoints.

## 3. Export FP16 ONNX on the x86 GPU server

```bash
python yangluo_deploy/export/export_fp16.py \
  --bevfusion-root /path/to/bevfusion \
  --config /path/to/CUDA-BEVFusion/yangluo_deploy/generated/demo_overfit_resnet50.yaml \
  --ckpt /path/to/bevfusion/yangluo_runs/yangluo_resnet50_demo/epoch_100.pth \
  --save-root /path/to/CUDA-BEVFusion/model/yangluo_resnet50_fp16 \
  --num-cameras 2
```

Success requires these five non-empty files:

```text
camera.backbone.onnx
camera.vtransform.onnx
lidar.backbone.xyz.onnx
fuser.onnx
head.bbox.onnx
```

## 4. Generate runtime calibration

```bash
python yangluo_deploy/tools/convert_calibration.py \
  --input yangluogang/camera_calibration.json \
  --output yangluo_deploy/configs/calibration_runtime.json \
  --front-id 0 --rear-id 5
```

The converter applies the same FLU-to-RFU basis change used by the training
adapter.  Since the bag point cloud already has `frame_id=base_link`,
`lidar2ego` remains identity.

The converted 0/5 calibration is already checked in as
`configs/calibration_runtime.json`; the command above is the reproducible
regeneration step.

## 5. Build and run boundaries

Run `tools/preflight.py` before copying artifacts.  Build instructions for the
ARM64 core and ROS workspace are in `ros_ws/README.md`.  Build the TensorRT
plans only on the real Thor using `tensorrt/build_fp16_engines.sh`; the x86
host running an emulated ARM64 container cannot perform that validation.

The custom TensorRT builder itself can be compiled in the ARM64 container:

```bash
cmake -S yangluo_deploy/tensorrt -B build_yangluo_trt_builder
cmake --build build_yangluo_trt_builder -j$(nproc)
```

After copying the ONNX directory and build products to the real Thor:

```bash
bash yangluo_deploy/tensorrt/build_fp16_engines.sh \
  model/yangluo_resnet50_fp16 \
  yangluo_deploy/generated/build_arm64_core \
  build_yangluo_trt_builder/yangluo_trt_builder
```

This generates and immediately deserializes each plan on Thor.  The success
marker is `FP16_ENGINE_SET_OK`.  The LiDAR sparse backbone remains
`lidar.backbone.xyz.onnx` and is executed by the matching target
`libspconv.so`.

The final Thor bundle is relocatable and does not require `/workspace`.
After extracting it, use the wrappers below; they derive every project path
from their own location and deliberately reject the QEMU CUDA-driver stub:

```bash
bash CUDA-BEVFusion/yangluo_deploy/tools/build_thor_fp16.sh
bash CUDA-BEVFusion/yangluo_deploy/tools/run_thor_fp16.sh --check-only
bash CUDA-BEVFusion/yangluo_deploy/tools/run_thor_fp16.sh
```

The normal run requires an existing ROS master.  It starts the compiled node
directly so that a catkin workspace built under `/workspace` can be moved to a
different absolute directory on the real Thor.  Runtime libraries are found
through paths relative to the extracted bundle, while ROS, CUDA and TensorRT
SDK prefixes retain their vehicle-provided locations.
