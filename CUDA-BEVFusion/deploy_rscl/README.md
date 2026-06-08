# CUDA-BEVFusion RSCL Adapter

This directory runs CUDA-BEVFusion continuously from RSCL camera/lidar streams.
It reuses the PyTorch adapter's camera/lidar decoders and frame synchronizer,
but the inference backend is `libpybev` plus the TensorRT engines in this repo.

## Runtime Shape

- Cameras and lidar are subscribed from RSCL topics.
- `FrameSynchronizer` emits one synchronized frame at a time.
- `preprocess.py` resizes/crops/normalizes camera images and pads lidar points
  to 5 channels using NumPy/PIL.
- `runner.py` keeps one long-lived CUDA-BEVFusion core and calls:
  - `core.update(camera2lidar, camera_intrinsics, lidar2image, img_aug_matrix)`
  - `core.forward(img_fp16, points_fp16, with_normalization=False)`
- Online detections are published to `output_topic` as a JSON RawMessage with
  `timestamp_us` and `objects`. Offline bag runs can write the same payloads to
  JSONL for debugging.

Default detection payload:

```json
{
  "objects": [
    {
      "label": 0,
      "score": 0.82,
      "box": [12.3, -1.4, 0.8, 4.2, 1.8, 1.6, 1.57, 0.0, 0.0]
    }
  ],
  "timestamp_us": 1780407801234567
}
```

`box` is `[x, y, z, dx, dy, dz, yaw, vx, vy]` in lidar coordinates.

When `enable_map_segmentation: true`, the payload also includes a BEV map mask:

```json
{
  "map": {
    "classes": ["drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area", "divider"],
    "shape": [6, 200, 200],
    "threshold": 0.5,
    "encoding": "base64_uint8_chw",
    "data": "..."
  }
}
```

The map mask is thresholded from the TensorRT map head probabilities and encoded
as contiguous `uint8` in `[C,H,W]` order.

## Build CUDA-BEVFusion Python Binding

Build `libpybev.so` first:

```bash
export USE_Python=ON
export DEBUG_MODEL=bevfusion_df
export DEBUG_PROFILE=bevfusion_df
export DEBUG_PRECISION=fp16
bash tool/run.sh
```

The adapter expects:

```text
build/libpybev.so
model/bevfusion_df/lidar.backbone.xyz.onnx
model/bevfusion_df/build/camera.backbone.plan
model/bevfusion_df/build/camera.vtransform.plan
model/bevfusion_df/build/fuser.plan
model/bevfusion_df/build/head.bbox.plan
```

Relative paths are configured in `configs/bevfusion_rscl.yaml`.

## Offline rsclbag Run

Source RSCL first:

```bash
source /opt/senseauto_active/senseauto-rscl/resource/scripts/setup.sh
cd /path/to/CUDA-BEVFusion
```

Check message decode and synchronization only:

```bash
python3 -m deploy_rscl.rscl_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /workspace/mybag.000.rsclbag \
  --decode-only \
  --max-frames 3
```

Check image/lidar decode without loading CUDA-BEVFusion:

```bash
python3 -m pip install av
python3 -m deploy_rscl.rscl_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /workspace/mybag.000.rsclbag \
  --decode-images-only \
  --max-frames 3
```

Run continuous CUDA-BEVFusion inference from a bag:

```bash
python3 -m deploy_rscl.rscl_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /workspace/mybag.000.rsclbag \
  --max-frames 20 \
  --output-file runs/rsclbag_cuda_outputs.jsonl
```

Run the official CUDA-BEVFusion ResNet50 model instead of the custom
`bevfusion_df` model:

```bash
export USE_Python=ON
export DEBUG_MODEL=resnet50
export DEBUG_PROFILE=default
export DEBUG_PRECISION=fp16
source tool/environment.sh
bash tool/build_trt_engine.sh
cd build && cmake .. && make -j && cd ..

python3 -m deploy_rscl.rscl_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_resnet50_rscl.yaml \
  --bag /workspace/mybag.000.rsclbag \
  --max-frames 20 \
  --output-file runs/rsclbag_resnet50_outputs.jsonl
```

The ResNet50 config uses `model/resnet50`, `profile: default`,
`image_size: [256, 704]`, and point cloud range
`[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]`, matching the non-`bevfusion_df`
parameter branch in the CUDA Python binding.

## Optional Map Segmentation

CUDA-BEVFusion originally accelerates object detection only. This adapter adds
an optional TensorRT map head path. To enable it, provide a map segmentation
head engine:

```text
model/<model-name>/head.map.onnx
model/<model-name>/build/head.map.plan
```

The expected TensorRT binding names are:

```text
input:  middle
output: map
```

If your exported ONNX uses different binding names, set:

```yaml
map_input_binding: "middle"
map_output_binding: "map"
```

Enable the output in the RSCL config:

```yaml
enable_map_segmentation: true
map_plan: "../../model/resnet50/build/head.map.plan"
map_score_threshold: 0.5
map_classes:
  - "drivable_area"
  - "ped_crossing"
  - "walkway"
  - "stop_line"
  - "carpark_area"
  - "divider"
```

`tool/build_trt_engine.sh` will build `head.map.plan` automatically when
`model/<model-name>/head.map.onnx` exists. The first implementation runs the
detection and map heads through separate `libpybev` calls, so it recomputes the
shared BEV feature. It is functionally useful for integration; for final vehicle
latency, the next optimization is a single C++ forward that shares the fused BEV
feature between bbox and map heads.

For the custom map-only checkpoint trained with
`configs/custom_dataset/seg.yaml`, export to `model/bevfusion_df_seg`:

```bash
export BEVFUSION_DF_ROOT=/workspace/CUDA-BEVFusion/bevfusion-df

python3 qat/export-df-camera.py \
  --bevfusion-root ${BEVFUSION_DF_ROOT} \
  --config ${BEVFUSION_DF_ROOT}/configs/custom_dataset/seg.yaml \
  --ckpt ${BEVFUSION_DF_ROOT}/runs/epoch_21.pth \
  --save-root model/bevfusion_df_seg

python3 qat/export-df-scn.py \
  --config ${BEVFUSION_DF_ROOT}/configs/custom_dataset/seg.yaml \
  --ckpt ${BEVFUSION_DF_ROOT}/runs/epoch_21.pth \
  --save model/bevfusion_df_seg/lidar.backbone.onnx

python3 qat/export-df-mapseg.py \
  --bevfusion-root ${BEVFUSION_DF_ROOT} \
  --config ${BEVFUSION_DF_ROOT}/configs/custom_dataset/seg.yaml \
  --ckpt ${BEVFUSION_DF_ROOT}/runs/epoch_21.pth \
  --save-root model/bevfusion_df_seg

export DEBUG_MODEL=bevfusion_df_seg
export DEBUG_PROFILE=bevfusion_df
export DEBUG_PRECISION=fp16
export USE_Python=ON
source tool/environment.sh
bash tool/build_trt_engine.sh
cd build && cmake .. && make -j && cd ..
```

Then run map-only RSCL inference:

```bash
python3 -m deploy_rscl.rscl_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_seg_rscl.yaml \
  --bag /workspace/mybag.000.rsclbag \
  --max-frames 20 \
  --output-file runs/rsclbag_seg_outputs.jsonl
```

Visualize detections from the generated JSONL on each camera image and on a
lidar BEV image:

```bash
python3 -m deploy_rscl.visualize_detections \
  --adapter-config deploy_rscl/configs/bevfusion_resnet50_rscl.yaml \
  --bag /workspace/mybag.000.rsclbag \
  --detections runs/rsclbag_resnet50_outputs.jsonl \
  --output-dir runs/rsclbag_resnet50_vis \
  --max-frames 20 \
  --camera-line-width 8 \
  --bev-line-width 6 \
  --font-size 30
```

Each output frame directory contains:

- `<camera>_detections.jpg`
- `cameras_mosaic_detections.jpg`
- `lidar_bev_detections.png`
- `detections.json`

## Online RSCL Node

Run directly:

```bash
python3 -m deploy_rscl.rscl_node --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml
```

Before running TensorRT inference on a vehicle, validate online RSCL
subscriptions, decoding and synchronization without loading CUDA-BEVFusion:

```bash
python3 -m deploy_rscl.rscl_node \
  --adapter-config deploy_rscl/configs/bevfusion_resnet50_rscl.yaml \
  --dry-run \
  --print-rate 1
```

Use timestamp-only synchronization when camera decoding is not needed:

```bash
python3 -m deploy_rscl.rscl_node \
  --adapter-config deploy_rscl/configs/bevfusion_resnet50_rscl.yaml \
  --decode-only \
  --print-rate 1
```

Use full image/lidar decoding without TensorRT:

```bash
python3 -m deploy_rscl.rscl_node \
  --adapter-config deploy_rscl/configs/bevfusion_resnet50_rscl.yaml \
  --decode-images-only \
  --print-rate 1
```

Then run online inference:

```bash
python3 -m deploy_rscl.rscl_node \
  --adapter-config deploy_rscl/configs/bevfusion_resnet50_rscl.yaml \
  --print-rate 1
```

Or through mainboard:

```bash
mainboard -d deploy_rscl/configs/bevfusion_rscl.dag
```

## Calibration

`calibration_file` points to `../../calibration.json` by default, which resolves
to `CUDA-BEVFusion/calibration.json` from `deploy_rscl/configs`. Real
inference needs vehicle calibration with either the normalized adapter schema:

- `lidar2ego`
- per-camera `camera2ego`
- per-camera `camera_intrinsics`

or the project calibration schema:

- top-level `lidar2ego`
- top-level camera names matching `camera_order`
- per-camera `cam_intrinsic`
- per-camera `extrinsic`
- optional per-camera `cam_dist`

For the project schema, `calibration_extrinsic_direction` controls how
`extrinsic` is interpreted. The default is `lidar2camera`, matching the current
project calibration file description. If CUDA prints:

```text
Warning: camera geometry produced 0 valid frustum points. Check camera calibration matrices.
```

or `1 valid frustum points`, set:

```yaml
calibration_extrinsic_direction: "camera2lidar"
```

and rebuild/rerun the bag test. This is the fastest way to verify whether the
stored extrinsic direction is opposite of the config.

`undistort_images` defaults to `false`. Set it to `true` only when RSCL camera
frames are raw distorted images and OpenCV is available in the runtime
environment. Leave it `false` if the camera stream is already rectified.
