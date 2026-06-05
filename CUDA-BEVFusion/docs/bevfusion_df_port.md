# bevfusion_df FP16 Port

This note records the first-stage path for exporting and running the 21-class
`bevfusion_df` detection model with CUDA-BEVFusion.

## Scope

- Model: `H:\df_code\bevfusion\configs\custom_dataset\default.yaml`
- Checkpoint: `H:\df_code\bevfusion\runs\detect\epoch_2.pth`
- Target precision: FP16
- Task: 21-class 3D object detection
- RSCL integration is intentionally out of scope for this stage.

The raw camera images in `data/clip_S31a_1_20251007211500/camera` have mixed
resolutions. The first-stage runtime path therefore uses preprocessed
`images.tensor` and calls `forward_no_normalize`, matching the PyTorch data
pipeline more closely.

## Export ONNX

Run from `CUDA-BEVFusion`:

```bash
export BEVFUSION_DF_ROOT=/path/to/bevfusion

python qat/export-df-camera.py \
  --bevfusion-root "$BEVFUSION_DF_ROOT" \
  --config "$BEVFUSION_DF_ROOT/configs/custom_dataset/default.yaml" \
  --ckpt "$BEVFUSION_DF_ROOT/runs/detect/epoch_2.pth" \
  --save-root model/bevfusion_df

python qat/export-df-transfuser.py \
  --bevfusion-root "$BEVFUSION_DF_ROOT" \
  --config "$BEVFUSION_DF_ROOT/configs/custom_dataset/default.yaml" \
  --ckpt "$BEVFUSION_DF_ROOT/runs/detect/epoch_2.pth" \
  --save-root model/bevfusion_df

python qat/export-df-scn.py \
  --bevfusion-root "$BEVFUSION_DF_ROOT" \
  --config "$BEVFUSION_DF_ROOT/configs/custom_dataset/default.yaml" \
  --ckpt "$BEVFUSION_DF_ROOT/runs/detect/epoch_2.pth" \
  --save model/bevfusion_df/lidar.backbone.onnx
```

Expected outputs:

```text
model/bevfusion_df/camera.backbone.onnx
model/bevfusion_df/camera.vtransform.onnx
model/bevfusion_df/fuser.onnx
model/bevfusion_df/head.bbox.onnx
model/bevfusion_df/lidar.backbone.xyz.onnx
```

## Build TensorRT Engines

Set:

```bash
export DEBUG_MODEL=bevfusion_df
export DEBUG_PROFILE=bevfusion_df
export DEBUG_PRECISION=fp16
```

Then build:

```bash
bash tool/build_trt_engine.sh
```

Expected TensorRT outputs:

```text
model/bevfusion_df/build/camera.backbone.plan
model/bevfusion_df/build/camera.vtransform.plan
model/bevfusion_df/build/fuser.plan
model/bevfusion_df/build/head.bbox.plan
```

## Dump A Test Sample

Run from `CUDA-BEVFusion`:

```bash
python tool/dump-df-data.py \
  --bevfusion-root "$BEVFUSION_DF_ROOT" \
  --config "$BEVFUSION_DF_ROOT/configs/custom_dataset/default.yaml" \
  --out dump_df \
  --count 1
```

This creates:

```text
dump_df/00000/images.tensor
dump_df/00000/points.tensor
dump_df/00000/camera_intrinsics.tensor
dump_df/00000/camera2lidar.tensor
dump_df/00000/lidar2image.tensor
dump_df/00000/img_aug_matrix.tensor
```

## Run C++ Inference

Set:

```bash
export DEBUG_MODEL=bevfusion_df
export DEBUG_PROFILE=bevfusion_df
export DEBUG_PRECISION=fp16
export DEBUG_DATA=dump_df/00000
```

Then run:

```bash
bash tool/run.sh
```

When `images.tensor` is present, `src/main.cpp` skips raw image normalization
and calls `forward_no_normalize`.

## Run Python Inference

Build with:

```bash
export USE_Python=ON
bash tool/run.sh
```

Then:

```bash
python tool/pybev.py
```

`tool/pybev.py` automatically uses `images.tensor` when present.

## Current Limits

- INT8/PTQ is not enabled for this profile yet.
- The existing PTQ camera quantization logic assumes a ResNet-style backbone;
  the `bevfusion_df` profile uses SwinTransformer.
- Raw mixed-resolution image normalization is not implemented in C++ for this
  profile. Use `images.tensor` first.
- The segmentation config is not covered; this path is detection only.
