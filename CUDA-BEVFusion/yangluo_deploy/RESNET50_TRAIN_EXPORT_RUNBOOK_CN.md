# 洋逻港 ResNet50 训练、FP16 ONNX 导出与 Thor 部署复现手册

本文记录已经实际走通的流程。新增内容全部位于 `yangluo_deploy/`，不会替换乘用车
Swin、QAT 或 RSCL 部署路径。

## 1. 执行环境与边界

| 阶段 | 环境 | 已验证配置 |
| --- | --- | --- |
| 训练与 ONNX 导出 | x86 GPU 容器 `bevfusion_train` | Python 3.8.13、PyTorch 1.10.1、CUDA 11.3、NVIDIA L20 |
| ARM64 编译与静态检查 | 容器 `yangluo_cuda_bev` | AArch64、CUDA 13.0、TensorRT 10.13.3、ROS1 Noetic |
| TensorRT Plan 与在线推理 | 真实 DRIVE AGX Thor | 必须有真实 NVIDIA GPU/驱动，不能由 QEMU 代替 |

以下 x86 命令默认训练工程位于 `/home/psq/bevfusion`，并在容器中执行：

```bash
sudo docker exec -it -w /home/psq/bevfusion bevfusion_train bash
```

## 2. 固定的数据和模型合同

- 前视、后视共两路相机；训练配置中的相机顺序必须与部署顺序一致。
- 部署 topic 为相机 0（前视）和相机 5（后视）。
- 网络输入图像为 `256x704`。
- 点特征为 `[x, y, z, intensity, 0]`，第五维不是点时间戳。
- 当前模型为 11 类，顺序如下：

```text
truck
car
traffic_cone
crane_tyre
quay_crane
agv
crane_spreader
tricyclist
fence
forklift
pedestrian
```

类别数量或顺序改变时，必须一起修改训练数据/配置、重新训练、重新导出 detection head，
并同步更新 `yangluo_deploy/configs/classes.txt`。不能只修改类别文本文件。

## 3. 准备 ResNet50 预训练权重

将官方 ImageNet ResNet50 权重放到：

```text
/home/psq/bevfusion/pretrained/resnet50-0676ba61.pth
```

校验：

```bash
cd /home/psq/bevfusion
sha256sum pretrained/resnet50-0676ba61.pth
```

正确 SHA256：

```text
0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a
```

该权重只初始化相机 ResNet50 backbone。旧 Swin 模型的 `epoch_100.pth` 不参与训练。

## 4. 生成独立的 ResNet50 配置

```bash
cd /home/psq/bevfusion

python yangluo_deploy/training/make_resnet50_config.py \
  --input yangluo_adapter/demo_overfit.yaml \
  --output yangluo_deploy/generated/demo_overfit_resnet50.yaml \
  --pretrained /home/psq/bevfusion/pretrained/resnet50-0676ba61.pth
```

检查关键字段：

```bash
grep -nE 'load_from:|resume_from:|type: ResNet|depth: 50|out_indices:|in_channels:|checkpoint:|max_epochs:' \
  yangluo_deploy/generated/demo_overfit_resnet50.yaml
```

必须满足：

```text
load_from: null
resume_from: null
type: ResNet
depth: 50
out_indices: [1, 2, 3]
in_channels: [512, 1024, 2048]
```

`load_from` 和 `resume_from` 均为 `null`，表示除 ResNet50 ImageNet 初始化外，BEVFusion
其余部分从头训练。

## 5. 训练前检查

至少确认：

```bash
cd /home/psq/bevfusion

python -c '
from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval
path = "yangluo_deploy/generated/demo_overfit_resnet50.yaml"
configs.load(path, recursive=True)
cfg = Config(recursive_eval(configs), filename=path)
assert cfg.load_from is None
assert cfg.resume_from is None
assert cfg.model.encoders.camera.backbone.type == "ResNet"
assert cfg.model.encoders.camera.backbone.depth == 50
assert cfg.model.encoders.camera.neck.in_channels == [512, 1024, 2048]
assert len(cfg.object_classes) == cfg.model.heads.object.num_classes
print("RESNET50_CONFIG_OK classes=", len(cfg.object_classes))
'
```

本次实测还完成了 dataset/model 构建检查：训练集 118 帧、验证集 118 帧、两路图像
张量 `[2,3,256,704]`、点张量 `[60000,5]`。加载 ImageNet 权重时出现
`unexpected key: fc.weight, fc.bias` 是正常现象，因为分类层不会用于 BEVFusion。

## 6. 启动训练

```bash
cd /home/psq/bevfusion

torchpack dist-run -np 1 python tools/train.py \
  yangluo_deploy/generated/demo_overfit_resnet50.yaml \
  --run-dir yangluo_runs/yangluo_resnet50_118f_from_scratch_v1
```

为了防止 SSH 断开，可在宿主机后台启动并记录退出码：

```bash
sudo docker exec -d bevfusion_train bash -lc '
cd /home/psq/bevfusion
run_dir=yangluo_runs/yangluo_resnet50_118f_from_scratch_v1
mkdir -p "$run_dir"
torchpack dist-run -np 1 python tools/train.py \
  yangluo_deploy/generated/demo_overfit_resnet50.yaml \
  --run-dir "$run_dir" \
  >"$run_dir/train.log" 2>&1
printf "%s\n" "$?" >"$run_dir/train.status"
'
```

查看状态：

```bash
sudo docker exec bevfusion_train bash -lc '
cd /home/psq/bevfusion
run_dir=yangluo_runs/yangluo_resnet50_118f_from_scratch_v1
if [ -f "$run_dir/train.status" ]; then
  printf "exit_code="; cat "$run_dir/train.status"
else
  echo RUNNING
fi
tail -n 40 "$run_dir/train.log" 2>/dev/null || true
'
```

训练成功条件：`exit_code=0` 且存在非空 `epoch_100.pth`。本次检查点为：

```text
/home/psq/bevfusion/yangluo_runs/yangluo_resnet50_118f_from_scratch_v1/epoch_100.pth
SHA256: 6cea657d8f6a52ba0b3e9ba21fbf42f66692d78277e2fc85a04663076ba25718
```

本次训练/验证使用相同的 118 帧，最终 mAP 约 0.52；它只能证明流程和拟合能力，不能
代表独立数据泛化效果。后续拿到更多数据后需要重新划分 train/val 并重新训练。

## 7. 离线准备 ONNX 包

不要升级训练环境中的 PyTorch、CUDA、MMCV、TorchPack、NumPy 或 Protobuf。将匹配
Python 3.8 x86_64 的 ONNX wheel 上传到服务器，然后隔离安装：

```bash
cd /home/psq/bevfusion

python -m pip install --no-index --no-deps \
  --target yangluo_deploy/python_packages/onnx_1_16_2 \
  /home/psq/onnx-1.16.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

export PYTHONPATH=/home/psq/bevfusion/yangluo_deploy/python_packages/onnx_1_16_2${PYTHONPATH:+:$PYTHONPATH}
python -c 'import onnx; print(onnx.__version__)'
```

预期 ONNX 版本为 `1.16.2`。FP16 阶段不需要 `pytorch-quantization`，也不强制安装
ONNX Runtime 或 ONNX Simplifier。

## 8. 导出五个 FP16 ONNX

每次新模型使用新的输出目录，避免覆盖上一个版本：

```bash
cd /home/psq/bevfusion

export PYTHONPATH=/home/psq/bevfusion/yangluo_deploy/python_packages/onnx_1_16_2${PYTHONPATH:+:$PYTHONPATH}

python yangluo_deploy/export/export_fp16.py \
  --bevfusion-root /home/psq/bevfusion \
  --config /home/psq/bevfusion/yangluo_deploy/generated/demo_overfit_resnet50.yaml \
  --ckpt /home/psq/bevfusion/yangluo_runs/yangluo_resnet50_118f_from_scratch_v1/epoch_100.pth \
  --save-root /home/psq/bevfusion/model/yangluo_resnet50_fp16 \
  --num-cameras 2
```

必须生成以下五个非空文件：

```text
camera.backbone.onnx
camera.vtransform.onnx
lidar.backbone.xyz.onnx
fuser.onnx
head.bbox.onnx
```

快速检查：

```bash
find /home/psq/bevfusion/model/yangluo_resnet50_fp16 \
  -maxdepth 1 -type f -name "*.onnx" -printf "%f %s bytes\n" | sort
```

已验证的张量合同：

- `camera.backbone`: `img [1,2,3,256,704]` 和 `depth [1,2,1,256,704]`；
- `camera.vtransform`: `[1,80,128,128] -> [1,80,64,64]`；
- `lidar.backbone.xyz`: `[1,5] -> [1,256,64,64]`；
- `fuser`: camera `[1,80,64,64]`、lidar `[1,256,64,64]`，输出 `[1,512,64,64]`；
- `head.bbox`: 输入 `[1,512,64,64]`，输出 `score/rot/dim/reg/height/vel`。

`camera.backbone`、`camera.vtransform`、`fuser`、`head.bbox` 应通过标准 ONNX checker。
`lidar.backbone.xyz.onnx` 使用 CUDA-BEVFusion 自定义 SparseConvolution 图，由 `libspconv`
解析，不能用标准 ONNX checker 的失败判断它无效，也不要为通过标准 checker 而改写它。

## 9. 模型类别与文件校验

导出后必须同时确认：

1. 配置 `object_classes` 数量；
2. checkpoint detection head 类别数量；
3. `head.bbox.onnx` 中 `classes_eye` 形状；
4. `classes.txt` 的行数和顺序。

当前四项均为 11 类，`classes_eye` 为 `[11,11]`。打包前执行：

```bash
wc -l yangluo_deploy/configs/classes.txt
sha256sum model/yangluo_resnet50_fp16/*.onnx
```

### 9.1 使用训练检查点进行可视化

原始 `tools/visualize.py` 保留给乘用车数据，不要直接修改。洋逻数据的相机标定集中
存放在数据集级 `config/camera_calibration.json`，使用独立入口：

```bash
torchpack dist-run -np 1 python \
  yangluo_deploy/visualization/visualize_resnet50.py \
  yangluo_deploy/generated/demo_overfit_resnet50.yaml \
  --mode pred \
  --checkpoint yangluo_runs/yangluo_resnet50_118f_from_scratch_v1/epoch_100.pth \
  --split val \
  --bbox-score 0.05 \
  --out-dir vis_yangluo/vis_detect
```

如果脚本不能从图像路径向上找到标定文件，追加：

```bash
--camera-calibration /absolute/path/to/camera_calibration.json
```

该入口只在当前进程替换标定加载和洋逻框体 Z 原点兼容逻辑，不修改乘用车可视化。

## 10. 标定和 ARM64 部署

运行时标定由原始相机 0/5 参数生成：

```bash
python yangluo_deploy/tools/convert_calibration.py \
  --input yangluogang/camera_calibration.json \
  --output yangluo_deploy/configs/calibration_runtime.json \
  --front-id 0 --rear-id 5
```

比较两份 JSON 时应解析后比较内容，不要用 `cmp` 比较缩进和换行。

最终可迁移 Thor 包为：

```text
yangluo_thor_resnet50_fp16_v2_portable.tar.gz
SHA256: 14197f00072febb26f1f4b687ff819ffbe6c6c165bf94d74ece278d7d38558a7
```

包可以解压到任意目录。真实 Thor 容器内依次执行：

```bash
cd /path/to/extracted/bundle
sha256sum -c CUDA-BEVFusion/yangluo_deploy/generated/thor_bundle_SHA256SUMS
bash CUDA-BEVFusion/yangluo_deploy/tools/build_thor_fp16.sh
bash CUDA-BEVFusion/yangluo_deploy/tools/run_thor_fp16.sh --check-only
bash CUDA-BEVFusion/yangluo_deploy/tools/run_thor_fp16.sh
```

`build_thor_fp16.sh` 在真实 Thor 上生成并反序列化验证四个 TensorRT FP16 Plan。
LiDAR sparse backbone 继续使用 ONNX + `libspconv.so`，不生成 TensorRT Plan。

ROS 输出验证：

```bash
rostopic type /perception/bevfusion/objects
rostopic echo -n 1 /perception/bevfusion/objects
rostopic hz /perception/bevfusion/objects
```

期望类型为 `yangluo_bevfusion_msgs/DetectedObjectArray`。

## 11. 后续新数据/新类别的最短重复流程

1. 更新数据 PKL 和训练配置中的类别列表；
2. 重新运行 `make_resnet50_config.py`；
3. 使用新的 `--run-dir` 训练，不加载旧 Swin checkpoint；
4. 使用新的 checkpoint 和新的 ONNX 输出目录重新导出；
5. 校验配置、checkpoint、ONNX head 和 `classes.txt` 的类别数量/顺序；
6. 生成新模型包，不覆盖当前可回滚版本；
7. 在真实 Thor 上重新生成 TensorRT Plan；Plan 不能跨模型或 TensorRT/GPU 环境复用；
8. 先运行 `--check-only`，再启动 ROS，并用 bag/实车检查输出 topic。

## 12. 常见错误

- **误加载旧模型**：确认 `load_from: null`、`resume_from: null`。
- **类别顺序不一致**：训练配置、checkpoint head、ONNX head、`classes.txt` 必须一致。
- **直接复用旧 Plan**：任何 ONNX、TensorRT、CUDA 或 GPU 环境变化后都应重新构建 Plan。
- **在 QEMU 中生成 Plan**：无真实 GPU 时只能编译和静态检查，不能生成可信 TensorRT Plan。
- **把 CUDA stub 带到实车**：真实 Thor 的 `LD_LIBRARY_PATH` 中不得包含
  `/tmp/yangluo_cuda_driver_stub`。
- **把同一 118 帧 mAP 当成泛化指标**：它只用于本阶段打通部署流程。
