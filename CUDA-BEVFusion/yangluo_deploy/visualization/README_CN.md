# 洋逻港 ResNet50 可视化

不要修改或替换 `tools/visualize.py`。该文件继续服务乘用车数据。

港口数据使用数据集级 `config/camera_calibration.json`，请通过独立入口运行：
当前实车物理对应关系为 `cam5 -> 标定 0`、`cam10 -> 标定 10`；独立入口会按此关系读取参数。

```bash
cd /home/psq/bevfusion

torchpack dist-run -np 1 python \
  yangluo_deploy/visualization/visualize_resnet50.py \
  yangluo_deploy/generated/demo_overfit_resnet50.yaml \
  --mode pred \
  --checkpoint yangluo_runs/yangluo_resnet50_118f_from_scratch_v1/epoch_100.pth \
  --split val \
  --bbox-score 0.05 \
  --out-dir vis_yangluo/vis_detect
```

脚本默认从图像路径逐级向上寻找 `config/camera_calibration.json`。如果数据目录中
没有该文件，可显式指定：

```bash
  --camera-calibration /absolute/path/to/camera_calibration.json
```

首次读取每个相机时应看到 `YANGLUO_VIS_CALIBRATION_OK`。输出目录包括
`camera-0/`、`camera-1/` 和 `lidar/`。
