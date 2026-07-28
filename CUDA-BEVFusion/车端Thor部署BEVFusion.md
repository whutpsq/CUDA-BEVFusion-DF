# 车端 Thor 部署 BEVFusion

## 1. 车端目录

部署根目录：

```text
/ota/qinghua/bevfusion/package_aarch64_thor
├── bin/
│   ├── rscl_bevfusion_online_node
│   └── rscl_bevfusion_bag_runner
├── lib/
│   ├── libbevfusion_core.so
│   ├── libcustom_layernorm.so
│   ├── libspconv.so
│   ├── librscl_online_backend_ad.so
│   └── FFmpeg 等依赖库
├── model/bevfusion_df/
│   ├── camera.backbone.onnx
│   ├── camera.vtransform.onnx
│   ├── fuser.onnx
│   ├── head.bbox.onnx
│   ├── lidar.backbone.xyz.onnx
│   └── build/*.plan
├── deploy_rscl/configs/bevfusion_rscl.yaml
├── calibration.json
└── run_env.sh
```

已验证环境：

```text
架构：AArch64
CUDA：12.8
TensorRT：10.10.10
Thor CUDA SM：sm_101
```

## 2. 环境变量

`run_env.sh` 推荐内容：

```bash
#!/usr/bin/env bash

export PACKAGE_ROOT=/ota/qinghua/bevfusion/package_aarch64_thor

export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_Lib="$CUDA_HOME/targets/aarch64-linux/lib"

export TensorRT_Lib=/usr/lib/aarch64-linux-gnu

export SENSEAUTO_ROOT=/opt/senseauto/senseauto-3rdparty/1.20260227-150511-aba9d.thor.default.57df68.l29-base-thor
export SENSEAUTO_3RDPARTY_LIB="$SENSEAUTO_ROOT/3rdparty/lib"
export CAPNP_LIB="$SENSEAUTO_ROOT/3rdparty/capnp/lib"

export RSCL_ONLINE_BACKEND_LIB="$PACKAGE_ROOT/lib/librscl_online_backend_ad.so"
export RSCL_ONLINE_BACKEND_DEEPBIND=0

export CYBER_DOMAIN_ID=123

export LD_LIBRARY_PATH="$PACKAGE_ROOT/lib:$SENSEAUTO_3RDPARTY_LIB:$CAPNP_LIB:$CUDA_Lib:$TensorRT_Lib:/usr/lib/aarch64-linux-gnu/tegra${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

注意：不要在需要被 `source` 的 `run_env.sh` 中加入：

```bash
set -euo pipefail
```

加载环境：

```bash
cd /ota/qinghua/bevfusion/package_aarch64_thor
source ./run_env.sh
```

检查环境：

```bash
echo "$LD_LIBRARY_PATH"
echo "$RSCL_ONLINE_BACKEND_LIB"
echo "$RSCL_ONLINE_BACKEND_DEEPBIND"
echo "$CYBER_DOMAIN_ID"
```

## 3. 正式同步配置

`deploy_rscl/configs/bevfusion_rscl.yaml` 中使用：

```yaml
sync_tolerance_ms: 300.0
camera_time_offsets_ms: [350, 350, -100, -100, -100, 350]
lidar_time_offset_ms: 0
```

六个相机 offset 的顺序必须与 `camera_topics` 保持一致。

说明：

- 当前 `300 ms` 配置用于容忍车端传感器同步误差。
- `500 ms` 只建议用于诊断，过大的时间差会降低相机和点云的空间对应精度。
- 修改 YAML 后必须重启 online node，运行中的进程不会动态重新加载配置。
- 命令行参数会覆盖 YAML 中的配置。

## 4. 启动前动态库检查

```bash
cd /ota/qinghua/bevfusion/package_aarch64_thor
source ./run_env.sh

ldd ./bin/rscl_bevfusion_online_node |
grep 'not found' || echo "online node 动态库完整"

ldd ./lib/librscl_online_backend_ad.so |
grep 'not found' || echo "online backend 动态库完整"

ldd ./bin/rscl_bevfusion_bag_runner |
grep 'not found' || echo "bag runner 动态库完整"
```

三个检查均不应出现 `not found`。

## 5. 正式在线运行

前台运行：

```bash
cd /ota/qinghua/bevfusion/package_aarch64_thor
source ./run_env.sh

./bin/rscl_bevfusion_online_node \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --timestamp-source payload \
  --print-rate 1
```

相机输入：

```text
/sensor/camera/center_camera_fov120/encode
/sensor/camera/left_front_camera/encode
/sensor/camera/right_front_camera/encode
/sensor/camera/rear_camera/encode
/sensor/camera/left_rear_camera/encode
/sensor/camera/right_rear_camera/encode
```

LiDAR 输入：

```text
/perception/lidar/preproc_points_cloud
```

检测输出：

```text
/perception/bevfusion/objects
```

正常日志：

```text
synced_frames=10 ... decode_errors=0 objects=3
```

- `decode_errors=0`：解码正常。
- `objects=N`：该帧输出 N 个目标。
- 使用 `--print-rate 1` 时日志帧号可能跳跃，这是打印限频，不代表中间帧推理失败。
- 刚启动时可能需要等待六路视频的关键帧。

## 6. 后台运行

```bash
cd /ota/qinghua/bevfusion/package_aarch64_thor
source ./run_env.sh

mkdir -p logs run

nohup ./bin/rscl_bevfusion_online_node \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --timestamp-source payload \
  --print-rate 1 \
  > logs/bevfusion_online.log 2>&1 &

echo $! > run/bevfusion_online.pid
echo "PID=$(cat run/bevfusion_online.pid)"
```

查看日志：

```bash
tail -f logs/bevfusion_online.log
```

查看进程：

```bash
ps -p "$(cat run/bevfusion_online.pid)" \
  -o pid,ppid,stat,%cpu,%mem,cmd
```

正常停止：

```bash
kill "$(cat run/bevfusion_online.pid)"
```

## 7. 保存在线推理 JSONL

测试时可以增加：

```text
--output-file /tmp/bevfusion_online.jsonl
```

示例：

```bash
timeout 120s ./bin/rscl_bevfusion_online_node \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --timestamp-source payload \
  --print-rate 0 \
  --output-file /tmp/bevfusion_online.jsonl \
  > /tmp/bevfusion_online.log 2>&1
```

检查结果：

```bash
wc -l /tmp/bevfusion_online.jsonl

head -1 /tmp/bevfusion_online.jsonl |
python3 -m json.tool
```

输出示例：

```json
{
  "objects": [
    {
      "label": 0,
      "score": 0.432129,
      "box": [17.3, 5.45, 0.56, 1.71, 3.62, 1.52, 2.01, 0.17, -0.22]
    }
  ],
  "timestamp_us": 1785162991000000
}
```

`box` 顺序：

```text
[x, y, z, width, length, height, yaw, vx, vy]
```

仓库默认类别中：

```text
label=0 -> car
```

## 8. 分阶段诊断

### 8.1 仅检查原始消息订阅

不解码图像、不运行 TensorRT：

```bash
./bin/rscl_bevfusion_online_node \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --timestamp-source payload \
  --decode-only \
  --message-debug \
  --message-debug-limit 1000 \
  --print-rate 0
```

### 8.2 检查视频解码与同步

解码六路图像和 LiDAR，但不运行 TensorRT：

```bash
./bin/rscl_bevfusion_online_node \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --timestamp-source payload \
  --decode-images-only \
  --sync-debug \
  --sync-debug-limit 200 \
  --print-rate 0
```

### 8.3 临时覆盖同步参数

临时将容差设为 `300 ms`：

```text
--sync-tolerance-ms 300
```

临时覆盖六路相机 offset：

```text
--camera-time-offsets-ms 350,350,-100,-100,-100,350
```

## 9. 离线 RSCL bag 测试

仅测试解码和同步：

```bash
cd /ota/qinghua/bevfusion/package_aarch64_thor
source ./run_env.sh

./bin/rscl_bevfusion_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /ota/qinghua/mybag.000.rsclbag \
  --decode-only \
  --max-frames 20
```

运行完整离线推理：

```bash
./bin/rscl_bevfusion_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /ota/qinghua/mybag.000.rsclbag \
  --max-frames 20 \
  --output-file /tmp/bevfusion_bag_result.jsonl
```

检查结果：

```bash
wc -l /tmp/bevfusion_bag_result.jsonl
head -1 /tmp/bevfusion_bag_result.jsonl | python3 -m json.tool
```

## 10. 验证 RSCL 输出

查看输出频率：

```bash
timeout 10s rscl_channel hz \
  /perception/bevfusion/objects
```

当前输出类型为 `RawMessage`。`rscl_channel echo` 可能显示：

```text
Header Not Supported
```

这是因为当前发布器没有填写 RSCL 原生 header，时间戳保存在 JSON payload 的 `timestamp_us` 字段中。

独立 RawMessage 探针已经验证可以接收到：

```text
payload_bytes=362
payload_timestamp_us=1785162991000000
type=RawMessage
```

下游程序应按 `RawMessage` 获取正文并解析 JSON。

## 11. 常见现象

### `exit_status=124`

使用 `timeout` 后被正常终止，不代表推理失败。应结合 `synced_frames`、`decode_errors` 和 JSONL 行数判断。

### `missing_camera`

至少一路相机的已解码队列为空。增大 `sync_tolerance_ms` 不能解决 `q=0`，需要检查该路数据发布、关键帧和 FFmpeg 解码。

### `outside_tolerance`

六路相机都有帧，但最近帧与 LiDAR 的时间差超过阈值。可以调整：

```yaml
camera_time_offsets_ms
sync_tolerance_ms
```

### `decode_errors=0 objects=N`

表示完整的在线同步、解码和 TensorRT 推理成功。

## 12. 已验证链路

当前已经完成以下端到端验证：

```text
六路相机 + LiDAR
        ↓
Payload 时间戳与 offset 同步
        ↓
FFmpeg 视频解码
        ↓
TensorRT BEVFusion 推理
        ↓
JSON 检测结果
        ↓
RSCL RawMessage 发布
        ↓
独立节点成功接收
```

已验证结果包括：

- TensorRT 10.10.10 四个 plan 可在 Thor 加载。
- `libcustom_layernorm.so` 和 `libbevfusion_core.so` 包含原生 `sm_101` CUDA 代码。
- AArch64 CUDA 12.8 `libspconv.so` 可正常加载。
- 离线 bag runner 可完成同步和推理。
- 在线节点可持续输出检测目标。
- 独立 RawMessage 探针可接收 `/perception/bevfusion/objects`。
