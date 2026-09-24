# CUDA-BEVFusion RSCL/Thor 适配说明与原版差异

## 1. 文档目的

本文说明当前项目与 NVIDIA 开源 CUDA-BEVFusion 的主要区别，并总结本人在 RSCL 接入和 NVIDIA DRIVE Thor 平台部署方面完成的工程工作。

本项目不是重新设计一套 BEVFusion 网络，而是在 NVIDIA CUDA-BEVFusion 高性能推理框架基础上，面向实际车辆数据、RSCL 通信框架和 Thor 异构计算平台完成的一套工程化适配。其核心价值是将原版以离线样例推理为主的程序，扩展为能够接收车端多传感器数据、完成在线同步与预处理、执行 TensorRT 推理并通过 RSCL 发布结果的完整链路。

本文以仓库中的 `origin/master`（NVIDIA 原版基线，提交 `6dfdcd2`）和当前 `main`（提交 `e16b69b`）进行对比。由于 NVIDIA 上游仍可能继续更新，后续如同步上游代码，应重新核对差异。

## 2. 总体结论

NVIDIA 原版 CUDA-BEVFusion 主要解决的是 BEVFusion 模型的高性能 CUDA/TensorRT 推理问题，提供相机、LiDAR、BEV 特征融合和 3D 目标检测等核心计算模块，以及基于样例数据的 C++/Python 推理接口。

本项目保留了原版的核心推理框架，主要新增和改造了以下内容：

1. 接入 RSCL 通信与 rsclbag 数据，支持六路相机和一路 LiDAR 的在线订阅及离线回放。
2. 实现以 LiDAR 为时间基准的多传感器同步，支持多种时间戳来源、逐传感器时间补偿和同步诊断。
3. 实现 RSCL 相机、LiDAR 消息的解析，以及 H.264/H.265 图像解码和 Cap'n Proto LiDAR 快速解析。
4. 按训练数据处理方式完成六路相机在线去畸变、缩放、裁剪、归一化和标定矩阵构造。
5. 增加面向自定义 `bevfusion_df` 模型的输入尺寸、点云范围、体素参数和模型导出适配。
6. 在目标检测之外增加可选的 BEV 地图分割推理能力。
7. 增加 C++ RSCL 离线 runner、在线 node、RawMessage 结果发布、JSONL 落盘和可视化工具。
8. 完成 Thor AArch64 交叉编译、目标依赖选择、TensorRT plan/plugin 适配、部署打包和分阶段诊断。

因此，与原版相比，本项目的主要差异不是单一 CUDA 算子的修改，而是从“模型推理样例”到“车端在线感知组件”的系统化扩展。

## 3. 原版与本项目的差异对比

| 对比项 | NVIDIA 开源 CUDA-BEVFusion | 本项目 |
| --- | --- | --- |
| 主要定位 | CUDA/TensorRT 加速的 BEVFusion 推理实现 | 面向 RSCL 和 Thor 的车端多传感器在线推理系统 |
| 数据输入 | 已准备好的样例图片、Tensor 文件或 Python 数组 | 六路 RSCL 编码相机流、RSCL LiDAR 消息、rsclbag 回放 |
| 运行方式 | 单帧/样例数据推理为主 | 离线 rsclbag 连续推理和在线 RSCL 节点持续运行 |
| 多传感器同步 | 默认输入已经完成帧级配对 | 以 LiDAR 为锚点，从各相机队列选择最近帧，并进行容差判断 |
| 时间处理 | 不负责车端不同消息时间基准的统一 | 支持 payload/header/receive 时间戳选择、相机独立 offset、LiDAR offset 和同步调试 |
| 相机解码 | 读取 JPEG 或预处理 Tensor | 支持 RSCL RawMessage 以及 FFmpeg H.264/H.265 连续解码 |
| LiDAR 解码 | 直接读取浮点/半精度点云 Tensor | 支持 RawMessage、紧凑点格式和 `AdsfiLidarPointCloud` Cap'n Proto 消息解析 |
| 相机预处理 | 使用原版固定样例输入和归一化流程 | 增加六相机在线去畸变、remap 缓存、resize/crop/normalize 和运行日志标记 |
| 标定输入 | 从样例 Tensor 读取投影矩阵 | 从 `calibration.json` 读取内外参和畸变参数，在线生成模型需要的矩阵 |
| 模型配置 | 以 nuScenes 的 resnet50、resnet50int8、swint 等配置为主 | 增加 `bevfusion_df`/分割模型 profile、显式 plan/ONNX 路径和 YAML 运行时切换 |
| 感知任务 | 主要为 3D 目标检测 | 支持 3D 目标检测，并增加可选 BEV 地图分割 head |
| 输出方式 | 控制台结果、样例可视化图片或 Python 返回值 | RSCL `RawMessage` JSON 发布、JSONL 文件输出及离线可视化 |
| 目标平台 | 原版构建流程主要面向 x86_64 CUDA 环境 | 增加 Thor AArch64 工具链、CUDA 12.8、TensorRT、spconv、Protobuf、FFmpeg、OpenCV 和 RSCL 依赖适配 |
| 部署能力 | 未提供面向本车端环境的完整部署包 | 提供交叉编译、依赖收集、打包、动态库检查和在线启动脚本/说明 |
| 调试能力 | 模型加载、推理耗时等基础日志 | 增加 decode-only、decode-images-only、sync-debug、阶段日志、输出落盘和错误边界定位 |

## 4. RSCL 数据链路适配

### 4.1 在线输入与输出

本项目增加了 `rscl_bevfusion_online_node`，按照 YAML 配置订阅六路相机和一路 LiDAR。当前检测配置中的典型话题为：

- 相机：`/sensor/camera/*/encode` 六路编码视频流；
- LiDAR：`/perception/lidar/preproc_points_cloud`；
- 输出：`/perception/bevfusion/objects`。

在线节点通过 RSCL backend 创建订阅者和发布者。相机消息经独立解码器转换为 RGB 图像；LiDAR 消息优先使用 `AdsfiLidarPointCloud` 的强类型 Cap'n Proto 路径提取时间戳、点步长、点数量和数据区，从而避免通用动态反射带来的额外开销和阻塞风险。

推理结果封装为 `RawMessage` JSON。目标检测输出保留类别编号、置信度和九维框参数，结构示例如下：

```json
{
  "objects": [
    {
      "label": 0,
      "score": 0.91,
      "box": [1.2, 3.4, 0.6, 1.8, 4.2, 1.6, 0.1, 0.0, 0.0]
    }
  ],
  "timestamp_us": 1785162991000000
}
```

如果启用地图分割，消息还可包含类别、形状、阈值和 `base64_uint8_chw` 编码的 BEV mask。

### 4.2 rsclbag 离线推理

本项目增加了 `rscl_bevfusion_bag_runner`，使相同的解码、同步、预处理和推理链路能够在 rsclbag 上离线运行。离线 runner 可以用于：

- 检查 bag 是否能正常打开以及目标话题是否存在；
- 只验证时间戳和多传感器同步；
- 验证六路视频解码；
- 执行完整 TensorRT 推理；
- 将结果保存为 JSONL，供统计和可视化工具使用。

离线和在线路径共用 C++ adapter core，减少了“离线验证通过但在线处理逻辑不同”的风险。

## 5. 多传感器时间同步

NVIDIA 原版假定送入模型的六路相机和点云已经对齐，不负责中间件消息队列中的时间同步。本项目新增了 `FrameSynchronizer`，其策略如下：

1. 将 LiDAR 帧作为同步锚点。
2. 六路相机分别维护有限长度队列，避免某一路阻塞其他相机的接收和解码。
3. 对每个 LiDAR 时间戳，从每个相机队列中选择时间差绝对值最小的图像。
4. 只有六路相机与 LiDAR 的时间差都不超过 `sync_tolerance_ms` 时，才生成完整同步帧。
5. 支持 `camera_time_offsets_ms` 和 `lidar_time_offset_ms`，用于补偿不同传感器或链路的固定延迟。
6. 支持 payload、RSCL header、receive 等时间戳来源，以适配实时运行和 bag 回放的不同情况。

同步诊断会输出 `missing_camera`、`outside_tolerance`、每路队列长度、最近时间差和建议 offset。这些信息可用于区分“话题没有数据”“视频尚未解出关键帧”和“时间偏差超限”等不同故障。

需要注意：仓库 YAML 中的同步参数只是当前版本的默认值；车端实际参数属于车辆标定和数据链路配置，应以实测日志为准，不应在更新程序时覆盖已经调好的车端参数。

## 6. 图像解码、去畸变与模型预处理

原版程序主要读取已准备好的图像或 Tensor。本项目为了处理车端原始编码流，增加了以下链路：

```text
RSCL 编码相机消息
        ↓
FFmpeg H.264/H.265 解码
        ↓
OpenCV 相机去畸变
        ↓
resize / crop / normalize
        ↓
六相机 N×3×H×W 模型输入
```

去畸变使用 OpenCV rational pinhole 模型，畸变参数顺序为 `k1,k2,p1,p2,k3,k4,k5,k6`。程序使用 `initUndistortRectifyMap` 构建映射表，并按相机和输入分辨率缓存 remap，避免每帧重复计算。去畸变在 resize、crop 和 normalize 之前执行，以保持与训练数据处理流程一致。

如果 YAML 设置 `undistort_images: true`，但程序没有链接目标平台的 OpenCV core/imgproc/calib3d，程序会明确报错，而不是静默使用未经校正的图像。

## 7. 自定义模型与推理核心改造

### 7.1 `bevfusion_df` 模型 profile

在保留原版模型 profile 的基础上，本项目增加了面向自定义模型的参数分支，包括：

- 原始相机尺寸按 1920×1080 处理；
- 网络图像输入为 256×704；
- 点云范围调整为 `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]`；
- 体素大小调整为 `[0.2, 0.2, 0.2]`；
- BEV 几何范围、网格尺寸、检测阈值和模型路径随 profile 配置；
- 模型文件可由 `cuda_model_root` 统一确定，也可在 YAML 中逐项覆盖。

同时增加了自定义模型的相机 backbone、视角变换、LiDAR backbone、融合模块、检测 head 和地图分割 head 的 ONNX 导出辅助脚本。

### 7.2 可选 BEV 地图分割

原版核心流程默认在融合特征后进入 3D 检测 head。本项目对推理核心进行了拆分，使融合后的 BEV feature 可以根据配置进入：

- `head.bbox`：输出 3D 目标检测结果；
- `head.map`：输出 BEV 地图语义分割结果。

目标检测和地图分割可分别启用。`tool/build_trt_engine.sh` 会按 ONNX 文件是否存在生成相应 TensorRT plan，并在构建检测 head 时加载自定义 LayerNorm plugin。

## 8. Thor AArch64 适配

### 8.1 构建系统改造

NVIDIA 原版构建配置中包含较多 x86_64 环境假设。本项目在 CMake 和构建脚本中增加了目标架构相关参数，主要包括：

- 使用 `cmake/toolchains/aarch64-thor.cmake` 或 SenseAuto Thor toolchain；
- 区分 host 工具与 target 头文件、库文件；
- 显式设置 AArch64 CUDA、TensorRT、spconv 和 Protobuf 路径；
- 支持 RSCL、FFmpeg、OpenCV 等车端依赖的目标库选择；
- 避免将 x86 库、AArch64 库和不同 CUDA 版本的产物混入同一构建目录；
- 构建 `bevfusion`、`libbevfusion_core.so`、`libcustom_layernorm.so`、bag runner、online node 和 RSCL backend。

`deploy_rscl/cpp/build_aarch64_thor.sh` 是当前 Thor 交叉编译入口，`package_aarch64_thor.sh` 用于收集可执行文件、项目动态库、spconv、Protobuf、FFmpeg、配置和模型文件，形成车端部署目录。

### 8.2 TensorRT plan 与自定义插件

TensorRT plan 与生成它的 TensorRT 版本、GPU 架构和插件实现紧密相关。因此，本项目对以下问题进行了专门适配：

- 相机 backbone、view transform、fuser 和 bbox head 的 plan 分别构建和验证；
- `head.bbox` 构建时加载 `libcustom_layernorm.so`；
- `lidar.backbone.xyz.onnx` 由 spconv 运行时加载，不按普通 TensorRT ONNX plan 处理；
- AArch64 交叉链接可使用 target stub，但车端运行时必须加载真实 CUDA/TensorRT 库；
- plan 应在 Thor 或完全一致的 TensorRT/CUDA/GPU 环境中生成并执行 load test，不能仅凭 x86 主机编译成功判断可用。

### 8.3 部署与可观测性

本项目增加了车端所需的部署结构、环境变量示例、`ldd` 检查、日志落盘、后台运行和结果探针说明。运行时可按以下阶段逐步验证：

1. 静态检查：确认二进制为 AArch64 ELF，依赖和 RPATH 符合预期。
2. 车端动态库检查：使用 `ldd` 确认不存在 `not found`。
3. `--decode-only`：检查订阅、消息接收和时间同步。
4. `--decode-images-only`：检查六路视频解码与同步，不加载模型。
5. 完整离线推理：检查预处理、TensorRT、结果 JSONL。
6. 完整在线推理：检查持续同步、目标输出和 RSCL 发布。

## 9. 当前端到端流程

```text
六路 RSCL 编码相机                         RSCL LiDAR
        │                                      │
        ├── FFmpeg 解码                        ├── RawMessage/Cap'n Proto 解析
        │                                      │
        └────────────── 时间戳校正与最近帧同步 ─┘
                               │
                  标定读取与六相机在线去畸变
                               │
                   resize / crop / normalize
                               │
               Camera Tensor + LiDAR Point Tensor
                               │
               CUDA-BEVFusion / TensorRT / spconv
                        ┌──────┴──────┐
                    3D 目标检测    BEV 地图分割（可选）
                        └──────┬──────┘
                               │
                    JSON / JSONL / RSCL RawMessage
```

## 10. 已验证范围与边界

根据当前仓库代码、提交记录和既有 Thor 调试记录，已经完成的验证包括：

- RSCL 六相机和 LiDAR 数据接入；
- rsclbag 离线同步、视频解码和完整推理；
- 六路相机去畸变进入模型预处理链路；
- AArch64 交叉编译和 Thor 部署包运行；
- Thor TensorRT 10.10.10 plan 加载；
- `libcustom_layernorm.so`、`libbevfusion_core.so` 和 AArch64 CUDA 12.8 `libspconv.so` 的目标侧使用；
- 在线节点持续输出检测结果；
- `/perception/bevfusion/objects` 的 RSCL RawMessage 发布和独立节点接收。

上述结论仅适用于已经验证过的 Thor 软件栈、模型 plan、动态库和车辆配置。以下内容仍需在每次换模型、升级 TensorRT/CUDA、修改标定或更换车辆后重新确认：

- TensorRT plan 与目标 TensorRT 版本、GPU SM 和 plugin 是否一致；
- 六路相机顺序、内外参和畸变参数是否与训练配置一致；
- LiDAR 点格式、point step 和坐标系约定是否一致；
- 实际车辆的时间 offset 和同步容差是否仍然有效；
- 下游是否接受当前私有 RawMessage JSON 结构，或需要转换为公司标准障碍物消息。

## 11. 主要代码位置

| 模块 | 主要路径 | 作用 |
| --- | --- | --- |
| RSCL 配置 | `deploy_rscl/configs/*.yaml` | 话题、模型、同步、预处理、标定和输出配置 |
| C++ adapter 接口 | `deploy_rscl/cpp/include/rscl_adapter/` | 数据结构、同步、解码、预处理和 runner 接口 |
| 消息解码 | `deploy_rscl/cpp/src/codecs.cpp` | 相机、LiDAR、RawMessage 和结果 JSON 编解码 |
| 时间同步 | `deploy_rscl/cpp/src/sync.cpp` | LiDAR 锚定的六相机最近帧同步 |
| 图像与标定预处理 | `deploy_rscl/cpp/src/preprocess.cpp` | 标定加载、去畸变、缩放、裁剪和归一化 |
| CUDA-BEVFusion 调用 | `deploy_rscl/cpp/src/runner.cpp` | 模型加载、Tensor 构造、检测/分割推理 |
| 离线 runner | `deploy_rscl/cpp/src/rscl_bag_runner.cpp` | rsclbag 连续帧推理 |
| 在线 node | `deploy_rscl/cpp/src/rscl_online_node.cpp` | 在线订阅、同步、推理、发布和日志 |
| RSCL SDK glue | `deploy_rscl/cpp/vehicle_rscl_online_node_ad.cpp` | RSCL subscriber/publisher 和强类型 LiDAR 解析 |
| Thor 构建 | `deploy_rscl/cpp/build_aarch64_thor.sh` | Thor AArch64 交叉编译 |
| Thor 打包 | `deploy_rscl/cpp/package_aarch64_thor.sh` | 生成车端部署目录 |
| 推理核心扩展 | `src/bevfusion/bevfusion.*`、`src/bevfusion/head-map.*` | 融合特征复用和地图分割 head |
| 模型导出 | `qat/export-df-*.py` | 自定义检测/分割模型 ONNX 导出 |
| TensorRT 构建 | `tool/build_trt_engine.sh` | 检测、分割 plan 和 LayerNorm plugin 构建 |

## 12. 个人工作总结表述

可将本项目中的个人工作概括为：

> 基于 NVIDIA 开源 CUDA-BEVFusion，完成面向车辆 RSCL 中间件和 NVIDIA DRIVE Thor 平台的工程化适配。实现六路编码相机与 LiDAR 的 RSCL 在线订阅、rsclbag 离线回放、LiDAR 锚定的多传感器时间同步、FFmpeg 视频解码、相机在线去畸变与标定预处理、TensorRT/spconv 推理、检测与可选 BEV 地图分割结果发布；同时完成 Thor AArch64 交叉编译、TensorRT plan 与自定义插件适配、依赖打包、分阶段诊断和车端端到端验证，使原版离线样例推理程序具备在 Thor 车端持续运行的能力。

