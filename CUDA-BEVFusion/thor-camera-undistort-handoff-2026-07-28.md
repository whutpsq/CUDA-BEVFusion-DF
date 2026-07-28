# Thor BEVFusion 原始相机去畸变交接（2026-07-28）

## 当前结论

车端六路 `/encode` 输入是原始畸变图像，而检测与分割训练/测试流水线均使用 `LoadMultiViewImageFromFiles(undistort=true)`。原有 C++ 在线节点未去畸变，因此检测和分割都存在训练与车端预处理不一致。检测仍能输出目标不代表输入正确，LiDAR 分支可能掩盖部分相机几何误差；地图分割对相机几何更敏感。

训练端精确处理方式：

```python
cv2.initUndistortRectifyMap(K, D, None, K, image.size, cv2.CV_16SC2)
cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR,
          borderMode=cv2.BORDER_CONSTANT)
```

`D` 顺序为 `k1,k2,p1,p2,k3,k4,k5,k6`，输出继续使用原始内参 `K`。

## 已确认环境

车端 OpenCV：

```text
/opt/senseauto/senseauto-3rdparty/1.20260227-150511-aba9d.thor.default.57df68.l29-base-thor/3rdparty/opencv4
```

交叉环境 OpenCV：

```text
/opt/senseauto/senseauto-3rdparty/1.20260129-021043-e6b2b.pilotmdc-rscl-new.default.4e3eb0.pilotrscl-thor-dev-mp/3rdparty/opencv4
```

两边均为 OpenCV 4.10、SONAME `.410`。交叉环境的 `core/imgproc/calib3d` 已确认是 ARM AArch64。车端存在 `core/imgproc/calib3d/features2d/flann`，`ldd libopencv_calib3d.so.410` 无缺失依赖。

`calibration.json` 的 `front/front_left/front_right/rear/rear_left/rear_right` 均包含完整 pinhole 8 参数，顺序与 `camera_order` 一致。

## 已修改但尚未完成目标验证

本轮修改文件：

```text
deploy_rscl/cpp/include/rscl_adapter/preprocess.hpp
deploy_rscl/cpp/src/preprocess.cpp
CMakeLists.txt
deploy_rscl/cpp/build_aarch64_thor.sh
deploy_rscl/configs/bevfusion_rscl.yaml
deploy_rscl/configs/bevfusion_seg_rscl.yaml
deploy_rscl/cpp/README.md
```

实现内容：读取并验证六路 `cam_dist`；用 OpenCV rational 模型在 resize/crop/normalize 前去畸变；按相机和原始分辨率缓存 `CV_16SC2` map；Thor 构建脚本默认启用 AArch64 OpenCV；检测和分割配置改成 `undistort_images: true`。首次建图日志为：

```text
preprocess_stage=undistort_map_ready camera_index=N width=W height=H
```

已完成 `git diff --check` 和六路标定完整性检查。尚未在交叉环境编译、尚未部署 Thor、尚未验证运行时 OpenCV 依赖和六路建图日志。本 Windows 环境无法启动 Bash/WSL，因此还需在 Linux 交叉环境运行 `bash -n`。

工作树还有此前 Thor/RSCL 调试修改，不要清理、覆盖或重置它们。

## 明天第一步：同步后静态检查

```bash
cd /data1/psq/CUDA-BEVFusion

bash -n deploy_rscl/cpp/build_aarch64_thor.sh

grep -n -E \
  'RSCL_ENABLE_OPENCV_UNDISTORT|RSCL_OPENCV_ROOT|undistort_map_ready|initUndistortRectifyMap|cv::remap' \
  CMakeLists.txt \
  deploy_rscl/cpp/build_aarch64_thor.sh \
  deploy_rscl/cpp/src/preprocess.cpp
```

成功标准：`bash -n` 无输出且退出码 0，`grep` 能找到构建参数和实现。

## 明天第二步：交叉编译

```bash
cd /data1/psq/CUDA-BEVFusion

export CUDA_HOME=/usr/local/thor/cuda-12.8
export TENSORRT_ROOT=/usr/local/thor/aarch64-linux-gnu
export THOR_TOOLCHAIN_ROOT=/usr/local/thor/aarch64--glibc--bleeding-edge-2024.02-1
export BEVFUSION_CUDA_ARCHS=101

export RSCL_ENABLE_FFMPEG_DECODER=ON
export RSCL_FFMPEG_ROOT=/data1/psq/third_party/ffmpeg_aarch64_thor

export RSCL_ENABLE_OPENCV_UNDISTORT=ON
export RSCL_OPENCV_ROOT=/opt/senseauto/senseauto-3rdparty/1.20260129-021043-e6b2b.pilotmdc-rscl-new.default.4e3eb0.pilotrscl-thor-dev-mp/3rdparty/opencv4

export BUILD_DIR=/data1/psq/CUDA-BEVFusion/build_aarch64_thor_sm101

bash deploy_rscl/cpp/build_aarch64_thor.sh \
  2>&1 | tee /data1/psq/CUDA-BEVFusion/rebuild_sm101_opencv_undistort.log

echo "exit_status=${PIPESTATUS[0]}"
tail -50 /data1/psq/CUDA-BEVFusion/rebuild_sm101_opencv_undistort.log
```

成功标准包括：

```text
RSCL OpenCV undistortion enabled
[100%] Built target rscl_bevfusion_online_node
exit_status=0
```

编译后检查：

```bash
BUILD=/data1/psq/CUDA-BEVFusion/build_aarch64_thor_sm101
file "$BUILD/rscl_bevfusion_online_node"
readelf -d "$BUILD/rscl_bevfusion_online_node" |
grep -E 'opencv_(core|imgproc|calib3d)'
```

## 部署与验证注意事项

只替换新编译的可执行文件并保留备份。不要整份覆盖车端已经调好的 YAML；本地同步参数可能是旧值。车端继续保留：

```yaml
sync_tolerance_ms: 300.0
camera_time_offsets_ms: [350, 350, -100, -100, -100, 350]
lidar_time_offset_ms: 0
undistort_images: true
```

车端环境补充：

```bash
export OPENCV_LIB=/opt/senseauto/senseauto-3rdparty/1.20260227-150511-aba9d.thor.default.57df68.l29-base-thor/3rdparty/opencv4/lib
export LD_LIBRARY_PATH="$OPENCV_LIB:$LD_LIBRARY_PATH"
```

验证标准：

1. `ldd ./bin/rscl_bevfusion_online_node | grep 'not found'` 无输出。
2. 首次完整推理出现 6 个 `undistort_map_ready`，索引 0 到 5。
3. `synced_frames=` 正常且 `decode_errors=0`。
4. 检测输出对象、分割输出地图均正常。
5. 记录修复前后耗时和 CPU。全分辨率六路 CPU remap 可能增加开销；正确性确认后再考虑 GPU/ISP 优化。

若新程序失败，恢复替换前备份并临时设 `undistort_images: false`。这只恢复旧行为，旧行为仍与本次训练预处理不一致。