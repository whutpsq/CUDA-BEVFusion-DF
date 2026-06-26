# CUDA-BEVFusion RSCL C++ Adapter

This directory contains the C++ replacement for the Python `deploy_rscl`
runtime.  The code that is independent from the SenseTime RSCL SDK is built as:

```bash
cmake .. -DBUILD_RSCL_CPP=ON
make -j
```

The target is:

```text
librscl_adapter_cpp.a
rscl_bevfusion_bag_runner
```

It reuses `bevfusion_core` directly and does not require `libpybev.so` or a
Python runtime on the vehicle.

## What Was Ported

- YAML adapter config loading for the existing `deploy_rscl/configs/*.yaml`.
- JSON calibration loading for the existing `calibration.json` schema.
- Frame synchronization with the same lidar-anchored nearest-camera policy as
  `FrameSynchronizer` in Python.
- Camera preprocessing:
  - decode JPEG/PNG bytes through stb
  - accept decoded `rgb8`, `bgr8`, `rgba8`, `bgra8`, and `nv12`
  - resize, bottom crop and center crop with the same augmentation matrix
  - normalize to CHW float tensors with the configured mean/std
- Lidar preprocessing:
  - decode float32 point-cloud bytes
  - decode the observed 16-byte RSCL compact point format as int16 cm xyz
  - filter by `point_cloud_range`
  - pad to 5 features for CUDA-BEVFusion
- CUDA-BEVFusion inference through `bevfusion::Core`.
- JSON detection output compatible with the Python adapter:
  - `objects`
  - `timestamp_us`
  - optional `map` mask with `base64_uint8_chw`

## RSCL SDK Boundary

The repository does not contain RSCL C++ headers, so SDK-specific code is kept
out of the build.  Vehicle code should convert RSCL messages to the adapter
types and feed the pipeline:

```cpp
#include "rscl_adapter/codecs.hpp"
#include "rscl_adapter/config.hpp"
#include "rscl_adapter/pipeline.hpp"

using namespace rscl_adapter;

int main() {
  AdapterConfig cfg = load_adapter_config("deploy_rscl/configs/bevfusion_rscl.yaml");
  BevFusionPipeline pipeline(cfg);

  // In each RSCL camera callback:
  std::string output_json;
  CameraPacket camera = decode_camera_packet(
      camera_topic,
      timestamp_us,
      image_bytes,
      image_size,
      encoding,      // "rgb8", "bgr8", "nv12", or empty for JPEG/PNG
      image_width,   // only needed for raw image encodings
      image_height); // only needed for raw image encodings
  if (pipeline.add_camera(camera, &output_json) && !output_json.empty()) {
    // publish output_json to cfg.output_topic as RawMessage
  }

  // In the RSCL lidar callback:
  LidarPacket lidar = decode_lidar_packet(
      cfg.lidar_topic,
      timestamp_us,
      point_bytes,
      point_bytes_size,
      cfg.point_dim,
      point_step,
      width);
  if (pipeline.add_lidar(lidar, &output_json) && !output_json.empty()) {
    // publish output_json to cfg.output_topic as RawMessage
  }
}
```

For H264/H265 camera topics, the C++ adapter can decode the video packet
statefully through FFmpeg when built with `RSCL_ENABLE_FFMPEG_DECODER=ON` and
the FFmpeg development packages are installed. If FFmpeg is unavailable on the
vehicle, decode the packet with the vehicle multimedia stack or RSCL camera SDK
first, then pass decoded RGB/BGR/NV12 bytes to `decode_camera_packet`.

## Notes

- `undistort_images: true` is intentionally rejected in the C++ adapter for now.
  Use rectified camera streams, or add an OpenCV-backed branch in
  `deploy_rscl/cpp/src/preprocess.cpp` if the vehicle runtime provides OpenCV.
- `camera_topics` and `camera_order` must have the same length and order as the
  model calibration.
- The C++ runner mirrors `src/python.cpp::BEVFusion::load`, including the
  `bevfusion_df` versus ResNet50 parameter branch.

## Offline rsclbag Runner

The executable `rscl_bevfusion_bag_runner` runs continuous-frame inference from
an rsclbag once a vehicle RSCL C++ backend is linked:

### Recommended Ubuntu Flow With senseauto-pkm

If the RSCL packages have already been installed with:

```bash
senseauto-pkm install --module *.tar.gz
```

and `/opt` contains `senseauto`, `senseauto_active`, or
`senseauto_install_path`, use the installed SDK directly. No `.sdk_cache`
extraction is required.

Source the environment helper first:

```bash
cd /path/to/CUDA-BEVFusion
source deploy_rscl/cpp/env_senseauto.sh /opt/senseauto_active
```

If `/opt/senseauto_active` does not exist on your machine, omit the argument and
the script will try `/opt/senseauto_active`, `/opt/senseauto`, then
`/opt/senseauto_install_path`:

```bash
source deploy_rscl/cpp/env_senseauto.sh
```

The script exports:

- `SENSEAUTO_INSTALL_ROOT`
- `RSCL_SDK_ROOT`
- `RSCL_MSGS_ROOT`
- `RSCL_THIRDPARTY_ROOT`
- `LD_LIBRARY_PATH` entries for RSCL, messages, and third-party libraries
- `PATH` entry for the SDK `rsclbag` tool if present

Then build:

```bash
rm -rf build_rscl
mkdir -p build_rscl && cd build_rscl
cmake .. \
  -DBUILD_RSCL_CPP=ON \
  -DBUILD_RSCL_BAG_RUNNER=ON \
  -DSENSEAUTO_INSTALL_ROOT=/opt/senseauto_active \
  -DRSCL_ENABLE_FFMPEG_DECODER=ON
make -j
```

Use a fresh build directory after changing RSCL SDK paths. The SenseAuto
third-party package may contain its own protobuf headers and libraries; the
CMake file keeps those paths target-local to `rscl_bevfusion_bag_runner` so
they do not override the protobuf used by CUDA-BEVFusion's ONNX parser.

`-DSENSEAUTO_INSTALL_ROOT` is optional after sourcing the helper because CMake
also reads the exported environment variables. Passing it explicitly makes the
build log easier to audit.

During configure, check that CMake prints paths similar to:

```text
-- RSCL_SDK_ROOT = /opt/senseauto_active/.../senseauto-rscl
-- RSCL_MSGS_ROOT = /opt/senseauto_active/.../senseauto-msgs
-- RSCL_THIRDPARTY_ROOT = /opt/senseauto_active/.../senseauto-3rdparty
```

If those lines do not appear, pass the exact module roots manually:

```bash
cmake .. \
  -DBUILD_RSCL_CPP=ON \
  -DBUILD_RSCL_BAG_RUNNER=ON \
  -DRSCL_SDK_ROOT=/opt/senseauto_active/tmp/senseauto-rscl \
  -DRSCL_MSGS_ROOT=/opt/senseauto_active/tmp/senseauto-msgs \
  -DRSCL_THIRDPARTY_ROOT=/opt/senseauto_active/tmp/senseauto-3rdparty \
  -DRSCL_ENABLE_FFMPEG_DECODER=ON
```

Run the smoke tests before full inference:

```bash
cd /path/to/CUDA-BEVFusion
source deploy_rscl/cpp/env_senseauto.sh /opt/senseauto_active

./build/rscl_bevfusion_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /path/to/mybag.000.rsclbag \
  --decode-only \
  --max-frames 5

./build/rscl_bevfusion_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /path/to/mybag.000.rsclbag \
  --decode-images-only \
  --max-frames 5
```

After the decode checks pass, run full BEVFusion inference:

```bash
./build/rscl_bevfusion_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /path/to/mybag.000.rsclbag \
  --max-frames 20 \
  --output-file runs/cpp_rscl_outputs.jsonl
```

The three stages mean:

- `--decode-only` verifies that the RSCL bag opens, topic filters work, and
  camera/lidar timestamps can synchronize.
- `--decode-images-only` additionally verifies that camera and lidar payloads
  can be decoded into adapter packets.
- Full inference additionally requires CUDA/TensorRT engines, calibration, and
  a valid lidar/camera geometry configuration.

If FFmpeg development packages are unavailable, set
`-DRSCL_ENABLE_FFMPEG_DECODER=OFF` and feed decoded image messages only. Encoded
H264/H265 camera topics require either FFmpeg support in this adapter or vehicle
SDK decoding before calling `decode_camera_packet`.

### Fallback Flow With 695_x86 Tarballs

If the SDK is not installed under `/opt`, prepare the 695_x86 SDK cache. On
Windows/PowerShell:

```powershell
cd H:\github\Lidar_AI_Solution\CUDA-BEVFusion
powershell -ExecutionPolicy Bypass -File deploy_rscl\cpp\setup_695_x86_sdk.ps1 `
  -ReleaseRoot H:\df_code\695_x86 `
  -WithThirdParty
```

On Linux/WSL:

```bash
cd /path/to/CUDA-BEVFusion
WITH_THIRDPARTY=1 bash deploy_rscl/cpp/setup_695_x86_sdk.sh /path/to/695_x86 .sdk_cache/695_x86
```

`WITH_THIRDPARTY=1` / `-WithThirdParty` extracts the large `senseauto-3rdparty`
package. It is usually needed at runtime by `libad_serde_capnp.so` and related
Cap'n Proto/KJ/protobuf dependencies.

Then build the C++ runner. CMake auto-detects `.sdk_cache/695_x86`, or you can
pass `RSCL_RELEASE_ROOT` / `RSCL_SDK_ROOT` explicitly:

```bash
mkdir -p build && cd build
cmake .. \
  -DBUILD_RSCL_CPP=ON \
  -DBUILD_RSCL_BAG_RUNNER=ON \
  -DRSCL_RELEASE_ROOT=../.sdk_cache/695_x86 \
  -DRSCL_ENABLE_FFMPEG_DECODER=ON
make -j
```

The concrete vehicle backend lives in
`deploy_rscl/cpp/vehicle_rscl_bag_reader_ad_bag.cpp`. It uses the RSCL `ad_bag`
reader plus dynamic reflection to convert each bag message into the JSON payload
expected by the shared decoding and BEVFusion pipeline:

```cpp
#include "rscl_adapter/bag_reader.hpp"

namespace rscl_adapter {

class VehicleRsclBagReader : public BagReader {
 public:
  VehicleRsclBagReader(const std::string& bag_path,
                       const std::vector<std::string>& included_topics) {
    // Create the RSCL SDK bag reader here and filter to included_topics.
  }

  bool is_valid() const override {
    // Return SDK reader validity.
  }

  bool read_next(BagMessage* message) override {
    // Read the next SDK message.
    // Fill:
    //   message->topic
    //   message->timestamp_us if the SDK exposes it outside the payload
    //   message->payload with RawMessage JSON bytes, JPEG/PNG bytes,
    //                    decoded RGB/BGR/NV12 bytes, or point cloud bytes
    //
    // If payload is decoded/typed image bytes, also fill:
    //   message->encoding, image_width, image_height
    //
    // If payload is decoded/typed point cloud bytes, also fill:
    //   message->point_step, point_width
    //
    // Return false on EOF.
  }
};

std::unique_ptr<BagReader> create_rscl_bag_reader(
    const std::string& bag_path,
    const std::vector<std::string>& included_topics) {
  return std::unique_ptr<BagReader>(new VehicleRsclBagReader(bag_path, included_topics));
}

}  // namespace rscl_adapter
```

Build with the backend source:

```bash
cmake .. \
  -DBUILD_RSCL_CPP=ON \
  -DRSCL_BAG_BACKEND_SOURCE=/path/to/vehicle_rscl_bag_reader.cpp
make -j
```

If the bag backend cannot reflect a message type, it falls back to the raw bag
buffer so the runner can still surface the topic and timestamp for debugging.
