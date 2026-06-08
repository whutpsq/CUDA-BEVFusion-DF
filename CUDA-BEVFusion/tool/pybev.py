# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensor
import libpybev

model = os.environ["DEBUG_MODEL"]
precision = os.environ["DEBUG_PRECISION"]
data  = os.environ["DEBUG_DATA"]
profile = os.environ.get("DEBUG_PROFILE", "default")

image_names = [
    "0-FRONT.jpg",
    "1-FRONT_RIGHT.jpg",
    "2-FRONT_LEFT.jpg",
    "3-BACK.jpg",
    "4-BACK_LEFT.jpg",
    "5-BACK_RIGHT.jpg"
]

core = libpybev.load_bevfusion(
    f"model/{model}/build/camera.backbone.plan",
    f"model/{model}/build/camera.vtransform.plan",
    f"model/{model}/lidar.backbone.xyz.onnx",
    f"model/{model}/build/fuser.plan",
    f"model/{model}/build/head.bbox.plan",
    precision,
    profile
)

if core is None:
    print("Failed to create core")
    exit(0)

core.print()

def collect_frame_roots(root: str) -> List[str]:
    root_path = Path(root)
    if (root_path / "points.tensor").exists() or (root_path / "images.tensor").exists():
        return [str(root_path)]
    frames = [str(path) for path in sorted(root_path.iterdir()) if path.is_dir() and (path / "points.tensor").exists()]
    return frames or [str(root_path)]


def run_one_frame(frame_root: str):
    images_tensor_file = f"{frame_root}/images.tensor"
    with_normalization = not os.path.exists(images_tensor_file)
    if with_normalization:
        images = []
        for file in image_names:
            if file.endswith(".jpg"):
                image = cv2.imread(f"{frame_root}/{file}")
                image = image[..., ::-1]
                images.append(image)
        images = np.stack(images, axis=0)[None]
    else:
        images = tensor.load(images_tensor_file)

    camera_intrinsics = tensor.load(f"{frame_root}/camera_intrinsics.tensor")
    camera2lidar = tensor.load(f"{frame_root}/camera2lidar.tensor")
    lidar2image = tensor.load(f"{frame_root}/lidar2image.tensor")
    img_aug_matrix = tensor.load(f"{frame_root}/img_aug_matrix.tensor")
    points = tensor.load(f"{frame_root}/points.tensor")

    core.update(
        camera2lidar,
        camera_intrinsics,
        lidar2image,
        img_aug_matrix
    )
    return core.forward(images, points, with_normalization=with_normalization, with_dlpack=False)


frame_roots = collect_frame_roots(data)
for frame_index, frame_root in enumerate(frame_roots):
    boxes = run_one_frame(frame_root)
    np.set_printoptions(3, suppress=True, linewidth=300)
    if len(frame_roots) > 1:
        print(f"[Frame {frame_index:04d}] {frame_root}")
    print(boxes[:10])
