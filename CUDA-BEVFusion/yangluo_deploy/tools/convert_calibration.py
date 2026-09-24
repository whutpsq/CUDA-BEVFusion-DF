#!/usr/bin/env python3
"""Convert supplied RFU-to-camera calibration to BEVFusion FLU matrices."""

import argparse
import json
import re
from pathlib import Path


FLU_TO_RFU = [
    [0.0, -1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def matmul(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def inverse(matrix):
    a = [list(map(float, row)) + [1.0 if r == c else 0.0 for c in range(4)] for r, row in enumerate(matrix)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise RuntimeError("singular calibration matrix")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for row in range(4):
            if row == col:
                continue
            scale = a[row][col]
            a[row] = [a[row][i] - scale * a[col][i] for i in range(8)]
    return [row[4:] for row in a]


def matrix4(values):
    if len(values) != 16:
        raise RuntimeError("extrinsic must contain 16 values")
    return [list(map(float, values[r * 4 : (r + 1) * 4])) for r in range(4)]


def intrinsic4(values):
    if len(values) != 9:
        raise RuntimeError("intrinsic must contain 9 values")
    out = [[0.0] * 4 for _ in range(4)]
    for r in range(3):
        for c in range(3):
            out[r][c] = float(values[r * 3 + c])
    out[3][3] = 1.0
    return out


def load_json_with_comments(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(re.sub(r"/\*.*?\*/", "", text, flags=re.S))


def distortion8(values, camera_id):
    """Normalize OpenCV 4/5/8 coefficient models to the runtime 8-vector."""
    values = list(map(float, values))
    if len(values) == 4:
        values.append(0.0)
    if len(values) == 5:
        values.extend([0.0, 0.0, 0.0])
    if len(values) != 8:
        raise RuntimeError(
            f"camera {camera_id} must have 4, 5, or 8 distortion coefficients"
        )
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--front-id", default="0")
    parser.add_argument("--rear-id", default="10")
    args = parser.parse_args()

    raw = load_json_with_comments(args.input)
    by_id = {str(item["desc"]): item for item in raw["camera_params"]}
    cameras = {}
    for name, camera_id in (("front", args.front_id), ("rear", args.rear_id)):
        item = by_id[camera_id]
        flu_to_camera = matmul(matrix4(item["extrinsics"]), FLU_TO_RFU)
        cameras[name] = {
            "camera_intrinsics": intrinsic4(item["intrinsics"]),
            "camera2ego": inverse(flu_to_camera),
            "distortion": distortion8(item["distortion"], camera_id),
            "source_camera_id": camera_id,
            "width": int(item["width"]),
            "height": int(item["height"]),
        }

    output = {
        "coordinate_contract": "base_link FLU; supplied extrinsics are RFU vehicle to OpenCV camera",
        "lidar2ego": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "cameras": cameras,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PASS cameras=front:{args.front_id},rear:{args.rear_id} output={destination}")


if __name__ == "__main__":
    main()
