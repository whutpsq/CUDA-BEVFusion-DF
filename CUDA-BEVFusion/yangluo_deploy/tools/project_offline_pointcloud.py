#!/usr/bin/env python3
"""Project a binary PCD in the stitched Yangluo RFU frame onto two images.

The camera calibration extrinsics are interpreted as
``p_camera = T_camera_from_RFU @ p_RFU``.  No RFU/FLU conversion and no
individual-LiDAR extrinsic are applied.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw


PCD_DTYPES = {
    ("F", 4): "f4",
    ("F", 8): "f8",
    ("U", 1): "u1",
    ("U", 2): "u2",
    ("U", 4): "u4",
    ("I", 1): "i1",
    ("I", 2): "i2",
    ("I", 4): "i4",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", required=True)
    parser.add_argument("--front-image", required=True)
    parser.add_argument("--rear-image", required=True)
    parser.add_argument("--camera-calibration", required=True)
    parser.add_argument("--front-calibration-id", default="0")
    parser.add_argument("--rear-calibration-id", default="10")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--draw-stride", type=int, default=3)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--minimum-depth", type=float, default=0.5)
    parser.add_argument("--maximum-color-depth", type=float, default=60.0)
    return parser.parse_args()


def load_json_with_comments(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(re.sub(r"/\*.*?\*/", "", text, flags=re.S))


def read_binary_pcd(path):
    path = Path(path)
    header_lines = []
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError("PCD header ended before DATA")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded.startswith("DATA "):
                data_offset = stream.tell()
                break
    header = {}
    for line in header_lines:
        if not line or line.startswith("#"):
            continue
        key, *values = line.split()
        header[key.upper()] = values
    if header.get("DATA") != ["binary"]:
        raise RuntimeError("Only DATA binary PCD is supported")
    fields = header["FIELDS"]
    sizes = list(map(int, header["SIZE"]))
    types = header["TYPE"]
    counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
    if any(count != 1 for count in counts):
        raise RuntimeError("Only scalar PCD fields are supported")
    dtype_fields = []
    for name, size, kind in zip(fields, sizes, types):
        key = (kind, size)
        if key not in PCD_DTYPES:
            raise RuntimeError("Unsupported PCD field type: %s/%s" % key)
        dtype_fields.append((name, "<" + PCD_DTYPES[key]))
    point_count = int(header["POINTS"][0])
    points = np.fromfile(path, dtype=np.dtype(dtype_fields), count=point_count,
                         offset=data_offset)
    if len(points) != point_count:
        raise RuntimeError("PCD binary payload is truncated")
    for required in ("x", "y", "z"):
        if required not in points.dtype.names:
            raise RuntimeError("PCD is missing field " + required)
    xyz = np.column_stack((points["x"], points["y"], points["z"]))
    intensity = (np.asarray(points["intensity"], dtype=np.float64)
                 if "intensity" in points.dtype.names else None)
    return xyz.astype(np.float64), intensity, fields


def distort_normalized(x, y, coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if len(coefficients) == 0:
        return x, y
    if len(coefficients) not in (4, 5, 8):
        raise RuntimeError("Expected 4, 5, or 8 distortion values")
    padded = np.zeros(8, dtype=np.float64)
    padded[:len(coefficients)] = coefficients
    k1, k2, p1, p2, k3, k4, k5, k6 = padded
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial = numerator / denominator
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def depth_color(depth, maximum):
    ratio = max(0.0, min(float(depth) / maximum, 1.0))
    # RGB: near red, middle green, far blue.
    return (
        int(255.0 * (1.0 - ratio)),
        int(255.0 * (1.0 - abs(ratio - 0.5) * 2.0)),
        int(255.0 * ratio),
    )


def project(image_path, xyz_rfu, calibration, apply_distortion, args):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    intrinsic = np.asarray(calibration["intrinsics"], dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(calibration["extrinsics"], dtype=np.float64).reshape(4, 4)
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    camera_xyz = xyz_rfu @ rotation.T + translation
    depth = camera_xyz[:, 2]
    candidate = np.isfinite(camera_xyz).all(axis=1) & (depth > args.minimum_depth)
    indices = np.flatnonzero(candidate)
    normalized = camera_xyz[indices, :2] / depth[indices, None]
    x = normalized[:, 0]
    y = normalized[:, 1]
    if apply_distortion:
        x, y = distort_normalized(x, y, calibration.get("distortion", []))
    u = intrinsic[0, 0] * x + intrinsic[0, 2]
    v = intrinsic[1, 1] * y + intrinsic[1, 2]
    inside = (np.isfinite(u) & np.isfinite(v) &
              (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height))
    visible_indices = indices[inside]
    visible_u = u[inside]
    visible_v = v[inside]
    order = np.argsort(depth[visible_indices])[::-1]  # far first, near last
    draw = ImageDraw.Draw(image)
    radius = args.point_radius
    stride = max(1, args.draw_stride)
    for selected in order[::stride]:
        point_index = visible_indices[selected]
        px = int(round(float(visible_u[selected])))
        py = int(round(float(visible_v[selected])))
        color = depth_color(depth[point_index], args.maximum_color_depth)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                     fill=color)
    camera_position = -rotation.T @ translation
    return image, {
        "input_points": int(len(xyz_rfu)),
        "positive_depth_points": int(candidate.sum()),
        "projected_inside_image": int(inside.sum()),
        "drawn_points": int((len(order) + stride - 1) // stride),
        "image_size": [width, height],
        "camera_position_in_RFU_m": camera_position.tolist(),
        "extrinsic_contract": "p_camera = T_camera_from_RFU @ p_RFU",
        "coordinate_conversion_applied": False,
        "distortion_applied": bool(apply_distortion),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xyz, intensity, fields = read_binary_pcd(args.pcd)
    raw_calibration = load_json_with_comments(args.camera_calibration)
    cameras = {str(item["desc"]): item for item in raw_calibration["camera_params"]}
    jobs = [
        ("front", args.front_image, args.front_calibration_id),
        ("rear", args.rear_image, args.rear_calibration_id),
    ]
    report = {
        "pcd": str(Path(args.pcd).resolve()),
        "pcd_fields": fields,
        "point_count": int(len(xyz)),
        "camera_calibration": str(Path(args.camera_calibration).resolve()),
        "mapping": {},
    }
    for name, image_path, calibration_id in jobs:
        if calibration_id not in cameras:
            raise RuntimeError("Missing calibration ID " + calibration_id)
        camera_report = {"image": str(Path(image_path).resolve()),
                         "calibration_id": calibration_id}
        for distortion_name, apply_distortion in (("distorted", True),
                                                   ("pinhole", False)):
            overlay, statistics = project(
                image_path, xyz, cameras[calibration_id], apply_distortion, args
            )
            output_path = output_dir / (name + "_calib_" + calibration_id +
                                        "_" + distortion_name + ".jpg")
            overlay.save(output_path, quality=95)
            camera_report[distortion_name] = {
                "output": str(output_path.resolve()), **statistics
            }
            print("PROJECTION_OK", name, "calibration=" + calibration_id,
                  "distortion=" + str(apply_distortion),
                  "inside=" + str(statistics["projected_inside_image"]),
                  "output=" + str(output_path), flush=True)
        report["mapping"][name] = camera_report
    report_path = output_dir / "projection_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print("PROJECTION_REPORT_OK", report_path, flush=True)


if __name__ == "__main__":
    main()
