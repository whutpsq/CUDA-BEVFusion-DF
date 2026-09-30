#!/usr/bin/env python3
"""Capture synchronized Yangluo LiDAR-to-camera projection diagnostics.

The vehicle PointCloud2 payload is RFU while the runtime camera2ego matrices
and detections use FLU.  This script converts RFU points to FLU, selects the
same camera/LiDAR tuple used around a detection timestamp, and projects the
LiDAR points onto the *raw distorted* cam5/cam10 images.  It is intentionally
separate from the inference node so deployment behaviour is not changed.
"""

import argparse
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, PointCloud2
from yangluo_bevfusion_msgs.msg import DetectedObjectArray

from capture_live_swint18_result import (
    PALETTE,
    decode_image,
    object_corners_3d,
    object_yaw,
    pointcloud_xyzi,
    rfu_to_flu,
    select_nearest,
    stamp_seconds,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--front-topic", default="/cam5/compressed")
    parser.add_argument("--rear-topic", default="/cam10/compressed")
    parser.add_argument(
        "--lidar-topic", default="/perception/lidar/concated_points_cloud"
    )
    parser.add_argument(
        "--detection-topic", default="/perception/bevfusion/objects"
    )
    parser.add_argument(
        "--camera-offsets-ms", nargs=2, type=float, default=(48.0, 48.0)
    )
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=55.0)
    parser.add_argument(
        "--selection-window",
        type=float,
        default=3.0,
        help="After the first valid tuple, keep the best-t synchronized tuple for this many seconds.",
    )
    parser.add_argument("--point-stride", type=int, default=3)
    parser.add_argument("--minimum-camera-depth", type=float, default=1.0)
    parser.add_argument("--maximum-camera-depth", type=float, default=80.0)
    parser.add_argument(
        "--image-mode",
        choices=("raw", "undistorted", "both"),
        default="raw",
        help="Use raw for the current Swin checkpoint, which was trained without rectification.",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


def camera_geometry(camera, image_mode):
    camera2ego = np.asarray(camera["camera2ego"], dtype=np.float64)
    ego2camera = np.linalg.inv(camera2ego)
    rotation_vector, _ = cv2.Rodrigues(ego2camera[:3, :3])
    translation = ego2camera[:3, 3]
    intrinsic = np.asarray(camera["camera_intrinsics"], dtype=np.float64)[:3, :3]
    if image_mode == "raw":
        distortion = np.asarray(camera.get("distortion", []), dtype=np.float64)
    else:
        distortion = np.zeros(8, dtype=np.float64)
    return ego2camera, rotation_vector, translation, intrinsic, distortion


def prepare_projection_image(image, camera, image_mode):
    """Return an image consistent with the projection distortion model.

    Raw pixels must be paired with the supplied distortion coefficients.
    Undistorted pixels must be produced first and then paired with zero
    distortion.  Merely setting distortion to zero on the raw image is not a
    valid A/B comparison.
    """
    if image_mode == "raw":
        return image.copy()
    intrinsic = np.asarray(camera["camera_intrinsics"], dtype=np.float64)[:3, :3]
    distortion = np.asarray(camera.get("distortion", []), dtype=np.float64)
    return cv2.undistort(image, intrinsic, distortion, None, intrinsic)


def depth_color(depth, minimum_depth, maximum_depth):
    ratio = np.clip(
        (depth - minimum_depth) / max(1e-6, maximum_depth - minimum_depth),
        0.0,
        1.0,
    )
    hue = np.asarray(np.rint((1.0 - ratio) * 120.0), dtype=np.uint8)
    hsv = np.column_stack(
        (hue, np.full_like(hue, 255), np.full_like(hue, 255))
    ).reshape(-1, 1, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)


def draw_projected_lidar(image, points_flu, camera, args, image_mode):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    sampled = points_flu[:: max(1, args.point_stride), :3]
    ego2camera, rotation_vector, translation, intrinsic, distortion = (
        camera_geometry(camera, image_mode)
    )
    camera_xyz = (
        ego2camera[:3, :3] @ sampled.T
    ).T + translation.reshape(1, 3)
    projected, _ = cv2.projectPoints(
        sampled.reshape(-1, 1, 3),
        rotation_vector,
        translation,
        intrinsic,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    valid = (
        np.isfinite(projected).all(axis=1)
        & (camera_xyz[:, 2] >= args.minimum_camera_depth)
        & (camera_xyz[:, 2] <= args.maximum_camera_depth)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    pixels = np.rint(projected[valid]).astype(np.int32)
    depths = camera_xyz[valid, 2]
    colors = depth_color(
        depths, args.minimum_camera_depth, args.maximum_camera_depth
    )
    order = np.argsort(depths)[::-1]
    for index in order:
        u, v = pixels[index]
        color = tuple(int(value) for value in colors[index])
        cv2.circle(canvas, (int(u), int(v)), 2, color, -1, cv2.LINE_AA)
    return canvas, int(valid.sum()), int(len(sampled))


def clip_projected_segment(point0, point1, width, height):
    """Clip a floating-point segment before converting it to OpenCV integers.

    Strong lens distortion or a box corner close to the camera plane can
    produce finite projected coordinates far outside the int32 range accepted
    by cv2.clipLine.  Liang-Barsky clipping keeps all calculations in float64
    and only converts the already-clipped endpoints to bounded image pixels.
    """
    segment = np.asarray((point0, point1), dtype=np.float64)
    if not np.isfinite(segment).all() or width <= 0 or height <= 0:
        return None

    x0, y0 = segment[0]
    dx, dy = segment[1] - segment[0]
    lower = 0.0
    upper = 1.0
    maximum_x = float(width - 1)
    maximum_y = float(height - 1)

    for direction, distance in (
        (-dx, x0),
        (dx, maximum_x - x0),
        (-dy, y0),
        (dy, maximum_y - y0),
    ):
        if direction == 0.0:
            if distance < 0.0:
                return None
            continue
        ratio = distance / direction
        if direction < 0.0:
            if ratio > upper:
                return None
            lower = max(lower, ratio)
        else:
            if ratio < lower:
                return None
            upper = min(upper, ratio)

    clipped = segment[0] + np.asarray((lower, upper))[:, None] * (dx, dy)
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, maximum_x)
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, maximum_y)
    pixels = np.rint(clipped).astype(np.int32)
    return tuple(int(value) for value in pixels[0]), tuple(
        int(value) for value in pixels[1]
    )


def draw_clipped_boxes(image, objects, camera, image_mode):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    ego2camera, rotation_vector, translation, intrinsic, distortion = (
        camera_geometry(camera, image_mode)
    )
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    drawn = 0
    for obj in objects:
        corners = object_corners_3d(obj)
        camera_xyz = (
            ego2camera[:3, :3] @ corners.T
        ).T + translation.reshape(1, 3)
        projected, _ = cv2.projectPoints(
            corners.reshape(-1, 1, 3),
            rotation_vector,
            translation,
            intrinsic,
            distortion,
        )
        projected = projected.reshape(-1, 2)
        color = PALETTE[int(obj.class_id) % len(PALETTE)]
        edge_drawn = False
        for first, second in edges:
            if camera_xyz[first, 2] <= 0.2 or camera_xyz[second, 2] <= 0.2:
                continue
            clipped = clip_projected_segment(
                projected[first], projected[second], width, height
            )
            if clipped is None:
                continue
            clipped0, clipped1 = clipped
            cv2.line(canvas, clipped0, clipped1, color, 3, cv2.LINE_AA)
            edge_drawn = True
        if not edge_drawn:
            continue
        visible_corners = projected[
            (camera_xyz[:, 2] > 0.2)
            & np.isfinite(projected).all(axis=1)
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < height)
        ]
        if len(visible_corners):
            anchor = np.rint(visible_corners[np.argmin(visible_corners[:, 1])]).astype(int)
            cv2.putText(
                canvas,
                "%s %.2f" % (obj.class_name, obj.score),
                (int(anchor[0]), max(30, int(anchor[1]) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        drawn += 1
    return canvas, drawn


def add_header(image, text):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(
        canvas, text, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return canvas


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "/home/nvidia/psq/yangluo_swint18_projection_diagnostic_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))

    rospy.init_node(
        "yangluo_swint18_projection_diagnostic", anonymous=True, disable_signals=True
    )
    buffers = {
        "front": deque(maxlen=300),
        "rear": deque(maxlen=300),
        "lidar": deque(maxlen=300),
        "detections": deque(maxlen=100),
    }
    subscribers = (
        rospy.Subscriber(args.front_topic, CompressedImage,
                         lambda msg: buffers["front"].append(msg), queue_size=30),
        rospy.Subscriber(args.rear_topic, CompressedImage,
                         lambda msg: buffers["rear"].append(msg), queue_size=30),
        rospy.Subscriber(args.lidar_topic, PointCloud2,
                         lambda msg: buffers["lidar"].append(msg), queue_size=30),
        rospy.Subscriber(args.detection_topic, DetectedObjectArray,
                         lambda msg: buffers["detections"].append(msg), queue_size=20),
    )

    deadline = time.time() + args.timeout
    selected = None
    selected_error = float("inf")
    selection_deadline = None
    while time.time() < deadline and not rospy.is_shutdown():
        rospy.sleep(0.1)
        if not all(buffers.values()):
            continue
        for detection in reversed(buffers["detections"]):
            target = stamp_seconds(detection)
            front, front_diff = select_nearest(
                buffers["front"], target, args.camera_offsets_ms[0]
            )
            rear, rear_diff = select_nearest(
                buffers["rear"], target, args.camera_offsets_ms[1]
            )
            lidar, lidar_diff = select_nearest(buffers["lidar"], target)
            error = max(abs(front_diff), abs(rear_diff), abs(lidar_diff))
            if error <= args.sync_tolerance_ms and error < selected_error:
                selected_error = error
                selected = (
                    detection, front, rear, lidar,
                    front_diff, rear_diff, lidar_diff,
                )
                if selection_deadline is None:
                    selection_deadline = time.time() + max(0.0, args.selection_window)
        if selection_deadline is not None and time.time() >= selection_deadline:
            break

    for subscriber in subscribers:
        subscriber.unregister()
    if selected is None:
        raise RuntimeError("no synchronized detection/camera/lidar tuple found")

    detection, front_msg, rear_msg, lidar_msg, front_diff, rear_diff, lidar_diff = selected
    objects = [obj for obj in detection.objects if obj.score >= args.score_threshold]
    points_rfu = pointcloud_xyzi(lidar_msg)
    points_flu = rfu_to_flu(points_rfu)
    images = {"front": decode_image(front_msg), "rear": decode_image(rear_msg)}
    reports = {}

    modes = ("raw", "undistorted") if args.image_mode == "both" else (args.image_mode,)
    for name, topic, source_id in (
        ("front", args.front_topic, 0),
        ("rear", args.rear_topic, 10),
    ):
        camera = calibration["cameras"][name]
        reports[name] = {}
        for image_mode in modes:
            projection_image = prepare_projection_image(
                images[name], camera, image_mode
            )
            lidar_overlay, projected_count, sampled_count = draw_projected_lidar(
                projection_image, points_flu, camera, args, image_mode
            )
            combined_overlay, box_count = draw_clipped_boxes(
                lidar_overlay, objects, camera, image_mode
            )
            title = (
                "%s | calib %d | %s | lidar=%d/%d boxes=%d"
                % (topic, source_id, image_mode, projected_count, sampled_count, box_count)
            )
            lidar_overlay = add_header(
                lidar_overlay, title.replace("boxes=%d" % box_count, "points")
            )
            combined_overlay = add_header(combined_overlay, title)
            lidar_path = output_dir / (
                name + "_" + image_mode + "_lidar_projection.jpg"
            )
            combined_path = output_dir / (
                name + "_" + image_mode + "_lidar_and_boxes.jpg"
            )
            cv2.imwrite(
                str(lidar_path), lidar_overlay,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            cv2.imwrite(
                str(combined_path), combined_overlay,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            reports[name][image_mode] = {
                "topic": topic,
                "source_camera_id": source_id,
                "projected_point_count": projected_count,
                "sampled_point_count": sampled_count,
                "projected_box_count": box_count,
                "lidar_projection": str(lidar_path),
                "lidar_and_boxes": str(combined_path),
            }

    report = {
        "coordinate_contract": {
            "pointcloud_payload": "RFU",
            "projection_points": "RFU_to_FLU_then_camera2ego_inverse",
            "detections": "base_link FLU; pose is geometric box center",
        },
        "image_mode": args.image_mode,
        "camera_offsets_ms": list(args.camera_offsets_ms),
        "sync_difference_ms": {
            "front": front_diff,
            "rear": rear_diff,
            "lidar": lidar_diff,
        },
        "stamps": {
            "detection": stamp_seconds(detection),
            "front": stamp_seconds(front_msg),
            "rear": stamp_seconds(rear_msg),
            "lidar": stamp_seconds(lidar_msg),
        },
        "raw_point_count": int(len(points_rfu)),
        "score_threshold": args.score_threshold,
        "published_object_count": len(detection.objects),
        "rendered_object_count": len(objects),
        "cameras": reports,
        "objects": [
            {
                "class_id": int(obj.class_id),
                "class_name": obj.class_name,
                "score": float(obj.score),
                "position_flu": [
                    float(obj.pose.position.x),
                    float(obj.pose.position.y),
                    float(obj.pose.position.z),
                ],
                "dimensions": [
                    float(obj.dimensions.x),
                    float(obj.dimensions.y),
                    float(obj.dimensions.z),
                ],
                "yaw_flu": object_yaw(obj),
            }
            for obj in objects
        ],
    }
    report_path = output_dir / "projection_diagnostic_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("SWINT18_PROJECTION_DIAGNOSTIC_OK", flush=True)
    print("output_dir=" + str(output_dir), flush=True)
    print(
        "sync_ms front=%.3f rear=%.3f lidar=%.3f"
        % (front_diff, rear_diff, lidar_diff),
        flush=True,
    )
    print("report=" + str(report_path), flush=True)


if __name__ == "__main__":
    main()
