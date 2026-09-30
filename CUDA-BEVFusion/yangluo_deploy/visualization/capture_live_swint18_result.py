#!/usr/bin/env python3
"""Capture one synchronized Yangluo live detection result on ROS1.

This tool is isolated from the inference node.  It subscribes to the two
vehicle cameras, the raw RFU PointCloud2 stream and the published FLU
DetectedObjectArray, then saves front/rear camera overlays, a FLU BEV overlay,
a combined contact sheet and machine-readable capture metadata.
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


PALETTE = (
    (0, 255, 255),
    (0, 165, 255),
    (255, 180, 0),
    (80, 255, 80),
    (255, 80, 180),
    (180, 80, 255),
    (255, 255, 0),
    (0, 220, 120),
    (220, 120, 0),
    (120, 220, 255),
    (255, 120, 120),
    (120, 120, 255),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration",
        required=True,
        help="Runtime calibration JSON containing front/rear camera2ego matrices.",
    )
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
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=55.0)
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--bev-range", type=float, default=54.0)
    parser.add_argument("--bev-size", type=int, default=1080)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def stamp_seconds(message):
    return message.header.stamp.to_sec()


def select_nearest(messages, target_stamp, offset_ms=0.0):
    if not messages:
        return None, float("inf")
    selected = min(
        messages,
        key=lambda message: abs(
            stamp_seconds(message) + offset_ms / 1000.0 - target_stamp
        ),
    )
    difference_ms = (
        stamp_seconds(selected) + offset_ms / 1000.0 - target_stamp
    ) * 1000.0
    return selected, difference_ms


def decode_image(message):
    image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        header = getattr(message, "_connection_header", None) or {}
        raise RuntimeError("failed to decode image topic " + header.get("topic", ""))
    return image


def pointcloud_xyzi(message):
    fields = {field.name: field for field in message.fields}
    missing = [name for name in ("x", "y", "z", "intensity") if name not in fields]
    if missing:
        raise RuntimeError("PointCloud2 missing fields: " + ",".join(missing))
    byte_order = ">" if message.is_bigendian else "<"
    dtype = np.dtype({
        "names": ("x", "y", "z", "intensity"),
        "formats": (byte_order + "f4",) * 4,
        "offsets": tuple(fields[name].offset for name in ("x", "y", "z", "intensity")),
        "itemsize": message.point_step,
    })
    count = message.width * message.height
    structured = np.frombuffer(message.data, dtype=dtype, count=count)
    points = np.column_stack(
        (structured["x"], structured["y"], structured["z"], structured["intensity"])
    ).astype(np.float32, copy=False)
    return points[np.isfinite(points).all(axis=1)]


def rfu_to_flu(points):
    converted = points.copy()
    converted[:, 0] = points[:, 1]
    converted[:, 1] = -points[:, 0]
    return converted


def object_color(obj):
    return PALETTE[int(obj.class_id) % len(PALETTE)]


def object_yaw(obj):
    quaternion = obj.pose.orientation
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def object_corners_3d(obj):
    """Build box corners from a geometric-center ROS Pose."""
    half_x = obj.dimensions.x * 0.5
    half_y = obj.dimensions.y * 0.5
    half_z = obj.dimensions.z * 0.5
    local_xy = np.asarray(
        (
            (half_x, half_y),
            (-half_x, half_y),
            (-half_x, -half_y),
            (half_x, -half_y),
        ),
        dtype=np.float64,
    )
    yaw = object_yaw(obj)
    rotation = np.asarray(
        ((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw))),
        dtype=np.float64,
    )
    xy = local_xy @ rotation.T
    xy[:, 0] += obj.pose.position.x
    xy[:, 1] += obj.pose.position.y
    bottom = np.column_stack((xy, np.full(4, obj.pose.position.z - half_z)))
    top = np.column_stack((xy, np.full(4, obj.pose.position.z + half_z)))
    return np.vstack((bottom, top))


def draw_camera(image, objects, camera, title):
    canvas = image.copy()
    camera2ego = np.asarray(camera["camera2ego"], dtype=np.float64)
    ego2camera = np.linalg.inv(camera2ego)
    rotation_vector, _ = cv2.Rodrigues(ego2camera[:3, :3])
    translation = ego2camera[:3, 3]
    intrinsic = np.asarray(camera["camera_intrinsics"], dtype=np.float64)[:3, :3]
    distortion = np.asarray(camera.get("distortion", []), dtype=np.float64)
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
        if np.count_nonzero(camera_xyz[:, 2] > 0.2) < 2:
            continue
        color = object_color(obj)
        edge_drawn = False
        for first, second in edges:
            if camera_xyz[first, 2] <= 0.2 or camera_xyz[second, 2] <= 0.2:
                continue
            p0 = tuple(np.rint(projected[first]).astype(int))
            p1 = tuple(np.rint(projected[second]).astype(int))
            cv2.line(canvas, p0, p1, color, 3, cv2.LINE_AA)
            edge_drawn = True
        if not edge_drawn:
            continue
        visible = projected[camera_xyz[:, 2] > 0.2]
        anchor = np.rint(visible[np.argmin(visible[:, 1])]).astype(int)
        label = "%s %.2f" % (obj.class_name, obj.score)
        cv2.putText(
            canvas,
            label,
            (int(anchor[0]), max(35, int(anchor[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        drawn += 1
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        "%s | projected boxes=%d" % (title, drawn),
        (20, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas, drawn


def bev_pixel(x, y, bev_range, size):
    scale = size / (2.0 * bev_range)
    return int(round(size * 0.5 - y * scale)), int(round(size * 0.5 - x * scale))


def draw_bev(points_flu, objects, bev_range, size, point_stride):
    canvas = np.full((size, size, 3), 20, dtype=np.uint8)
    scale = size / (2.0 * bev_range)
    for distance in np.arange(-bev_range, bev_range + 0.1, 10.0):
        u0, v0 = bev_pixel(-bev_range, distance, bev_range, size)
        u1, v1 = bev_pixel(bev_range, distance, bev_range, size)
        cv2.line(canvas, (u0, v0), (u1, v1), (55, 55, 55), 1)
        u0, v0 = bev_pixel(distance, -bev_range, bev_range, size)
        u1, v1 = bev_pixel(distance, bev_range, bev_range, size)
        cv2.line(canvas, (u0, v0), (u1, v1), (55, 55, 55), 1)

    sampled = points_flu[::max(1, point_stride)]
    valid = (
        (np.abs(sampled[:, 0]) <= bev_range)
        & (np.abs(sampled[:, 1]) <= bev_range)
    )
    sampled = sampled[valid]
    u = np.rint(size * 0.5 - sampled[:, 1] * scale).astype(np.int32)
    v = np.rint(size * 0.5 - sampled[:, 0] * scale).astype(np.int32)
    u = np.clip(u, 0, size - 1)
    v = np.clip(v, 0, size - 1)
    z_ratio = np.clip((sampled[:, 2] + 2.0) / 5.0, 0.0, 1.0)
    colors = np.column_stack(
        (
            255.0 * (1.0 - z_ratio),
            255.0 * (1.0 - np.abs(z_ratio - 0.5) * 2.0),
            255.0 * z_ratio,
        )
    ).astype(np.uint8)
    canvas[v, u] = colors
    point_layer = cv2.dilate(canvas, np.ones((2, 2), dtype=np.uint8))
    canvas = np.maximum(canvas, point_layer)

    for obj in objects:
        corners = np.asarray(
            [[corner.x, corner.y] for corner in obj.corners], dtype=np.float64
        )
        polygon = np.asarray(
            [bev_pixel(x, y, bev_range, size) for x, y in corners],
            dtype=np.int32,
        )
        color = object_color(obj)
        cv2.polylines(canvas, [polygon], True, color, 3, cv2.LINE_AA)
        center = bev_pixel(obj.pose.position.x, obj.pose.position.y, bev_range, size)
        cv2.circle(canvas, center, 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "%s %.2f" % (obj.class_name, obj.score),
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    center = size // 2
    vehicle_length = int(round(14.0 * scale))
    vehicle_width = int(round(3.0 * scale))
    cv2.rectangle(
        canvas,
        (center - vehicle_width // 2, center - vehicle_length // 2),
        (center + vehicle_width // 2, center + vehicle_length // 2),
        (255, 255, 255),
        2,
    )
    cv2.arrowedLine(canvas, (center, center), (center, center - 60), (255, 255, 255), 3)
    cv2.putText(canvas, "FRONT +X", (center + 12, center - 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "LEFT +Y", (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "BEV FLU | objects=%d | range=+/-%.0fm" % (len(objects), bev_range),
                (25, size - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def resize_letterbox(image, width, height):
    ratio = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (int(round(image.shape[1] * ratio)), int(round(image.shape[0] * ratio))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "/home/nvidia/psq/yangluo_swint18_visualization_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))

    rospy.init_node("yangluo_swint18_live_visualizer", anonymous=True, disable_signals=True)
    buffers = {
        "front": deque(maxlen=300),
        "rear": deque(maxlen=300),
        "lidar": deque(maxlen=300),
        "detections": deque(maxlen=100),
    }
    subscribers = (
        rospy.Subscriber(args.front_topic, CompressedImage,
                         lambda message: buffers["front"].append(message), queue_size=30),
        rospy.Subscriber(args.rear_topic, CompressedImage,
                         lambda message: buffers["rear"].append(message), queue_size=30),
        rospy.Subscriber(args.lidar_topic, PointCloud2,
                         lambda message: buffers["lidar"].append(message), queue_size=30),
        rospy.Subscriber(args.detection_topic, DetectedObjectArray,
                         lambda message: buffers["detections"].append(message), queue_size=20),
    )

    deadline = time.time() + args.timeout
    selected = None
    while time.time() < deadline and not rospy.is_shutdown():
        rospy.sleep(0.1)
        if not all(buffers.values()):
            continue
        for detection in reversed(buffers["detections"]):
            target_stamp = stamp_seconds(detection)
            front, front_diff = select_nearest(
                buffers["front"], target_stamp, args.camera_offsets_ms[0]
            )
            rear, rear_diff = select_nearest(
                buffers["rear"], target_stamp, args.camera_offsets_ms[1]
            )
            lidar, lidar_diff = select_nearest(buffers["lidar"], target_stamp)
            if max(abs(front_diff), abs(rear_diff), abs(lidar_diff)) <= args.sync_tolerance_ms:
                selected = (detection, front, rear, lidar, front_diff, rear_diff, lidar_diff)
                break
        if selected is not None:
            break

    for subscriber in subscribers:
        subscriber.unregister()
    if selected is None:
        raise RuntimeError("no synchronized detection/camera/lidar tuple found before timeout")

    detection, front_message, rear_message, lidar_message, front_diff, rear_diff, lidar_diff = selected
    objects = [obj for obj in detection.objects if obj.score >= args.score_threshold]
    front_image = decode_image(front_message)
    rear_image = decode_image(rear_message)
    points_rfu = pointcloud_xyzi(lidar_message)
    points_flu = rfu_to_flu(points_rfu)

    front_result, front_boxes = draw_camera(
        front_image, objects, calibration["cameras"]["front"],
        "FRONT /cam5 | calib 0",
    )
    rear_result, rear_boxes = draw_camera(
        rear_image, objects, calibration["cameras"]["rear"],
        "REAR /cam10 | calib 10",
    )
    bev_result = draw_bev(
        points_flu, objects, args.bev_range, args.bev_size, args.point_stride
    )

    front_path = output_dir / "front_detections.jpg"
    rear_path = output_dir / "rear_detections.jpg"
    bev_path = output_dir / "bev_detections.jpg"
    combined_path = output_dir / "combined_detections.jpg"
    cv2.imwrite(str(front_path), front_result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(rear_path), rear_result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(bev_path), bev_result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    top = np.hstack(
        (resize_letterbox(front_result, 960, 540), resize_letterbox(rear_result, 960, 540))
    )
    bottom = np.full((1080, 1920, 3), 12, dtype=np.uint8)
    bev_large = resize_letterbox(bev_result, 1080, 1080)
    bottom[:, 420:1500] = bev_large
    combined = np.vstack((top, bottom))
    cv2.imwrite(str(combined_path), combined, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    report = {
        "coordinate_contract": {
            "pointcloud_payload": "RFU",
            "visualization_and_detections": "base_link FLU",
            "bev_axes": "up=+X front, left=+Y left",
        },
        "topics": {
            "front": args.front_topic,
            "rear": args.rear_topic,
            "lidar": args.lidar_topic,
            "detections": args.detection_topic,
        },
        "stamps": {
            "detection": stamp_seconds(detection),
            "front": stamp_seconds(front_message),
            "rear": stamp_seconds(rear_message),
            "lidar": stamp_seconds(lidar_message),
            "front_adjusted_difference_ms": front_diff,
            "rear_adjusted_difference_ms": rear_diff,
            "lidar_difference_ms": lidar_diff,
        },
        "camera_offsets_ms": list(args.camera_offsets_ms),
        "score_threshold": args.score_threshold,
        "raw_point_count": int(len(points_rfu)),
        "published_object_count": len(detection.objects),
        "rendered_object_count": len(objects),
        "front_projected_box_count": front_boxes,
        "rear_projected_box_count": rear_boxes,
        "objects": [
            {
                "class_id": int(obj.class_id),
                "class_name": obj.class_name,
                "score": float(obj.score),
                "position_flu": [
                    float(obj.pose.position.x), float(obj.pose.position.y), float(obj.pose.position.z)
                ],
                "dimensions": [
                    float(obj.dimensions.x), float(obj.dimensions.y), float(obj.dimensions.z)
                ],
                "yaw_flu": object_yaw(obj),
            }
            for obj in objects
        ],
        "outputs": {
            "front": str(front_path),
            "rear": str(rear_path),
            "bev": str(bev_path),
            "combined": str(combined_path),
        },
    }
    report_path = output_dir / "capture_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("SWINT18_VISUALIZATION_OK", flush=True)
    print("output_dir=" + str(output_dir), flush=True)
    print("objects=%d front_boxes=%d rear_boxes=%d points=%d" % (
        len(objects), front_boxes, rear_boxes, len(points_rfu)
    ), flush=True)
    print("sync_ms front=%.3f rear=%.3f lidar=%.3f" % (
        front_diff, rear_diff, lidar_diff
    ), flush=True)
    print("front=" + str(front_path), flush=True)
    print("rear=" + str(rear_path), flush=True)
    print("bev=" + str(bev_path), flush=True)
    print("combined=" + str(combined_path), flush=True)
    print("report=" + str(report_path), flush=True)


if __name__ == "__main__":
    main()
