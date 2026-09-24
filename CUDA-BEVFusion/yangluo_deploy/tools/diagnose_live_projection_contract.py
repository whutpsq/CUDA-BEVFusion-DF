#!/usr/bin/env python3
"""Diagnose live Yangluo camera/LiDAR mapping and point-frame conventions.

This tool is intentionally independent from the production ROS node.  It
captures one LiDAR message and buffered camera messages, selects the nearest
camera frames after applying the configured timestamp offsets, then renders a
4 x 4 projection matrix for every camera topic:

  columns: source calibration IDs 0, 1, 5 and 10
  rows:    four plausible XY conventions for the PointCloud2 payload

The supplied camera extrinsics are used directly as RFU-vehicle-to-camera
transforms.  This makes every hypothesis explicit and avoids hiding a basis
conversion inside a generated runtime calibration file.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CompressedImage, PointCloud2


CALIBRATION_IDS = ("0", "1", "5", "10")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-topics", nargs=2,
                        default=["/cam5/compressed", "/cam10/compressed"])
    parser.add_argument("--camera-offsets-ms", nargs=2, type=float,
                        default=[48.0, 48.0])
    parser.add_argument("--lidar-topic",
                        default="/perception/lidar/concated_points_cloud")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--buffer-seconds", type=float, default=1.0)
    parser.add_argument("--draw-stride", type=int, default=6)
    return parser.parse_args()


def load_json_with_comments(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(re.sub(r"/\*.*?\*/", "", text, flags=re.S))


def stamp_seconds(message):
    return message.header.stamp.to_sec()


def point_hypotheses(xyz):
    """Return candidate interpretations, all expressed in calibration RFU."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    return {
        # PointCloud2 is FLU: RFU = [-left, forward, up].
        "payload_FLU": np.column_stack((-y, x, z)),
        # PointCloud2 already contains the stitched RFU vehicle cloud.
        "payload_RFU": np.column_stack((x, y, z)),
        # Same two candidates with a 180 degree XY reversal.  These expose a
        # common front/back sign error without inventing a continuous fit.
        "payload_FLU_rot180": np.column_stack((y, -x, z)),
        "payload_RFU_rot180": np.column_stack((-x, -y, z)),
    }


def project_overlay(image, xyz_rfu, calibration, draw_stride):
    extrinsic = np.asarray(calibration["extrinsics"], dtype=np.float64).reshape(4, 4)
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    rotation_vector, _ = cv2.Rodrigues(rotation)
    intrinsic = np.asarray(calibration["intrinsics"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(calibration["distortion"], dtype=np.float64)

    camera_xyz = (rotation @ xyz_rfu.T).T + translation.reshape(1, 3)
    projected, _ = cv2.projectPoints(
        xyz_rfu.reshape(-1, 1, 3),
        rotation_vector,
        translation,
        intrinsic,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    depth = camera_xyz[:, 2]
    height, width = image.shape[:2]
    valid = (
        np.isfinite(projected).all(axis=1)
        & (depth > 0.5)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    valid_indices = np.flatnonzero(valid)
    overlay = image.copy()
    for index in valid_indices[::max(1, draw_stride)]:
        u = int(round(projected[index, 0]))
        v = int(round(projected[index, 1]))
        ratio = max(0.0, min(float(depth[index]), 60.0)) / 60.0
        color = (
            int(255 * ratio),
            int(255 * (1.0 - abs(ratio - 0.5) * 2.0)),
            int(255 * (1.0 - ratio)),
        )
        cv2.circle(overlay, (u, v), 3, color, -1, lineType=cv2.LINE_AA)
    return overlay, int(valid.sum()), int((depth > 0.5).sum())


def main():
    args = parse_args()
    if len(args.camera_topics) != len(args.camera_offsets_ms):
        raise RuntimeError("camera topic/offset counts differ")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_calibration = load_json_with_comments(args.camera_calibration)
    by_id = {str(item["desc"]): item for item in raw_calibration["camera_params"]}
    missing = [camera_id for camera_id in CALIBRATION_IDS if camera_id not in by_id]
    if missing:
        raise RuntimeError("missing camera calibration IDs: " + ",".join(missing))

    rospy.init_node("yangluo_projection_contract_diagnostic",
                    anonymous=True, disable_signals=True)
    buffers = {topic: deque(maxlen=100) for topic in args.camera_topics}
    subscribers = []
    for topic in args.camera_topics:
        subscribers.append(rospy.Subscriber(
            topic,
            CompressedImage,
            lambda message, selected_topic=topic:
                buffers[selected_topic].append(message),
            queue_size=20,
        ))

    deadline = time.time() + args.timeout
    while time.time() < deadline and any(len(queue) < 3 for queue in buffers.values()):
        rospy.sleep(0.05)
    if any(not queue for queue in buffers.values()):
        raise RuntimeError("camera buffers did not receive all topics")

    lidar_message = rospy.wait_for_message(
        args.lidar_topic, PointCloud2, timeout=max(0.1, deadline - time.time())
    )
    rospy.sleep(max(0.0, args.buffer_seconds))
    lidar_stamp = stamp_seconds(lidar_message)

    selected = {}
    for topic, offset_ms in zip(args.camera_topics, args.camera_offsets_ms):
        selected[topic] = min(
            buffers[topic],
            key=lambda message: abs(
                stamp_seconds(message) + offset_ms / 1000.0 - lidar_stamp
            ),
        )

    points = np.asarray(list(point_cloud2.read_points(
        lidar_message,
        field_names=["x", "y", "z", "intensity"],
        skip_nans=True,
    )), dtype=np.float64)
    xyz = points[:, :3]
    hypotheses = point_hypotheses(xyz)
    report = {
        "camera_calibration": str(Path(args.camera_calibration).resolve()),
        "lidar_topic": args.lidar_topic,
        "lidar_stamp": lidar_stamp,
        "lidar_frame_id": lidar_message.header.frame_id,
        "point_count": int(len(xyz)),
        "camera_offsets_ms": dict(zip(args.camera_topics, args.camera_offsets_ms)),
        "cameras": {},
    }

    cell_size = (480, 270)
    for camera_index, topic in enumerate(args.camera_topics):
        message = selected[topic]
        image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("failed to decode " + topic)
        topic_stem = "cam" + re.sub(r"[^0-9]", "", topic.split("/")[-2])
        raw_path = output_dir / (topic_stem + "_raw.jpg")
        cv2.imwrite(str(raw_path), image)

        rows = []
        camera_report = {
            "stamp": stamp_seconds(message),
            "adjusted_camera_minus_lidar_ms": (
                stamp_seconds(message)
                + args.camera_offsets_ms[camera_index] / 1000.0
                - lidar_stamp
            ) * 1000.0,
            "raw_camera_minus_lidar_ms": (
                stamp_seconds(message) - lidar_stamp
            ) * 1000.0,
            "image_shape": list(image.shape),
            "raw_image": str(raw_path),
            "hypotheses": {},
        }
        for hypothesis_name, xyz_rfu in hypotheses.items():
            cells = []
            for camera_id in CALIBRATION_IDS:
                overlay, inside, positive_depth = project_overlay(
                    image, xyz_rfu, by_id[camera_id], args.draw_stride
                )
                label = "%s | calib=%s | in=%d" % (
                    hypothesis_name, camera_id, inside
                )
                cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 75),
                              (0, 0, 0), -1)
                cv2.putText(overlay, label, (25, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.15,
                            (0, 255, 0), 3, cv2.LINE_AA)
                cells.append(cv2.resize(overlay, cell_size,
                                        interpolation=cv2.INTER_AREA))
                camera_report["hypotheses"][hypothesis_name + "/" + camera_id] = {
                    "projected_inside_image": inside,
                    "positive_depth_points": positive_depth,
                }
            rows.append(np.hstack(cells))
        contact_sheet = np.vstack(rows)
        contact_path = output_dir / (topic_stem + "_projection_matrix.jpg")
        cv2.imwrite(str(contact_path), contact_sheet,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        camera_report["projection_matrix"] = str(contact_path)
        report["cameras"][topic] = camera_report
        print("PROJECTION_MATRIX_OK", topic,
              "adjusted_diff_ms=%.3f" % camera_report["adjusted_camera_minus_lidar_ms"],
              "output=", contact_path, flush=True)

    report_path = output_dir / "projection_contract_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print("PROJECTION_CONTRACT_DIAGNOSTIC_OK", output_dir, flush=True)


if __name__ == "__main__":
    main()
