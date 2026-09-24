#!/usr/bin/env python3
"""Capture live Yangluo cameras/LiDAR and render calibration overlays."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CompressedImage, PointCloud2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--front-topic", default="/cam5/compressed")
    parser.add_argument("--rear-topic", default="/cam10/compressed")
    parser.add_argument("--lidar-topic", default="/perception/lidar/concated_points_cloud")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--draw-stride", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.calibration, "r", encoding="utf-8") as stream:
        calibration = json.load(stream)

    rospy.init_node("yangluo_live_lidar_projection", anonymous=True, disable_signals=True)
    print("WAITING_FOR_LIVE_MESSAGES", flush=True)
    camera_messages = {
        "front": rospy.wait_for_message(args.front_topic, CompressedImage, timeout=args.timeout),
        "rear": rospy.wait_for_message(args.rear_topic, CompressedImage, timeout=args.timeout),
    }
    lidar_message = rospy.wait_for_message(args.lidar_topic, PointCloud2, timeout=args.timeout)

    points = np.asarray(
        list(
            point_cloud2.read_points(
                lidar_message,
                field_names=["x", "y", "z", "intensity"],
                skip_nans=True,
            )
        ),
        dtype=np.float64,
    )
    xyz = points[:, :3]
    report = {
        "calibration": str(Path(args.calibration).resolve()),
        "lidar_topic": args.lidar_topic,
        "lidar_stamp": lidar_message.header.stamp.to_sec(),
        "lidar_frame_id": lidar_message.header.frame_id,
        "point_count": int(xyz.shape[0]),
        "cameras": {},
    }

    topics = {"front": args.front_topic, "rear": args.rear_topic}
    filenames = {"front": "cam5", "rear": "cam10"}
    for camera_name, message in camera_messages.items():
        image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode " + camera_name)

        camera = calibration["cameras"][camera_name]
        camera2ego = np.asarray(camera["camera2ego"], dtype=np.float64)
        ego2camera = np.linalg.inv(camera2ego)
        rotation = ego2camera[:3, :3]
        translation = ego2camera[:3, 3]
        rotation_vector, _ = cv2.Rodrigues(rotation)
        intrinsic = np.asarray(camera["camera_intrinsics"], dtype=np.float64)[:3, :3]
        distortion = np.asarray(camera["distortion"], dtype=np.float64)

        camera_xyz = (rotation @ xyz.T).T + translation.reshape(1, 3)
        projected, _ = cv2.projectPoints(
            xyz.reshape(-1, 1, 3), rotation_vector, translation, intrinsic, distortion
        )
        projected = projected.reshape(-1, 2)
        height, width = image.shape[:2]
        depth = camera_xyz[:, 2]
        valid = (
            np.isfinite(projected).all(axis=1)
            & (depth > 0.5)
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < height)
        )
        valid_indices = np.flatnonzero(valid)

        stem = filenames[camera_name]
        raw_path = output_dir / (stem + "_raw.jpg")
        overlay_path = output_dir / (stem + "_lidar_overlay.jpg")
        cv2.imwrite(str(raw_path), image)
        overlay = image.copy()
        for index in valid_indices[:: max(1, args.draw_stride)]:
            u = int(round(projected[index, 0]))
            v = int(round(projected[index, 1]))
            ratio = max(0.0, min(float(depth[index]), 60.0)) / 60.0
            color = (
                int(255 * ratio),
                int(255 * (1.0 - abs(ratio - 0.5) * 2.0)),
                int(255 * (1.0 - ratio)),
            )
            cv2.circle(overlay, (u, v), 1, color, -1, lineType=cv2.LINE_AA)

        label = "%s source=%s projected=%d" % (
            camera_name,
            camera.get("source_camera_id", "unknown"),
            len(valid_indices),
        )
        cv2.putText(
            overlay,
            label,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(overlay_path), overlay)

        stamp = message.header.stamp.to_sec()
        report["cameras"][camera_name] = {
            "topic": topics[camera_name],
            "source_camera_id": camera.get("source_camera_id"),
            "camera_stamp": stamp,
            "camera_minus_lidar_ms": (stamp - lidar_message.header.stamp.to_sec()) * 1000.0,
            "image_shape": list(image.shape),
            "positive_depth_points": int((depth > 0.5).sum()),
            "projected_inside_image": int(valid.sum()),
            "projected_inside_ratio": float(valid.mean()),
            "raw_image": str(raw_path),
            "overlay_image": str(overlay_path),
        }
        print(
            "PROJECTION_OK",
            camera_name,
            "source_id=",
            camera.get("source_camera_id"),
            "inside_image=",
            int(valid.sum()),
            "overlay=",
            overlay_path,
            flush=True,
        )

    report_path = output_dir / "projection_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("LIVE_LIDAR_PROJECTION_DONE", output_dir)


if __name__ == "__main__":
    main()
