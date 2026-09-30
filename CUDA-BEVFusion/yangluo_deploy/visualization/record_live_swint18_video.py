#!/usr/bin/env python3
"""Record a timed live Yangluo detection visualization video on ROS1."""

import argparse
from collections import Counter, deque
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, PointCloud2
from yangluo_bevfusion_msgs.msg import DetectedObjectArray

from capture_live_swint18_projection_diagnostic import draw_clipped_boxes
from capture_live_swint18_result import (
    decode_image,
    draw_bev,
    pointcloud_xyzi,
    resize_letterbox,
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
    parser.add_argument("--sync-tolerance-ms", type=float, default=55.0)
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--bev-range", type=float, default=54.0)
    parser.add_argument("--image-mode", choices=("raw", "undistorted"), default="raw")
    parser.add_argument("--output")
    parser.add_argument("--report")
    return parser.parse_args()


def add_camera_header(image, title, count):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        "%s | boxes=%d" % (title, count),
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def write_lines(canvas, lines, x, y, color=(235, 235, 235)):
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + index * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )


def compose_frame(front, rear, bev, objects, elapsed, duration, sync_diffs):
    top = np.hstack(
        (resize_letterbox(front, 960, 540), resize_letterbox(rear, 960, 540))
    )
    bottom = np.full((540, 1920, 3), 14, dtype=np.uint8)
    bottom[:, 690:1230] = resize_letterbox(bev, 540, 540)

    classes = Counter(obj.class_name for obj in objects)
    left_lines = [
        "Yangluo Swin-T 18-class FP16",
        "elapsed: %05.1f / %.1f s" % (elapsed, duration),
        "published boxes: %d" % len(objects),
        "minimum score this frame: %.2f" % (
            min([float(obj.score) for obj in objects]) if objects else 0.0
        ),
        "cam5 -> calibration 0",
        "cam10 -> calibration 10",
        "images: raw distorted",
    ]
    right_lines = [
        "ROS: /perception/bevfusion/objects",
        "LiDAR payload: RFU -> model FLU",
        "sync front: %+6.2f ms" % sync_diffs[0],
        "sync rear : %+6.2f ms" % sync_diffs[1],
        "sync lidar: %+6.2f ms" % sync_diffs[2],
        "classes:",
    ]
    right_lines.extend(
        "%s: %d" % item for item in sorted(classes.items())[:6]
    )
    write_lines(bottom, left_lines, 25, 60)
    write_lines(bottom, right_lines, 1250, 60)
    return np.vstack((top, bottom))


def main():
    args = parse_args()
    default_stem = "/home/nvidia/psq/yangluo_swint18_detection_video_" + datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    output_path = Path(args.output or (default_stem + ".mp4"))
    report_path = Path(args.report or (str(output_path) + ".json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (1920, 1080),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV failed to open MP4 writer: " + str(output_path))

    rospy.init_node("yangluo_swint18_video_recorder", anonymous=True, disable_signals=True)
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

    start_wait = time.time()
    start_time = None
    last_detection_stamp = None
    last_composite = None
    video_frames = 0
    rendered_results = 0
    dropped_sync = 0
    sync_samples = []
    class_counts = Counter()
    object_counts = []
    target_video_frames = int(round(args.duration * args.fps))

    try:
        while not rospy.is_shutdown():
            now = time.time()
            if start_time is None and now - start_wait > 30.0:
                raise RuntimeError("no synchronized input received in 30 seconds")
            if start_time is not None and now - start_time >= args.duration:
                break
            if not all(buffers.values()):
                rospy.sleep(0.02)
                continue

            detection = buffers["detections"][-1]
            detection_stamp = stamp_seconds(detection)
            if detection_stamp == last_detection_stamp:
                rospy.sleep(0.01)
                continue
            last_detection_stamp = detection_stamp

            front_msg, front_diff = select_nearest(
                buffers["front"], detection_stamp, args.camera_offsets_ms[0]
            )
            rear_msg, rear_diff = select_nearest(
                buffers["rear"], detection_stamp, args.camera_offsets_ms[1]
            )
            lidar_msg, lidar_diff = select_nearest(buffers["lidar"], detection_stamp)
            sync_error = max(abs(front_diff), abs(rear_diff), abs(lidar_diff))
            if sync_error > args.sync_tolerance_ms:
                dropped_sync += 1
                continue

            if start_time is None:
                start_time = time.time()
            elapsed = min(time.time() - start_time, args.duration)
            objects = [
                obj for obj in detection.objects if obj.score >= args.score_threshold
            ]
            front_image = decode_image(front_msg)
            rear_image = decode_image(rear_msg)
            points_flu = rfu_to_flu(pointcloud_xyzi(lidar_msg))

            front_result, front_boxes = draw_clipped_boxes(
                front_image, objects, calibration["cameras"]["front"], args.image_mode
            )
            rear_result, rear_boxes = draw_clipped_boxes(
                rear_image, objects, calibration["cameras"]["rear"], args.image_mode
            )
            front_result = add_camera_header(
                front_result, "FRONT /cam5 | calib 0", front_boxes
            )
            rear_result = add_camera_header(
                rear_result, "REAR /cam10 | calib 10", rear_boxes
            )
            bev_result = draw_bev(
                points_flu, objects, args.bev_range, 540, args.point_stride
            )
            composite = compose_frame(
                front_result,
                rear_result,
                bev_result,
                objects,
                elapsed,
                args.duration,
                (front_diff, rear_diff, lidar_diff),
            )

            desired_frames = min(
                target_video_frames,
                max(video_frames + 1, int(round(elapsed * args.fps))),
            )
            while video_frames < desired_frames:
                writer.write(composite)
                video_frames += 1
            last_composite = composite
            rendered_results += 1
            sync_samples.append((front_diff, rear_diff, lidar_diff))
            object_counts.append(len(objects))
            class_counts.update(obj.class_name for obj in objects)
    finally:
        for subscriber in subscribers:
            subscriber.unregister()
        if last_composite is not None:
            while video_frames < target_video_frames:
                writer.write(last_composite)
                video_frames += 1
        writer.release()

    if last_composite is None:
        raise RuntimeError("video contains no synchronized rendered frame")

    sync_array = np.asarray(sync_samples, dtype=np.float64)
    report = {
        "video": str(output_path),
        "requested_duration_seconds": args.duration,
        "encoded_duration_seconds": video_frames / args.fps,
        "fps": args.fps,
        "resolution": [1920, 1080],
        "encoded_frame_count": video_frames,
        "rendered_detection_results": rendered_results,
        "dropped_sync_results": dropped_sync,
        "score_threshold": args.score_threshold,
        "image_mode": args.image_mode,
        "camera_offsets_ms": list(args.camera_offsets_ms),
        "objects_per_result": {
            "minimum": min(object_counts),
            "maximum": max(object_counts),
            "mean": float(np.mean(object_counts)),
        },
        "class_counts": dict(class_counts),
        "sync_absolute_ms_mean": {
            "front": float(np.mean(np.abs(sync_array[:, 0]))),
            "rear": float(np.mean(np.abs(sync_array[:, 1]))),
            "lidar": float(np.mean(np.abs(sync_array[:, 2]))),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("SWINT18_VIDEO_RECORD_OK", flush=True)
    print("video=" + str(output_path), flush=True)
    print("report=" + str(report_path), flush=True)
    print(
        "encoded_frames=%d duration=%.1f rendered_results=%d dropped_sync=%d"
        % (video_frames, video_frames / args.fps, rendered_results, dropped_sync),
        flush=True,
    )


if __name__ == "__main__":
    main()
