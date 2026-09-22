#!/usr/bin/env python3
"""Validate the three ROS1 input topics without modifying the source bag."""

import argparse
import bisect
import math
import sys

import rosbag


CAMERAS = ["/df/camera/0/image/compressed", "/df/camera/5/image/compressed"]
LIDAR = "/perception/lidar/concated_points_cloud"
REQUIRED_FIELDS = {
    "x": (0, 7),
    "y": (4, 7),
    "z": (8, 7),
    "intensity": (16, 7),
    "timestamp": (24, 8),
    "ring": (32, 4),
    "label": (34, 4),
}


def stamp_seconds(message, record_time):
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is not None and stamp.to_nsec() != 0:
        return stamp.to_sec()
    return record_time.to_sec()


def nearest_abs(value, sorted_values):
    index = bisect.bisect_left(sorted_values, value)
    choices = []
    if index < len(sorted_values):
        choices.append(abs(value - sorted_values[index]))
    if index:
        choices.append(abs(value - sorted_values[index - 1]))
    return min(choices) if choices else math.inf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    args = parser.parse_args()
    topics = CAMERAS + [LIDAR]
    timestamps = {topic: [] for topic in topics}
    first = {}
    with rosbag.Bag(args.bag, "r") as bag:
        info = bag.get_type_and_topic_info().topics
        expected_types = {CAMERAS[0]: "sensor_msgs/CompressedImage", CAMERAS[1]: "sensor_msgs/CompressedImage",
                          LIDAR: "sensor_msgs/PointCloud2"}
        for topic, expected in expected_types.items():
            if topic not in info or info[topic].msg_type != expected:
                raise RuntimeError(f"{topic}: expected {expected}, got {getattr(info.get(topic), 'msg_type', None)}")
        for topic, message, record_time in bag.read_messages(topics=topics):
            timestamps[topic].append(stamp_seconds(message, record_time))
            first.setdefault(topic, message)

    lidar = first[LIDAR]
    fields = {field.name: (field.offset, field.datatype) for field in lidar.fields}
    if fields != REQUIRED_FIELDS:
        raise RuntimeError(f"unexpected PointCloud2 fields: {fields}")
    if lidar.point_step != 48 or lidar.header.frame_id != "base_link":
        raise RuntimeError(f"unexpected lidar layout: point_step={lidar.point_step} frame={lidar.header.frame_id}")

    tolerance = args.tolerance_ms / 1000.0
    for topic in topics:
        if not timestamps[topic]:
            raise RuntimeError(f"no messages on {topic}")
    for camera in CAMERAS:
        max_delta = max(nearest_abs(value, timestamps[camera]) for value in timestamps[LIDAR])
        if max_delta > tolerance:
            raise RuntimeError(f"{camera} max nearest delta {max_delta * 1000:.3f} ms exceeds tolerance")
        print(f"SYNC camera={camera} lidar_max_nearest_ms={max_delta * 1000:.3f}")
    print("PASS bag_contract counts=" + ",".join(f"{topic}:{len(timestamps[topic])}" for topic in topics))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL {error}", file=sys.stderr)
        sys.exit(1)

