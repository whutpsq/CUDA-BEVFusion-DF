#!/usr/bin/env python3
"""Extract camera frames synchronized to an extracted ROS1 PointCloud2 frame."""

import argparse
import io
import json
from pathlib import Path
import struct
from typing import Dict, Optional, Tuple

from extract_ros1_pointcloud_frame import (
    OP_CHUNK,
    OP_CONNECTION,
    OP_MSG_DATA,
    field_time,
    field_u32,
    iter_records,
    parse_fields,
    read_ros_string,
)


def parse_compressed_image(data: bytes) -> Dict[str, object]:
    seq, stamp_sec, stamp_nsec = struct.unpack_from("<III", data, 0)
    offset = 12
    frame_id, offset = read_ros_string(data, offset)
    image_format, offset = read_ros_string(data, offset)
    size = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    payload = data[offset : offset + size]
    if len(payload) != size:
        raise ValueError("truncated CompressedImage payload")
    return {
        "seq": seq,
        "stamp": stamp_sec + stamp_nsec * 1.0e-9,
        "stamp_sec": stamp_sec,
        "stamp_nsec": stamp_nsec,
        "frame_id": frame_id,
        "format": image_format,
        "data": payload,
    }


def image_extension(image_format: str, payload: bytes) -> str:
    normalized = image_format.lower()
    if "png" in normalized or payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if "jpeg" in normalized or "jpg" in normalized or payload.startswith(b"\xff\xd8"):
        return ".jpg"
    return ".bin"


def jpeg_size(payload: bytes) -> Optional[Tuple[int, int]]:
    if not payload.startswith(b"\xff\xd8"):
        return None
    offset = 2
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while offset + 4 <= len(payload):
        if payload[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            break
        marker = payload[offset]
        offset += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(payload):
            break
        segment_size = struct.unpack_from(">H", payload, offset)[0]
        if marker in sof_markers and offset + 7 <= len(payload):
            height, width = struct.unpack_from(">HH", payload, offset + 3)
            return width, height
        if segment_size < 2:
            break
        offset += segment_size
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--lidar-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--camera-topic", action="append", dest="camera_topics",
        default=None, help="Repeat for each camera topic",
    )
    parser.add_argument(
        "--camera-offset-ms", type=float, default=48.0,
        help="Value added to camera header stamps by the runtime synchronizer",
    )
    args = parser.parse_args()
    topics = args.camera_topics or ["/cam5/compressed", "/cam10/compressed"]

    lidar_metadata = json.loads(args.lidar_metadata.read_text(encoding="utf-8"))
    lidar_stamp = float(lidar_metadata["selected_header_stamp"])
    lidar_record_time = float(lidar_metadata["selected_record_time"])
    requested_offset = float(lidar_metadata["requested_offset_seconds"])
    offset_label = f"{requested_offset:g}s".replace(".", "p")
    camera_offset = args.camera_offset_ms / 1000.0
    connections: Dict[int, Dict[str, str]] = {}
    best: Dict[str, Tuple[float, float, Dict[str, object]]] = {}
    done = False

    with args.bag.open("rb") as stream:
        version = stream.readline()
        if version != b"#ROSBAG V2.0\n":
            raise ValueError(f"unsupported bag version: {version!r}")
        for header, data in iter_records(stream):
            op = header.get("op", b"\x00")[0]
            if op == OP_CONNECTION:
                connection_header = parse_fields(data)
                connections[field_u32(header, "conn")] = {
                    "topic": header["topic"].decode("utf-8"),
                    "type": connection_header.get("type", b"").decode("utf-8"),
                }
                continue
            if op != OP_CHUNK:
                continue
            compression = header.get("compression", b"").decode("ascii")
            if compression != "none":
                raise ValueError(f"unsupported bag compression: {compression}")
            for inner_header, inner_data in iter_records(io.BytesIO(data)):
                inner_op = inner_header.get("op", b"\x00")[0]
                if inner_op == OP_CONNECTION:
                    connection_header = parse_fields(inner_data)
                    connections[field_u32(inner_header, "conn")] = {
                        "topic": inner_header["topic"].decode("utf-8"),
                        "type": connection_header.get("type", b"").decode("utf-8"),
                    }
                    continue
                if inner_op != OP_MSG_DATA:
                    continue
                record_time = field_time(inner_header)
                connection = connections.get(field_u32(inner_header, "conn"))
                if connection and connection["topic"] in topics:
                    image = parse_compressed_image(inner_data)
                    adjusted_difference = abs(float(image["stamp"]) + camera_offset - lidar_stamp)
                    previous = best.get(connection["topic"])
                    if previous is None or adjusted_difference < previous[0]:
                        best[connection["topic"]] = adjusted_difference, record_time, image
                if len(best) == len(topics) and record_time > lidar_record_time + 1.0:
                    done = True
                    break
            if done:
                break

    missing = [topic for topic in topics if topic not in best]
    if missing:
        raise RuntimeError(f"camera topics not found: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_metadata = {
        "bag": str(args.bag.resolve()),
        "lidar_metadata": str(args.lidar_metadata.resolve()),
        "lidar_header_stamp": lidar_stamp,
        "lidar_record_time": lidar_record_time,
        "camera_offset_ms": args.camera_offset_ms,
        "matching_rule": "minimize abs(camera_header_stamp + offset - lidar_header_stamp)",
        "images": [],
    }
    for topic in topics:
        difference, record_time, image = best[topic]
        payload = image.pop("data")
        extension = image_extension(str(image["format"]), payload)
        camera_name = topic.strip("/").split("/")[0]
        output_path = args.output_dir / f"{camera_name}_frame_{offset_label}_synced{extension}"
        output_path.write_bytes(payload)
        size = jpeg_size(payload)
        item = {
            "topic": topic,
            "record_time": record_time,
            **image,
            "raw_camera_minus_lidar_ms": (float(image["stamp"]) - lidar_stamp) * 1000.0,
            "adjusted_camera_minus_lidar_ms": (
                float(image["stamp"]) + camera_offset - lidar_stamp
            ) * 1000.0,
            "absolute_adjusted_difference_ms": difference * 1000.0,
            "encoded_bytes": len(payload),
            "image_size": list(size) if size else None,
            "output": str(output_path.resolve()),
        }
        output_metadata["images"].append(item)
        print(
            f"topic={topic} stamp={float(image['stamp']):.9f} "
            f"adjusted_diff_ms={item['adjusted_camera_minus_lidar_ms']:.3f} "
            f"size={size} output={output_path.resolve()}"
        )

    metadata_path = args.output_dir / f"camera_frames_{offset_label}_synced.json"
    metadata_path.write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"metadata={metadata_path.resolve()}")
    print("ROS1_SYNCED_IMAGES_EXTRACT_OK")


if __name__ == "__main__":
    main()
