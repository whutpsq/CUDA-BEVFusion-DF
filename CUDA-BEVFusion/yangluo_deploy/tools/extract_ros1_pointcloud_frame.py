#!/usr/bin/env python3
"""Extract one PointCloud2 frame from an uncompressed ROS1 bag as binary PCD."""

import argparse
import io
import json
import math
from pathlib import Path
import struct
from typing import BinaryIO, Dict, Iterator, Optional, Tuple


OP_MSG_DATA = 0x02
OP_CHUNK = 0x05
OP_CONNECTION = 0x07


def read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"expected {size} bytes, got {len(data)}")
    return data


def read_u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(stream, 4))[0]


def parse_fields(raw: bytes) -> Dict[str, bytes]:
    result: Dict[str, bytes] = {}
    offset = 0
    while offset < len(raw):
        if offset + 4 > len(raw):
            raise ValueError("truncated bag header field length")
        size = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        value = raw[offset : offset + size]
        if len(value) != size:
            raise ValueError("truncated bag header field")
        offset += size
        key, separator, payload = value.partition(b"=")
        if not separator:
            raise ValueError("invalid bag header field")
        result[key.decode("ascii")] = payload
    return result


def iter_records(stream: BinaryIO) -> Iterator[Tuple[Dict[str, bytes], bytes]]:
    while True:
        length_raw = stream.read(4)
        if not length_raw:
            return
        if len(length_raw) != 4:
            raise EOFError("truncated record header length")
        header_size = struct.unpack("<I", length_raw)[0]
        header = parse_fields(read_exact(stream, header_size))
        data_size = read_u32(stream)
        yield header, read_exact(stream, data_size)


def field_u32(header: Dict[str, bytes], name: str) -> int:
    return struct.unpack("<I", header[name])[0]


def field_time(header: Dict[str, bytes], name: str = "time") -> float:
    sec, nsec = struct.unpack("<II", header[name])
    return sec + nsec * 1.0e-9


def read_ros_string(data: bytes, offset: int) -> Tuple[str, int]:
    size = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    value = data[offset : offset + size].decode("utf-8", errors="replace")
    return value, offset + size


def parse_pointcloud2(data: bytes) -> Dict[str, object]:
    offset = 0
    seq, stamp_sec, stamp_nsec = struct.unpack_from("<III", data, offset)
    offset += 12
    frame_id, offset = read_ros_string(data, offset)
    height, width = struct.unpack_from("<II", data, offset)
    offset += 8
    field_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    fields = []
    for _ in range(field_count):
        name, offset = read_ros_string(data, offset)
        field_offset = struct.unpack_from("<I", data, offset)[0]
        datatype = data[offset + 4]
        count = struct.unpack_from("<I", data, offset + 5)[0]
        offset += 9
        fields.append(
            {"name": name, "offset": field_offset, "datatype": datatype, "count": count}
        )
    is_bigendian = bool(data[offset])
    point_step, row_step = struct.unpack_from("<II", data, offset + 1)
    offset += 9
    payload_size = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    payload = data[offset : offset + payload_size]
    offset += payload_size
    if len(payload) != payload_size or offset >= len(data):
        raise ValueError("truncated PointCloud2 payload")
    is_dense = bool(data[offset])
    return {
        "seq": seq,
        "stamp": stamp_sec + stamp_nsec * 1.0e-9,
        "stamp_sec": stamp_sec,
        "stamp_nsec": stamp_nsec,
        "frame_id": frame_id,
        "height": height,
        "width": width,
        "fields": fields,
        "is_bigendian": is_bigendian,
        "point_step": point_step,
        "row_step": row_step,
        "data": payload,
        "is_dense": is_dense,
    }


POINT_FORMATS = {
    1: "b",
    2: "B",
    3: "h",
    4: "H",
    5: "i",
    6: "I",
    7: "f",
    8: "d",
}

POINT_SIZES = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}
PCD_TYPES = {1: "I", 2: "U", 3: "I", 4: "U", 5: "I", 6: "U", 7: "F", 8: "F"}


def point_value(payload: bytes, base: int, field: Dict[str, object], endian: str) -> float:
    datatype = int(field["datatype"])
    if datatype not in POINT_FORMATS or int(field["count"]) != 1:
        raise ValueError(f"unsupported PointField: {field}")
    return float(struct.unpack_from(endian + POINT_FORMATS[datatype], payload, base + int(field["offset"]))[0])


def write_pcd(path: Path, cloud: Dict[str, object], preserve_all_fields: bool) -> Tuple[int, list]:
    fields_by_name = {str(item["name"]): item for item in cloud["fields"]}
    required = ("x", "y", "z", "intensity")
    missing = [name for name in required if name not in fields_by_name]
    if missing:
        raise ValueError(f"PointCloud2 missing fields: {missing}")

    payload = cloud["data"]
    point_step = int(cloud["point_step"])
    row_step = int(cloud["row_step"])
    width = int(cloud["width"])
    height = int(cloud["height"])
    endian = ">" if cloud["is_bigendian"] else "<"
    selected_fields = list(cloud["fields"]) if preserve_all_fields else [fields_by_name[name] for name in required]
    unsupported = [
        field for field in selected_fields
        if int(field["datatype"]) not in POINT_FORMATS or int(field["count"]) != 1
    ]
    if unsupported:
        raise ValueError(f"unsupported fields for PCD output: {unsupported}")

    packed = bytearray()
    for row in range(height):
        row_base = row * row_step
        for column in range(width):
            base = row_base + column * point_step
            xyz = tuple(point_value(payload, base, fields_by_name[name], endian) for name in ("x", "y", "z"))
            if not all(math.isfinite(value) for value in xyz):
                continue
            for field in selected_fields:
                datatype = int(field["datatype"])
                source_format = endian + POINT_FORMATS[datatype]
                value = struct.unpack_from(source_format, payload, base + int(field["offset"]))[0]
                packed.extend(struct.pack("<" + POINT_FORMATS[datatype], value))

    output_point_step = sum(POINT_SIZES[int(field["datatype"])] for field in selected_fields)
    count = len(packed) // output_point_step
    field_names = [str(field["name"]) for field in selected_fields]
    field_sizes = [str(POINT_SIZES[int(field["datatype"])]) for field in selected_fields]
    field_types = [PCD_TYPES[int(field["datatype"])] for field in selected_fields]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(field_names)}\n"
        f"SIZE {' '.join(field_sizes)}\n"
        f"TYPE {' '.join(field_types)}\n"
        f"COUNT {' '.join('1' for _ in selected_fields)}\n"
        f"WIDTH {count}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {count}\n"
        "DATA binary\n"
    ).encode("ascii")
    path.write_bytes(header + packed)
    return count, field_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--topic", default="/perception/lidar/concated_points_cloud")
    parser.add_argument("--offset", type=float, default=10.0, help="Seconds after bag start")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preserve-all-fields", action="store_true",
        help="Write every supported scalar PointCloud2 field instead of XYZI only",
    )
    args = parser.parse_args()

    connections: Dict[int, Dict[str, str]] = {}
    bag_start: Optional[float] = None
    target: Optional[float] = None
    best: Optional[Tuple[float, float, bytes]] = None
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
                if bag_start is None:
                    bag_start = record_time
                    target = bag_start + args.offset
                connection = connections.get(field_u32(inner_header, "conn"))
                if connection and connection["topic"] == args.topic:
                    difference = abs(record_time - float(target))
                    if best is None or difference < best[0]:
                        best = difference, record_time, inner_data
                if best is not None and record_time > float(target) + 0.5:
                    done = True
                    break
            if done:
                break

    if bag_start is None or target is None:
        raise RuntimeError("bag contains no message data")
    if best is None:
        known = sorted({item["topic"] for item in connections.values()})
        raise RuntimeError(f"topic not found: {args.topic}; known topics={known}")

    difference, record_time, message_data = best
    cloud = parse_pointcloud2(message_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written_points, pcd_fields = write_pcd(args.output, cloud, args.preserve_all_fields)
    metadata_path = args.output.with_suffix(".json")
    metadata = {
        "bag": str(args.bag.resolve()),
        "topic": args.topic,
        "bag_start_time": bag_start,
        "requested_offset_seconds": args.offset,
        "target_bag_time": target,
        "selected_record_time": record_time,
        "selected_header_stamp": cloud["stamp"],
        "absolute_difference_ms": difference * 1000.0,
        "seq": cloud["seq"],
        "frame_id": cloud["frame_id"],
        "width": cloud["width"],
        "height": cloud["height"],
        "original_points": int(cloud["width"]) * int(cloud["height"]),
        "written_finite_points": written_points,
        "point_step": cloud["point_step"],
        "row_step": cloud["row_step"],
        "is_bigendian": cloud["is_bigendian"],
        "is_dense": cloud["is_dense"],
        "fields": cloud["fields"],
        "pcd_fields": pcd_fields,
        "coordinate_frame_note": "Raw PointCloud2 payload; no RFU-to-FLU conversion applied.",
        "pcd": str(args.output.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"bag_start={bag_start:.9f}")
    print(f"target_time={target:.9f}")
    print(f"selected_record_time={record_time:.9f}")
    print(f"difference_ms={difference * 1000.0:.3f}")
    print(f"header_stamp={float(cloud['stamp']):.9f}")
    print(f"frame_id={cloud['frame_id']}")
    print(f"points={written_points}")
    print(f"pcd={args.output.resolve()}")
    print(f"metadata={metadata_path.resolve()}")
    print("ROS1_POINTCLOUD_FRAME_EXTRACT_OK")


if __name__ == "__main__":
    main()
