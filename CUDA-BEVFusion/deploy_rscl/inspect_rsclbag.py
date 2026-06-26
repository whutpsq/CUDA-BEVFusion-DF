from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect RSCL bag message fields and payload shapes.")
    parser.add_argument("--bag", required=True, help="Path to .rsclbag.")
    parser.add_argument(
        "--max-messages",
        type=int,
        default=50,
        help="Stop after this many messages have been read from the bag.",
    )
    parser.add_argument(
        "--max-samples-per-topic",
        type=int,
        default=1,
        help="Print at most this many samples for each topic.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Optional topic filter. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bag_path = str(Path(args.bag).resolve())

    import rsclpy

    reader = open_reader(rsclpy, bag_path, set(args.include) if args.include else None)
    if not reader.is_valid():
        raise RuntimeError(f"Invalid rsclbag: {bag_path}")

    print(f"bag={bag_path}")
    header = safe_call(reader, "get_bag_header")
    if header is not None:
      # Header objects differ across SDK builds. Keep output defensive.
        print(f"bag_header_type={type(header).__name__}")
        print(f"bag_header_fields={field_names(header)}")
        print(f"bag_header_summary={summarize_object(header, depth=1)}")

    seen_per_topic: Dict[str, int] = defaultdict(int)
    total = 0

    while total < args.max_messages:
        msg = safe_call(reader, "read_next_message")
        if msg is None:
            break
        total += 1

        topic = channel_name(msg)
        if args.include and topic not in args.include:
            continue

        if seen_per_topic[topic] >= args.max_samples_per_topic:
            continue
        seen_per_topic[topic] += 1

        print("")
        print(f"[sample {total}] topic={topic}")
        print(f"  msg_type={type(msg).__name__}")
        print(f"  msg_fields={field_names(msg)}")
        print(f"  msg_summary={summarize_object(msg, depth=1)}")

        payload = message_payload(msg)
        print(f"  payload_type={type(payload).__name__}")
        print(f"  payload_summary={summarize_value(payload, depth=2)}")

        if isinstance(payload, dict):
            print(f"  payload_keys={sorted(payload.keys())}")
            print(f"  timestamp_us_candidate={extract_timestamp_us(payload, wrapper=msg)}")
            if "data" in payload:
                print(f"  data_type={type(payload['data']).__name__}")
                print(f"  data_summary={summarize_value(payload['data'], depth=1)}")
            if "points" in payload:
                print(f"  points_type={type(payload['points']).__name__}")
                print(f"  points_summary={summarize_value(payload['points'], depth=1)}")
        else:
            print(f"  timestamp_us_candidate={extract_timestamp_us(payload, wrapper=msg)}")
            raw = to_bytes(payload)
            print(f"  raw_len={len(raw)}")
            print(f"  raw_head_hex={raw[:32].hex()}")

    print("")
    print(f"messages_read={total}")
    print(f"topics_seen={dict(seen_per_topic)}")


def open_reader(rsclpy: Any, bag_path: str, include_channels: Optional[set[str]]) -> Any:
    if hasattr(rsclpy, "BagReaderAttribute"):
        attr = rsclpy.BagReaderAttribute()
        if include_channels is not None:
            attr.included_channels = include_channels
        return rsclpy.BagReader(bag_path, attr)
    return rsclpy.BagReader(bag_path)


def safe_call(obj: Any, name: str) -> Any:
    if not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    return value() if callable(value) else value


def channel_name(msg: Any) -> str:
    for attr in ("channel_name", "channelName", "topic", "topic_name"):
        if hasattr(msg, attr):
            value = getattr(msg, attr)
            return str(value() if callable(value) else value)
    return "<unknown>"


def message_payload(msg: Any) -> Any:
    if hasattr(msg, "message_obj") and getattr(msg, "message_obj") is not None:
        return getattr(msg, "message_obj")
    if hasattr(msg, "message_json") and getattr(msg, "message_json") is not None:
        value = getattr(msg, "message_json")
        if isinstance(value, dict):
            return value
        return object_to_mapping(value)
    try:
        return json.loads(to_bytes(msg).decode("utf-8"))
    except Exception:
        return msg


def object_to_mapping(obj: Any) -> Dict[str, Any]:
    return {name: maybe_get(obj, name) for name in field_names(obj)}


def field_names(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return sorted(obj.keys())
    names = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if not callable(value):
            names.append(name)
    return names


def maybe_get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    if hasattr(obj, name):
        value = getattr(obj, name)
        return value() if callable(value) else value
    return None


def summarize_object(obj: Any, depth: int = 1) -> str:
    if depth <= 0:
        return repr(obj)
    if isinstance(obj, dict):
        items = []
        for key in sorted(obj.keys())[:20]:
            items.append(f"{key}={summarize_value(obj[key], depth - 1)}")
        extra = "" if len(obj) <= 20 else f", ... (+{len(obj) - 20} keys)"
        return "{" + ", ".join(items) + extra + "}"
    return summarize_value(obj, depth)


def summarize_value(value: Any, depth: int = 1) -> str:
    if depth <= 0:
        return repr(value)
    if isinstance(value, dict):
        return summarize_object(value, depth)
    if isinstance(value, (list, tuple)):
        preview = [summarize_value(v, depth - 1) for v in value[:8]]
        extra = "" if len(value) <= 8 else f", ... (+{len(value) - 8} items)"
        return "[" + ", ".join(preview) + extra + "]"
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return f"bytes(len={len(raw)}, head={raw[:16].hex()})"
    return repr(value)


def to_bytes(msg: Any) -> bytes:
    if isinstance(msg, bytes):
        return msg
    if isinstance(msg, bytearray):
        return bytes(msg)
    if isinstance(msg, memoryview):
        return msg.tobytes()
    if isinstance(msg, str):
        return msg.encode("utf-8", errors="replace")
    for attr in ("data", "payload", "raw", "buffer"):
        if hasattr(msg, attr):
            value = getattr(msg, attr)
            if callable(value):
                value = value()
            return to_bytes(value)
    if hasattr(msg, "as_builder"):
        return to_bytes(msg.as_builder())
    raise TypeError(f"Unsupported payload type: {type(msg)!r}")


def extract_timestamp_us(data: Any, wrapper: Any = None) -> int:
    for source in (data, wrapper):
        if source is None:
            continue
        value = first_value(
            source,
            (
                "timestamp_us",
                "timestampUs",
                "time_us",
                "timestamp_ns",
                "timestampNs",
                "time_ns",
                "headerTimestamp",
                "header_time",
                "timestamp_ms",
                "timestampMs",
                "timestamp",
                "time",
            ),
        )
        if value is not None:
            return normalize_timestamp_us(value)
        header = maybe_get(source, "header")
        if header is not None:
            try:
                return extract_timestamp_us(header)
            except Exception:
                pass
    raise ValueError("No recognizable timestamp field found")


def first_value(obj: Any, names: Iterable[str]) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return None
    for name in names:
        value = maybe_get(obj, name)
        if value is not None:
            return value
    return None


def normalize_timestamp_us(value: Any) -> int:
    value_int = int(float(value))
    if value_int > 10_000_000_000_000_000:
        return value_int // 1000
    if value_int > 10_000_000_000_000:
        return value_int
    if value_int > 10_000_000_000:
        return value_int * 1000
    return value_int


if __name__ == "__main__":
    main()
