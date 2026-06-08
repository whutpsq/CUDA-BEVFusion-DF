from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .codecs import StatefulCameraDecoder, VideoFrameNotReady, decode_lidar_message
from .config import load_adapter_config
from .preprocess import _build_matrices, _undistort_image, load_calibration
from .sync import FrameSynchronizer, SyncedFrame
from .visualize_bag import _channel_name, _lidar_stats, _make_mosaic, _render_lidar_bev


NUSCENES_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

PALETTE = [
    (255, 80, 80),
    (80, 180, 255),
    (255, 190, 70),
    (120, 220, 120),
    (210, 120, 255),
    (255, 120, 200),
    (120, 255, 230),
    (255, 255, 120),
    (180, 180, 255),
    (255, 150, 100),
]


class DetectionVisualizer:
    def __init__(
        self,
        cfg: Any,
        detections_path: str,
        output_dir: str,
        camera_line_width: int = 8,
        bev_line_width: int = 6,
        font_size: int = 30,
    ) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_line_width = camera_line_width
        self.bev_line_width = bev_line_width
        self.font = _load_font(font_size)
        self.records = _load_detection_records(detections_path)
        self.records_by_timestamp = {
            int(record["timestamp_us"]): record for record in self.records if "timestamp_us" in record
        }
        self.sync = FrameSynchronizer(
            camera_topics=cfg.camera_topics,
            camera_order=cfg.camera_order,
            lidar_topic=cfg.lidar_topic,
            tolerance_ms=cfg.sync_tolerance_ms,
        )
        self.calibration = load_calibration(cfg.calibration_file, cfg.camera_order)
        self.matrices = _build_matrices(
            self.calibration,
            cfg.camera_order,
            cfg.calibration_extrinsic_direction,
        )
        self.camera_decoders = {topic: StatefulCameraDecoder() for topic in cfg.camera_topics}
        self.frames = 0
        self.decode_errors = 0

    def run(self, bag_path: str) -> None:
        import rsclpy

        reader = self._open_reader(rsclpy, bag_path, set([*self.cfg.camera_topics, self.cfg.lidar_topic]))
        if not reader.is_valid():
            raise RuntimeError(f"Invalid rsclbag: {bag_path}")

        header = reader.get_bag_header()
        print(f"bag begin_time={getattr(header, 'begin_time', None)} end_time={getattr(header, 'end_time', None)}")
        while True:
            msg = reader.read_next_message()
            if msg is None:
                break
            frame = self._add_message(msg)
            if frame is None:
                continue
            record = self._record_for_frame(frame)
            if record is not None:
                self._save_frame(frame, record)
                self.frames += 1
            if self.cfg.max_frames is not None and self.frames >= self.cfg.max_frames:
                break
            if self.frames >= len(self.records):
                break
        print(f"visualized_frames={self.frames} decode_errors={self.decode_errors} output_dir={self.output_dir}")

    def _open_reader(self, rsclpy: Any, bag_path: str, channels: set[str]) -> Any:
        if hasattr(rsclpy, "BagReaderAttribute"):
            attr = rsclpy.BagReaderAttribute()
            attr.included_channels = channels
            return rsclpy.BagReader(bag_path, attr)
        return rsclpy.BagReader(bag_path)

    def _add_message(self, msg: Any) -> Optional[SyncedFrame]:
        topic = _channel_name(msg)
        try:
            if topic in self.cfg.camera_topics:
                timestamp_us, image = self.camera_decoders[topic].decode(msg)
                return self.sync.add_camera(topic, timestamp_us, image)
            if topic == self.cfg.lidar_topic:
                timestamp_us, points = decode_lidar_message(msg, point_dim=self.cfg.point_dim)
                return self.sync.add_lidar(timestamp_us, points)
        except VideoFrameNotReady:
            return None
        except Exception:
            self.decode_errors += 1
            print(f"failed to decode topic={topic}")
            import traceback

            traceback.print_exc()
        return None

    def _record_for_frame(self, frame: SyncedFrame) -> Optional[Mapping[str, Any]]:
        if self.records_by_timestamp:
            return self.records_by_timestamp.get(int(frame.timestamp_us))
        if self.frames < len(self.records):
            return self.records[self.frames]
        return None

    def _save_frame(self, frame: SyncedFrame, record: Mapping[str, Any]) -> None:
        objects = _filter_objects(record.get("objects", []), min_score=0.0)
        frame_dir = self.output_dir / f"frame_{self.frames:06d}_{frame.timestamp_us}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        (frame_dir / "detections.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        drawn_cameras: Dict[str, Image.Image] = {}
        for camera_index, camera_name in enumerate(self.cfg.camera_order):
            image = frame.cameras[camera_name]
            if self.cfg.undistort_images:
                image = _undistort_image(image, self.calibration, camera_name)
            canvas = _draw_camera_detections(
                image.convert("RGB"),
                objects,
                self.matrices["lidar2image"][camera_index],
                image.size,
                self.camera_line_width,
                self.font,
            )
            canvas.save(frame_dir / f"{camera_name}_detections.jpg", quality=92)
            drawn_cameras[camera_name] = canvas

        _make_mosaic(drawn_cameras, self.cfg.camera_order).save(frame_dir / "cameras_mosaic_detections.jpg", quality=92)

        bev = _render_lidar_bev(frame.lidar, self.cfg.point_cloud_range)
        bev = _draw_bev_detections(bev, objects, self.cfg.point_cloud_range, self.bev_line_width, self.font)
        bev.save(frame_dir / "lidar_bev_detections.png")

        stats = _lidar_stats(frame.lidar, self.cfg.point_cloud_range)
        print(
            f"[{self.frames}] saved {frame_dir} "
            f"objects={len(objects)} lidar_points={stats['num_points']} in_range={stats.get('num_in_config_range')}"
        )


def _load_detection_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Detection record at line {line_no} must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"No detection records found in {path}")
    return records


def _filter_objects(objects: Any, min_score: float) -> List[Dict[str, Any]]:
    filtered = []
    for obj in objects or []:
        if not isinstance(obj, dict):
            continue
        box = obj.get("box")
        if box is None or len(box) < 7:
            continue
        score = float(obj.get("score", 0.0))
        if score < min_score:
            continue
        filtered.append(obj)
    return filtered


def _draw_camera_detections(
    image: Image.Image,
    objects: Sequence[Mapping[str, Any]],
    lidar2image: np.ndarray,
    image_size: tuple[int, int],
    line_width: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image_size
    for obj in objects:
        label = int(obj.get("label", -1))
        score = float(obj.get("score", 0.0))
        color = _color_for_label(label)
        corners = _box_corners_lidar(obj["box"])
        projected = _project_points(corners, lidar2image)
        valid = projected[:, 2] > 1e-3
        if valid.sum() < 2:
            continue
        pts = projected[:, :2]
        in_image = (
            valid
            & (pts[:, 0] >= 0)
            & (pts[:, 0] < width)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < height)
        )
        if not in_image.any():
            continue
        for a, b in BOX_EDGES:
            if valid[a] and valid[b]:
                draw.line((tuple(pts[a]), tuple(pts[b])), fill=color, width=line_width)
        x0, y0 = pts[in_image].min(axis=0)
        name = _label_name(label)
        text = f"{name} {score:.2f}"
        _draw_label(draw, (float(x0), float(y0)), text, color, font)
    return image


def _draw_bev_detections(
    image: Image.Image,
    objects: Sequence[Mapping[str, Any]],
    point_cloud_range: Iterable[float],
    line_width: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    pc_range = np.asarray(list(point_cloud_range), dtype=np.float32)
    x_min, y_min, _z_min, x_max, y_max, _z_max = pc_range[:6]
    width, height = image.size

    for obj in objects:
        label = int(obj.get("label", -1))
        score = float(obj.get("score", 0.0))
        color = _color_for_label(label)
        corners = _box_corners_lidar(obj["box"])[:4]
        pts = [_bev_xy_to_pixel(float(x), float(y), x_min, y_min, x_max, y_max, width, height) for x, y in corners[:, :2]]
        draw.line([*pts, pts[0]], fill=color, width=line_width)

        center = _bev_xy_to_pixel(float(obj["box"][0]), float(obj["box"][1]), x_min, y_min, x_max, y_max, width, height)
        heading = float(obj["box"][6])
        arrow_end = _bev_xy_to_pixel(
            float(obj["box"][0]) + math.cos(heading) * max(float(obj["box"][3]), 1.0),
            float(obj["box"][1]) + math.sin(heading) * max(float(obj["box"][3]), 1.0),
            x_min,
            y_min,
            x_max,
            y_max,
            width,
            height,
        )
        draw.line((center, arrow_end), fill=color, width=line_width)
        _draw_label(draw, (center[0] + 6, center[1] + 6), f"{_label_name(label)} {score:.2f}", color, font)
    return image


def _box_corners_lidar(box: Sequence[float]) -> np.ndarray:
    x, y, z, dx, dy, dz, yaw = [float(v) for v in box[:7]]
    local = np.asarray(
        [
            [dx / 2, dy / 2, -dz / 2],
            [dx / 2, -dy / 2, -dz / 2],
            [-dx / 2, -dy / 2, -dz / 2],
            [-dx / 2, dy / 2, -dz / 2],
            [dx / 2, dy / 2, dz / 2],
            [dx / 2, -dy / 2, dz / 2],
            [-dx / 2, -dy / 2, dz / 2],
            [-dx / 2, dy / 2, dz / 2],
        ],
        dtype=np.float32,
    )
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return local @ rot.T + np.asarray([x, y, z], dtype=np.float32)


def _project_points(points: np.ndarray, lidar2image: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    proj = homo @ lidar2image.T
    depth = proj[:, 2:3]
    xy = proj[:, :2] / np.maximum(depth, 1e-6)
    return np.concatenate([xy, depth], axis=1)


def _bev_xy_to_pixel(
    x: float,
    y: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    px = int((y - y_min) / max(y_max - y_min, 1e-6) * (width - 1))
    py = int((x_max - x) / max(x_max - x_min, 1e-6) * (height - 1))
    return px, py


def _color_for_label(label: int) -> tuple[int, int, int]:
    return PALETTE[label % len(PALETTE)] if label >= 0 else (255, 255, 255)


def _label_name(label: int) -> str:
    if 0 <= label < len(NUSCENES_CLASSES):
        return NUSCENES_CLASSES[label]
    return str(label)


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arialbd.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    anchor: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = anchor
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    top = max(0.0, y - text_h - 10.0)
    left = max(0.0, x)
    pad_x = 8
    pad_y = 5
    draw.rectangle(
        (left, top, left + text_w + 2 * pad_x, top + text_h + 2 * pad_y),
        fill=color,
    )
    draw.text(
        (left + pad_x, top + pad_y),
        text,
        fill=(0, 0, 0),
        font=font,
        stroke_width=1,
        stroke_fill=(255, 255, 255),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize RSCL BEVFusion detections on camera images and lidar BEV.")
    parser.add_argument("--adapter-config", default="deploy_rscl/configs/bevfusion_resnet50_rscl.yaml")
    parser.add_argument("--bag", required=True, help="Path to .rsclbag.")
    parser.add_argument("--detections", required=True, help="JSONL file produced by rscl_bag_runner.")
    parser.add_argument("--output-dir", default="runs/rsclbag_detection_visualization")
    parser.add_argument("--max-frames", type=int, help="Optional frame limit.")
    parser.add_argument("--camera-line-width", type=int, default=8)
    parser.add_argument("--bev-line-width", type=int, default=6)
    parser.add_argument("--font-size", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_adapter_config(args.adapter_config)
    if args.max_frames is not None:
        cfg.max_frames = args.max_frames
    visualizer = DetectionVisualizer(
        cfg,
        args.detections,
        args.output_dir,
        camera_line_width=args.camera_line_width,
        bev_line_width=args.bev_line_width,
        font_size=args.font_size,
    )
    visualizer.run(args.bag)


if __name__ == "__main__":
    main()
