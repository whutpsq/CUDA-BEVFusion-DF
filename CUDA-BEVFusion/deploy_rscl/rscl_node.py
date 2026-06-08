from __future__ import annotations

import argparse
import time
import traceback
from typing import Any, Optional

from .codecs import (
    decode_lidar_message,
    decode_message_timestamp_us,
    encode_detection_message,
    StatefulCameraDecoder,
    VideoFrameNotReady,
)
from .config import RsclAdapterConfig, load_adapter_config
from .preprocess import build_model_input, load_calibration
from .runner import BevFusionRunner
from .sync import FrameSynchronizer, SyncedFrame


class BevFusionRsclNode:
    def __init__(
        self,
        cfg: RsclAdapterConfig,
        decode_only: bool = False,
        decode_images_only: bool = False,
        print_rate: float = 1.0,
    ) -> None:
        import rsclpy

        self.rsclpy = rsclpy
        self.cfg = cfg
        self.decode_only = decode_only
        self.decode_images_only = decode_images_only
        self.print_interval_s = 1.0 / print_rate if print_rate > 0.0 else 0.0
        self.synced_frames = 0
        self.decode_errors = 0
        self.inference_errors = 0
        self.last_print_time = 0.0
        self.node = rsclpy.Node(cfg.node_name)
        self.sync = FrameSynchronizer(
            camera_topics=cfg.camera_topics,
            camera_order=cfg.camera_order,
            lidar_topic=cfg.lidar_topic,
            tolerance_ms=cfg.sync_tolerance_ms,
        )
        self.calibration = None
        self.runner = None
        self.publisher = None
        if not self._dry_run:
            if not cfg.calibration_file and not cfg.allow_identity_calibration:
                raise RuntimeError(
                    "calibration_file is required for CUDA-BEVFusion RSCL inference. "
                    "Identity calibration makes camera geometry empty and can produce invalid inference. "
                    "Set calibration_file in the adapter config, or set allow_identity_calibration: true for decode-only smoke tests."
                )
            self.calibration = load_calibration(cfg.calibration_file, cfg.camera_order)
            self.runner = BevFusionRunner(cfg)
            self.publisher = self.node.create_publisher(cfg.output_topic, cfg.output_message_type)
        self.camera_decoders = {
            topic: StatefulCameraDecoder() for topic in cfg.camera_topics
        }
        self._create_subscribers()
        self._print_startup()

    def spin(self) -> None:
        self.node.spin()

    @property
    def _dry_run(self) -> bool:
        return self.decode_only or self.decode_images_only

    def _create_subscribers(self) -> None:
        for topic in self.cfg.camera_topics:
            self.node.create_subscriber(
                topic,
                self.cfg.input_message_type,
                lambda msg, topic=topic: self._on_camera(topic, msg),
            )
        self.node.create_subscriber(
            self.cfg.lidar_topic,
            self.cfg.input_message_type,
            self._on_lidar,
        )

    def _on_camera(self, topic: str, msg: Any) -> None:
        try:
            if self.decode_only:
                timestamp_us = decode_message_timestamp_us(msg)
                image = None
            else:
                timestamp_us, image = self.camera_decoders[topic].decode(msg)
            frame = self.sync.add_camera(topic, timestamp_us, image)
            self._process_if_ready(frame)
        except VideoFrameNotReady:
            return
        except Exception:
            self.decode_errors += 1
            traceback.print_exc()

    def _on_lidar(self, msg: Any) -> None:
        try:
            timestamp_us, points = decode_lidar_message(msg, point_dim=self.cfg.point_dim)
            frame = self.sync.add_lidar(timestamp_us, points)
            self._process_if_ready(frame)
        except Exception:
            self.decode_errors += 1
            traceback.print_exc()

    def _process_if_ready(self, frame: Optional[SyncedFrame]) -> None:
        if frame is None:
            return
        self.synced_frames += 1
        if self._dry_run:
            self._print_status(frame, objects=None)
            return
        try:
            assert self.calibration is not None
            assert self.runner is not None
            assert self.publisher is not None
            data = build_model_input(frame, self.cfg, self.calibration)
            outputs = self.runner.infer(data)
            payload = encode_detection_message(outputs[0], self.cfg.score_threshold, frame.timestamp_us)
            if payload or self.cfg.publish_empty_frame:
                self.publisher.publish(payload)
            objects = _count_objects(payload)
            self._print_status(frame, objects=objects)
        except Exception:
            self.inference_errors += 1
            traceback.print_exc()

    def _print_startup(self) -> None:
        mode = "inference"
        if self.decode_only:
            mode = "decode-only"
        elif self.decode_images_only:
            mode = "decode-images-only"
        print(
            "RSCL BEVFusion node started "
            f"mode={mode} cameras={len(self.cfg.camera_topics)} lidar={self.cfg.lidar_topic} "
            f"sync_tolerance_ms={self.cfg.sync_tolerance_ms}"
        )
        if self._dry_run:
            print("Dry-run mode: TensorRT/libpybev is not loaded and no detections are published.")
        else:
            print(f"Publishing detections to {self.cfg.output_topic}")

    def _print_status(self, frame: SyncedFrame, objects: Optional[int]) -> None:
        now = time.monotonic()
        if self.last_print_time and self.print_interval_s > 0.0 and now - self.last_print_time < self.print_interval_s:
            return
        self.last_print_time = now
        fields = [
            f"synced_frames={self.synced_frames}",
            f"timestamp_us={frame.timestamp_us}",
            f"decode_errors={self.decode_errors}",
            f"inference_errors={self.inference_errors}",
            f"lidar_points={_payload_len(frame.lidar)}",
        ]
        if self.decode_images_only:
            fields.append(f"image_sizes={_image_sizes(frame)}")
        elif self.decode_only:
            fields.append(f"cameras={sorted(frame.cameras.keys())}")
        if objects is not None:
            fields.append(f"objects={objects}")
        print(" ".join(fields))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BEVFusion as an RSCL Python node.")
    parser.add_argument(
        "--adapter-config",
        default="deploy_rscl/configs/bevfusion_rscl.yaml",
        help="Path to the RSCL adapter yaml/json config.",
    )
    parser.add_argument("--decode-only", action="store_true", help="Subscribe and synchronize timestamps only.")
    parser.add_argument(
        "--decode-images-only",
        action="store_true",
        help="Decode camera images and lidar, synchronize frames, but do not load CUDA-BEVFusion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --decode-images-only.",
    )
    parser.add_argument(
        "--print-rate",
        type=float,
        default=1.0,
        help="Status print frequency in Hz. Set 0 to print every synchronized frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_adapter_config(args.adapter_config)
    decode_images_only = args.decode_images_only or args.dry_run

    import rsclpy

    rsclpy.init(cfg.module_name)
    node = BevFusionRsclNode(
        cfg,
        decode_only=args.decode_only,
        decode_images_only=decode_images_only,
        print_rate=args.print_rate,
    )
    node.spin()
    rsclpy.waitforshutdown()


def _count_objects(payload: str) -> int:
    try:
        import json

        return len(json.loads(payload).get("objects", []))
    except Exception:
        return -1


def _payload_len(value: Any) -> int:
    try:
        return int(len(value))
    except Exception:
        return -1


def _image_sizes(frame: SyncedFrame) -> dict[str, Any]:
    sizes = {}
    for name, image in frame.cameras.items():
        sizes[name] = list(image.size) if hasattr(image, "size") else None
    return sizes


if __name__ == "__main__":
    main()
