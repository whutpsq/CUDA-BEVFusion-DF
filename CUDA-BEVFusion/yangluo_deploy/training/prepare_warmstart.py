#!/usr/bin/env python3
"""Drop architecture-specific Swin/neck and optimizer data from a checkpoint."""

import argparse
from pathlib import Path

import torch


DROP_PREFIXES = (
    "encoders.camera.backbone.",
    "encoders.camera.neck.",
    "module.encoders.camera.backbone.",
    "module.encoders.camera.neck.",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-classes", type=int, default=11)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    kept = {key: value for key, value in state.items() if not key.startswith(DROP_PREFIXES)}
    dropped = len(state) - len(kept)
    head = next(
        (value for key, value in kept.items() if key.endswith("heads.object.heatmap_head.1.weight")),
        None,
    )
    if head is None or int(head.shape[0]) != args.num_classes:
        raise RuntimeError(f"expected a {args.num_classes}-class Yangluo object head")

    output = {
        "meta": {
            "source_checkpoint": str(Path(args.input).resolve()),
            "purpose": "ResNet50 warm start; Swin backbone and camera neck removed",
            "object_classes": args.num_classes,
        },
        "state_dict": kept,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    print(f"PASS kept={len(kept)} dropped={dropped} output={destination}")


if __name__ == "__main__":
    main()
