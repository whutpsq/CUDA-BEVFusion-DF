#!/usr/bin/env python3
"""Keep only pretrained tensors whose names and shapes fit a target config."""

import argparse
from pathlib import Path

import torch
from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("source")
    parser.add_argument("output")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)
    target = build_model(cfg.model).state_dict()

    checkpoint = torch.load(args.source, map_location="cpu")
    source = checkpoint.get("state_dict", checkpoint)
    if not isinstance(source, dict):
        raise ValueError("checkpoint has no state_dict-like mapping")

    compatible = {}
    mismatched = []
    unexpected = []
    for source_name, value in source.items():
        target_name = source_name
        if target_name not in target and target_name.startswith("module."):
            target_name = target_name[len("module.") :]
        if target_name not in target:
            unexpected.append(source_name)
        elif tuple(value.shape) != tuple(target[target_name].shape):
            mismatched.append(
                f"{source_name}: {tuple(value.shape)} -> {tuple(target[target_name].shape)}"
            )
        else:
            compatible[target_name] = value

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(checkpoint) if isinstance(checkpoint, dict) else {}
    result["state_dict"] = compatible
    result.setdefault("meta", {})
    result["meta"] = dict(result["meta"])
    result["meta"]["compatible_for_config"] = str(args.config)
    torch.save(result, output)

    missing = len(target) - len(compatible)
    print(f"wrote {output}")
    print(
        f"kept={len(compatible)} missing_target={missing} "
        f"shape_mismatch={len(mismatched)} unexpected_source={len(unexpected)}"
    )
    if mismatched:
        print("shape mismatches (expected for class-dependent heads):")
        for item in mismatched:
            print(f"  {item}")
    print("PASS compatible checkpoint")


if __name__ == "__main__":
    main()

