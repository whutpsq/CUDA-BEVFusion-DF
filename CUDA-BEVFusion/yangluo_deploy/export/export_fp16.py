#!/usr/bin/env python3
"""Export the isolated two-camera Yangluo ResNet50 model as FP16-ready ONNX."""

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bevfusion-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--num-cameras", type=int, default=2)
    return parser.parse_args()


def export_camera(args, utilities):
    import onnx
    import torch
    from torch import nn

    try:
        from onnxsim import simplify
    except ImportError:
        simplify = None

    class CameraModule(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, img, depth):
            batch, cameras, channels, height, width = img.size()
            img = img.view(batch * cameras, channels, height, width)
            feat = self.model.encoders.camera.backbone(img)
            feat = self.model.encoders.camera.neck(feat)
            if not isinstance(feat, torch.Tensor):
                feat = feat[0]
            bn, channels, height, width = map(int, feat.size())
            feat = feat.view(batch, int(bn / batch), channels, height, width)
            vtransform = self.model.encoders.camera.vtransform
            batch, cameras, channels, feat_h, feat_w = map(int, feat.shape)
            depth = depth.view(batch * cameras, *depth.shape[2:])
            feat = feat.view(batch * cameras, channels, feat_h, feat_w)
            depth_feat = vtransform.dtransform(depth)
            result = vtransform.depthnet(torch.cat([depth_feat, feat], dim=1))
            depth_weights = result[:, : vtransform.D].softmax(dim=1)
            camera_feature = result[:, vtransform.D : vtransform.D + vtransform.C].permute(0, 2, 3, 1)
            return camera_feature, depth_weights

    model, cfg = utilities.load_bevfusion_model(args.config, args.ckpt)
    utilities.assert_checkpoint_matches_object_classes(cfg, args.ckpt)
    shapes = utilities.camera_shapes(cfg)
    output = Path(args.save_root)
    output.mkdir(parents=True, exist_ok=True)

    camera_model = CameraModule(model).cuda().eval()
    downsample = model.encoders.camera.vtransform.downsample.cuda().eval()
    image = torch.zeros(
        1, args.num_cameras, 3, shapes["image_h"], shapes["image_w"], device="cuda"
    )
    depth = torch.zeros(
        1, args.num_cameras, 1, shapes["image_h"], shapes["image_w"], device="cuda"
    )
    downsample_input = torch.zeros(
        1, shapes["camera_channels"], shapes["bev_h"], shapes["bev_w"], device="cuda"
    )

    camera_path = output / "camera.backbone.onnx"
    vtransform_path = output / "camera.vtransform.onnx"
    with torch.no_grad():
        torch.onnx.export(
            camera_model,
            (image, depth),
            str(camera_path),
            input_names=["img", "depth"],
            output_names=["camera_feature", "camera_depth_weights"],
            opset_version=13,
            do_constant_folding=True,
        )
        if simplify is not None:
            simplified, valid = simplify(onnx.load(str(camera_path)))
            if not valid:
                raise RuntimeError("simplified camera ONNX validation failed")
            onnx.save(simplified, str(camera_path))
        torch.onnx.export(
            downsample,
            downsample_input,
            str(vtransform_path),
            input_names=["feat_in"],
            output_names=["feat_out"],
            opset_version=13,
            do_constant_folding=True,
        )
    print(f"PASS camera_onnx={camera_path} num_cameras={args.num_cameras}")


def run_isolated_exporters(args):
    export_root = Path(__file__).resolve().parent
    python = sys.executable
    commands = [
        [
            python,
            str(export_root / "export_lidar.py"),
            "--bevfusion-root",
            args.bevfusion_root,
            "--config",
            args.config,
            "--ckpt",
            args.ckpt,
            "--save",
            str(Path(args.save_root) / "lidar.backbone.onnx"),
        ],
        [
            python,
            str(export_root / "export_fuser_head.py"),
            "--bevfusion-root",
            args.bevfusion_root,
            "--config",
            args.config,
            "--ckpt",
            args.ckpt,
            "--save-root",
            args.save_root,
        ],
    ]
    for command in commands:
        print("RUN", " ".join(command))
        subprocess.run(command, check=True, cwd=str(export_root))


def validate_outputs(save_root):
    required = [
        "camera.backbone.onnx",
        "camera.vtransform.onnx",
        "lidar.backbone.xyz.onnx",
        "fuser.onnx",
        "head.bbox.onnx",
    ]
    missing = [name for name in required if not (Path(save_root) / name).is_file()]
    empty = [name for name in required if (Path(save_root) / name).is_file() and (Path(save_root) / name).stat().st_size == 0]
    if missing or empty:
        raise RuntimeError(f"ONNX export incomplete: missing={missing}, empty={empty}")
    print("PASS fp16_onnx=" + ",".join(required))


def main():
    args = parse_args()
    if args.num_cameras != 2:
        raise RuntimeError("Yangluo runtime contract requires exactly two cameras")
    export_root = Path(__file__).resolve().parent
    training_root = Path(args.bevfusion_root).resolve()
    sys.path.insert(0, str(training_root))
    sys.path.insert(0, str(export_root))
    import df_export_utils as utilities

    Path(args.save_root).mkdir(parents=True, exist_ok=True)
    export_camera(args, utilities)
    run_isolated_exporters(args)
    validate_outputs(args.save_root)


if __name__ == "__main__":
    main()
