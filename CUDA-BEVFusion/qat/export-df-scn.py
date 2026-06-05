import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import torch

from df_export_utils import add_project_root, default_checkpoint, load_bevfusion_model


def parse_args():
    parser = argparse.ArgumentParser("Export bevfusion_df lidar sparse backbone ONNX")
    parser.add_argument("--bevfusion-root", default=os.environ.get("BEVFUSION_DF_ROOT", r"H:\df_code\bevfusion"))
    parser.add_argument("--config", default=r"H:\df_code\bevfusion\configs\custom_dataset\default.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--save", default="model/bevfusion_df/lidar.backbone.onnx")
    parser.add_argument("--inverse", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    add_project_root(args.bevfusion_root)
    if args.ckpt is None:
        args.ckpt = default_checkpoint(args.bevfusion_root)

    from lean import exptool, funcs

    save = args.save
    if args.inverse:
        save = os.path.splitext(save)[0] + ".zyx.onnx"
    else:
        save = os.path.splitext(save)[0] + ".xyz.onnx"

    os.makedirs(os.path.dirname(save), exist_ok=True)

    model, cfg = load_bevfusion_model(args.config, args.ckpt)
    model = model.encoders.lidar.backbone.cuda().eval().half()
    model = funcs.layer_fusion_bn_relu(model)

    in_channels = int(cfg.model.encoders.lidar.backbone.in_channels)
    voxels = torch.zeros(1, in_channels, device="cuda").half()
    coors = torch.zeros(1, 4, device="cuda").int()
    exptool.export_onnx(model, voxels, coors, 1, args.inverse, save)


if __name__ == "__main__":
    main()
