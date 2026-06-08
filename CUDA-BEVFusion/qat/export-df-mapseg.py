import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.fp16_utils import auto_fp16

from df_export_utils import (
    add_project_root,
    camera_shapes,
    load_bevfusion_model,
)


class SubclassFuser(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @auto_fp16(apply_to=("features",))
    def forward(self, features):
        if self.parent.fuser is not None:
            x = self.parent.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.parent.decoder["backbone"](x)
        x = self.parent.decoder["neck"](x)
        return x[0]


class SubclassHeadMap(nn.Module):
    def __init__(self, parent, cfg, preserve_grid_sample=False):
        super().__init__()
        self.parent = parent
        self.preserve_grid_sample = preserve_grid_sample
        self.export_transform = ExportableBEVGridTransform(cfg.model.heads.map.grid_transform)

    @auto_fp16(apply_to=("x",))
    def forward(self, x):
        if "map" not in self.parent.heads:
            raise ValueError("The checkpoint/config does not contain heads.map")
        head = self.parent.heads["map"]
        if self.preserve_grid_sample:
            return head(x)
        x = self.export_transform(x)
        return torch.sigmoid(head.classifier(x))


class ExportableBEVGridTransform(nn.Module):
    """Approximate BEVGridTransform with TensorRT-friendly Resize + Slice."""

    def __init__(self, grid_transform_cfg):
        super().__init__()
        input_scope = grid_transform_cfg.input_scope
        output_scope = grid_transform_cfg.output_scope
        self.out_h = self._num_steps(output_scope[0])
        self.out_w = self._num_steps(output_scope[1])
        self.resize_h = max(int(round((input_scope[0][1] - input_scope[0][0]) / output_scope[0][2])), self.out_h)
        self.resize_w = max(int(round((input_scope[1][1] - input_scope[1][0]) / output_scope[1][2])), self.out_w)
        self.crop_y = max(int(round((output_scope[0][0] - input_scope[0][0]) / output_scope[0][2])), 0)
        self.crop_x = max(int(round((output_scope[1][0] - input_scope[1][0]) / output_scope[1][2])), 0)

    @staticmethod
    def _num_steps(scope):
        return int(round((float(scope[1]) - float(scope[0])) / float(scope[2])))

    def forward(self, x):
        x = F.interpolate(x, size=(self.resize_h, self.resize_w), mode="bilinear", align_corners=False)
        return x[:, :, self.crop_y : self.crop_y + self.out_h, self.crop_x : self.crop_x + self.out_w]


def parse_args():
    parser = argparse.ArgumentParser("Export bevfusion_df map segmentation fuser/head ONNX")
    parser.add_argument("--bevfusion-root", default=os.environ.get("BEVFUSION_DF_ROOT", r"H:\df_code\bevfusion"))
    parser.add_argument("--config", default=r"H:\df_code\bevfusion\configs\custom_dataset\seg.yaml")
    parser.add_argument("--ckpt", default=r"H:\df_code\bevfusion\runs\epoch_21.pth")
    parser.add_argument("--save-root", default="model/bevfusion_df_seg")
    parser.add_argument("--fuser-opset", type=int, default=13)
    parser.add_argument("--map-opset", type=int, default=13)
    parser.add_argument(
        "--preserve-grid-sample",
        action="store_true",
        help="Export the original grid_sample transform. Requires a PyTorch/ONNX version that supports it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    add_project_root(args.bevfusion_root)
    model, cfg = load_bevfusion_model(args.config, args.ckpt)
    shapes = camera_shapes(cfg)
    os.makedirs(args.save_root, exist_ok=True)

    fuser = SubclassFuser(model).cuda().eval()
    headmap = SubclassHeadMap(model, cfg, preserve_grid_sample=args.preserve_grid_sample).cuda().eval()

    camera_features = torch.randn(
        1,
        int(cfg.model.encoders.camera.vtransform.out_channels),
        shapes["bev_down_h"],
        shapes["bev_down_w"],
        device="cuda",
    )
    lidar_features = torch.randn(
        1,
        int(cfg.model.fuser.in_channels[1]),
        shapes["bev_down_h"],
        shapes["bev_down_w"],
        device="cuda",
    )

    with torch.no_grad():
        middle = fuser([camera_features, lidar_features])
        head_input = torch.randn_like(middle)
        print(f"Detected middle shape: {tuple(middle.shape)}")

        fuser_onnx = os.path.join(args.save_root, "fuser.onnx")
        torch.onnx.export(
            fuser,
            [camera_features, lidar_features],
            fuser_onnx,
            opset_version=args.fuser_opset,
            input_names=["camera", "lidar"],
            output_names=["middle"],
        )
        print(f"Exported {fuser_onnx}")

        map_onnx = os.path.join(args.save_root, "head.map.onnx")
        torch.onnx.export(
            headmap,
            head_input,
            map_onnx,
            opset_version=args.map_opset,
            input_names=["middle"],
            output_names=["map"],
        )
        print(f"Exported {map_onnx}")


if __name__ == "__main__":
    main()
