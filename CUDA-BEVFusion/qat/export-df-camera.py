import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import onnx
import torch
from onnxsim import simplify
from torch import nn

from df_export_utils import add_project_root, camera_shapes, default_checkpoint, load_bevfusion_model


class SubclassCameraModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img, depth):
        b, n, c, h, w = img.size()
        img = img.view(b * n, c, h, w)

        feat = self.model.encoders.camera.backbone(img)
        feat = self.model.encoders.camera.neck(feat)
        if not isinstance(feat, torch.Tensor):
            feat = feat[0]

        bn, c, h, w = map(int, feat.size())
        feat = feat.view(b, int(bn / b), c, h, w)

        vtransform = self.model.encoders.camera.vtransform
        b, n, c, fh, fw = map(int, feat.shape)
        depth = depth.view(b * n, *depth.shape[2:])
        feat = feat.view(b * n, c, fh, fw)

        depth_feat = vtransform.dtransform(depth)
        x = torch.cat([depth_feat, feat], dim=1)
        x = vtransform.depthnet(x)

        depth_weights = x[:, : vtransform.D].softmax(dim=1)
        camera_feature = x[:, vtransform.D : (vtransform.D + vtransform.C)].permute(0, 2, 3, 1)
        return camera_feature, depth_weights


def parse_args():
    parser = argparse.ArgumentParser("Export bevfusion_df camera ONNX")
    parser.add_argument("--bevfusion-root", default=os.environ.get("BEVFUSION_DF_ROOT", r"H:\df_code\bevfusion"))
    parser.add_argument("--config", default=r"H:\df_code\bevfusion\configs\custom_dataset\default.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--save-root", default="model/bevfusion_df")
    return parser.parse_args()


def main():
    args = parse_args()
    add_project_root(args.bevfusion_root)
    if args.ckpt is None:
        args.ckpt = default_checkpoint(args.bevfusion_root)
    model, cfg = load_bevfusion_model(args.config, args.ckpt)
    shapes = camera_shapes(cfg)

    os.makedirs(args.save_root, exist_ok=True)
    camera_model = SubclassCameraModule(model).cuda().eval()
    downsample_model = model.encoders.camera.vtransform.downsample.cuda().eval()

    img = torch.zeros(1, 6, 3, shapes["image_h"], shapes["image_w"], device="cuda")
    depth = torch.zeros(1, 6, 1, shapes["image_h"], shapes["image_w"], device="cuda")
    downsample_in = torch.zeros(
        1, shapes["camera_channels"], shapes["bev_h"], shapes["bev_w"], device="cuda"
    )

    with torch.no_grad():
        camera_backbone_onnx = os.path.join(args.save_root, "camera.backbone.onnx")
        torch.onnx.export(
            camera_model,
            (img, depth),
            camera_backbone_onnx,
            input_names=["img", "depth"],
            output_names=["camera_feature", "camera_depth_weights"],
            opset_version=13,
            do_constant_folding=True,
        )
        try:
            onnx_model = onnx.load(camera_backbone_onnx)
            onnx_model, check = simplify(onnx_model)
            assert check, "Simplified camera.backbone ONNX could not be validated"
            onnx.save(onnx_model, camera_backbone_onnx)
        except Exception as error:
            print(f"WARNING: onnxsim failed for {camera_backbone_onnx}; keeping original ONNX. Error: {error}")
        print(f"Exported {camera_backbone_onnx}")

        camera_vtransform_onnx = os.path.join(args.save_root, "camera.vtransform.onnx")
        torch.onnx.export(
            downsample_model,
            downsample_in,
            camera_vtransform_onnx,
            input_names=["feat_in"],
            output_names=["feat_out"],
            opset_version=13,
            do_constant_folding=True,
        )
        print(f"Exported {camera_vtransform_onnx}")


if __name__ == "__main__":
    main()
