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
    assert_checkpoint_matches_object_classes,
    camera_shapes,
    default_checkpoint,
    load_bevfusion_model,
    load_config,
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


class SubclassHeadBBox(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.classes_eye = nn.Parameter(torch.eye(parent.heads.object.num_classes).float())

    @staticmethod
    @auto_fp16(apply_to=("inputs", "classes_eye"))
    def head_forward(self, inputs, classes_eye):
        batch_size = int(inputs.shape[0])
        lidar_feat = self.shared_conv(inputs)
        lidar_feat_flatten = lidar_feat.view(batch_size, int(lidar_feat.shape[1]), -1)
        bev_pos = self.bev_pos.to(lidar_feat.dtype).repeat(batch_size, 1, 1).to(lidar_feat.device)

        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner

        if self.test_cfg["dataset"] == "nuScenes" and self.num_classes >= 10:
            local_max[:, 8] = heatmap[:, 8]
            local_max[:, 9] = heatmap[:, 9]
        elif self.test_cfg["dataset"] == "Waymo":
            local_max[:, 1] = heatmap[:, 1]
            local_max[:, 2] = heatmap[:, 2]

        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, int(heatmap.shape[1]), -1)
        top_proposals = heatmap.view(batch_size, -1).topk(k=self.num_proposals, dim=-1, largest=True)[1]
        top_proposals_class = top_proposals // int(heatmap.shape[-1])
        top_proposals_index = top_proposals % int(heatmap.shape[-1])
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        self.one_hot = classes_eye.index_select(0, top_proposals_class.view(-1))[None].permute(0, 2, 1)
        query_feat += self.class_encoding(self.one_hot)
        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        ret_dicts = []
        for i in range(self.num_decoder_layers):
            query_feat = self.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)
            res_layer = self.prediction_heads[i](query_feat)
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
            ret_dicts.append(res_layer)
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

        ret_dicts[0]["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )
        ret_dicts[0]["dense_heatmap"] = dense_heatmap

        if self.auxiliary is False:
            return ret_dicts[-1]

        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return new_res

    def get_bboxes(self, preds_dict, one_hot):
        batch_score = preds_dict["heatmap"].sigmoid()
        batch_score = batch_score * preds_dict["query_heatmap_score"] * one_hot
        batch_center = preds_dict["center"]
        batch_height = preds_dict["height"]
        batch_dim = preds_dict["dim"]
        batch_rot = preds_dict["rot"]
        batch_vel = preds_dict["vel"] if "vel" in preds_dict else None
        return [batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel]

    def forward(self, x):
        for head_type, head in self.parent.heads.items():
            if head_type == "object":
                pred_dict = self.head_forward(head, x, self.classes_eye)
                return self.get_bboxes(pred_dict, head.one_hot)
        raise ValueError("bevfusion_df export requires an object head")


class CustomLayerNormImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps, x_shape):
        return F.layer_norm(input, normalized_shape, weight, bias, eps)

    @staticmethod
    def symbolic(g, input, normalized_shape, weight, bias, eps, x_shape):
        y = g.op("nv::CustomLayerNormalization", input, weight, bias, axis_i=-1, epsilon_f=eps)
        y.setType(input.type().with_sizes(x_shape))
        return y


class CustomLayerNorm(nn.LayerNorm):
    def forward(self, input):
        return CustomLayerNormImpl.apply(input, self.normalized_shape, self.weight, self.bias, self.eps, input.size())

    @staticmethod
    def convert(old):
        new = CustomLayerNorm(old.normalized_shape, old.eps, old.elementwise_affine)
        if new.weight is not None:
            new.weight.data = old.weight.data
            new.bias.data = old.bias.data
        return new


def replace_layernorm(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            parent, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), child, CustomLayerNorm.convert(module))


def parse_args():
    parser = argparse.ArgumentParser("Export bevfusion_df fuser/head ONNX")
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
    cfg = load_config(args.config)
    assert_checkpoint_matches_object_classes(cfg, args.ckpt)
    model, cfg = load_bevfusion_model(args.config, args.ckpt)
    shapes = camera_shapes(cfg)
    os.makedirs(args.save_root, exist_ok=True)

    fuser = SubclassFuser(model).cuda().eval()
    headbbox = SubclassHeadBBox(model).cuda().eval()
    replace_layernorm(headbbox)

    camera_features = torch.randn(
        1, int(cfg.model.encoders.camera.vtransform.out_channels), shapes["bev_down_h"], shapes["bev_down_w"], device="cuda"
    )
    lidar_features = torch.randn(1, int(cfg.model.fuser.in_channels[1]), shapes["bev_down_h"], shapes["bev_down_w"], device="cuda")
    head_input = torch.randn(1, int(cfg.model.heads.object.in_channels), shapes["bev_down_h"], shapes["bev_down_w"], device="cuda")

    with torch.no_grad():
        fuser_onnx = os.path.join(args.save_root, "fuser.onnx")
        torch.onnx.export(
            fuser,
            [camera_features, lidar_features],
            fuser_onnx,
            opset_version=13,
            input_names=["camera", "lidar"],
            output_names=["middle"],
        )
        print(f"Exported {fuser_onnx}")

        head_onnx = os.path.join(args.save_root, "head.bbox.onnx")
        torch.onnx.export(
            headbbox,
            head_input,
            head_onnx,
            opset_version=13,
            input_names=["middle"],
            output_names=["score", "rot", "dim", "reg", "height", "vel"],
        )
        print(f"Exported {head_onnx}")


if __name__ == "__main__":
    main()
