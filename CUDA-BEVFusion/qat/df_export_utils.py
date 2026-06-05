import os
import sys

import torch


def add_project_root(root):
    root = os.path.abspath(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def default_checkpoint(root):
    candidates = [
        os.path.join(root, "runs", "detect", "epoch_2.pth"),
        os.path.join(root, "runs", "epoch_2.pth"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def load_config(config_path):
    from mmcv import Config
    from mmdet3d.utils import recursive_eval
    from torchpack.utils.config import configs

    configs.load(config_path, recursive=True)
    return Config(recursive_eval(configs), filename=config_path)


def load_bevfusion_model(config_path, checkpoint_path):
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model

    cfg = load_config(config_path)
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    if "encoders" in cfg.model and "camera" in cfg.model.encoders:
        backbone = cfg.model.encoders.camera.get("backbone", None)
        if backbone is not None and "init_cfg" in backbone:
            backbone.init_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.cuda().eval()
    return model, cfg


def checkpoint_object_num_classes(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    keys = [
        "heads.object.heatmap_head.1.weight",
        "module.heads.object.heatmap_head.1.weight",
    ]
    for key in keys:
        if key in state_dict:
            return int(state_dict[key].shape[0])
    suffix = "heads.object.heatmap_head.1.weight"
    for key, value in state_dict.items():
        if key.endswith(suffix):
            return int(value.shape[0])
    return None


def assert_checkpoint_matches_object_classes(cfg, checkpoint_path):
    expected = int(cfg.model.heads.object.num_classes)
    actual = checkpoint_object_num_classes(checkpoint_path)
    if actual is not None and actual != expected:
        raise RuntimeError(
            f"Checkpoint object head has {actual} classes, but config expects {expected}. "
            "Use the matching 21-class checkpoint/config pair before exporting head.bbox.onnx."
        )


def frustum_depth_bins(dbound):
    return int(round((float(dbound[1]) - float(dbound[0])) / float(dbound[2])))


def bound_size(bound):
    return int(round((float(bound[1]) - float(bound[0])) / float(bound[2])))


def camera_shapes(cfg):
    vtransform = cfg.model.encoders.camera.vtransform
    image_h, image_w = map(int, cfg.image_size)
    feat_h, feat_w = map(int, vtransform.feature_size)
    bev_h = bound_size(vtransform.ybound)
    bev_w = bound_size(vtransform.xbound)
    downsample = int(vtransform.get("downsample", 1))
    camera_channels = int(vtransform.out_channels)
    depth_bins = frustum_depth_bins(vtransform.dbound)
    return {
        "image_h": image_h,
        "image_w": image_w,
        "feat_h": feat_h,
        "feat_w": feat_w,
        "bev_h": bev_h,
        "bev_w": bev_w,
        "downsample": downsample,
        "camera_channels": camera_channels,
        "depth_bins": depth_bins,
        "bev_down_h": bev_h // downsample,
        "bev_down_w": bev_w // downsample,
    }
