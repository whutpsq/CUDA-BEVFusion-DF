import argparse
import os
import shutil
import sys

import torch

import tensor


def parse_args():
    parser = argparse.ArgumentParser("Dump bevfusion_df samples for CUDA-BEVFusion")
    parser.add_argument("--bevfusion-root", default=os.environ.get("BEVFUSION_DF_ROOT", r"H:\df_code\bevfusion"))
    parser.add_argument("--config", default=r"H:\df_code\bevfusion\configs\custom_dataset\default.yaml")
    parser.add_argument("--out", default="dump_df")
    parser.add_argument("--count", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    bevfusion_root = os.path.abspath(args.bevfusion_root)
    if bevfusion_root not in sys.path:
        sys.path.insert(0, bevfusion_root)

    from mmcv import Config
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.utils import recursive_eval
    from torchpack.utils.config import configs

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)

    os.makedirs(args.out, exist_ok=True)
    data_iter = iter(data_loader)
    for index in range(args.count):
        data = next(data_iter)
        root = os.path.join(args.out, f"{index:05d}")
        os.makedirs(root, exist_ok=True)

        torch.save(data, os.path.join(root, "example-data.pth"))
        metas = data["metas"].data[0]
        token = metas[0].get("token", str(index))
        with open(os.path.join(root, "token.txt"), "w", encoding="utf-8") as f:
            f.write(token)

        for cam_index, path in enumerate(metas[0].get("filename", [])):
            if os.path.exists(path):
                shutil.copyfile(path, os.path.join(root, f"{cam_index}-{os.path.basename(path)}"))

        tensor.save(data["img"].data[0].half(), os.path.join(root, "images.tensor"), True)
        tensor.save(data["points"].data[0][0].half(), os.path.join(root, "points.tensor"), True)
        tensor.save(data["camera_intrinsics"].data[0], os.path.join(root, "camera_intrinsics.tensor"), True)
        tensor.save(data["camera2ego"].data[0], os.path.join(root, "camera2ego.tensor"), True)
        tensor.save(data["lidar2ego"].data[0], os.path.join(root, "lidar2ego.tensor"), True)
        tensor.save(data["lidar2camera"].data[0], os.path.join(root, "lidar2camera.tensor"), True)
        tensor.save(data["camera2lidar"].data[0], os.path.join(root, "camera2lidar.tensor"), True)
        tensor.save(data["lidar2image"].data[0], os.path.join(root, "lidar2image.tensor"), True)
        tensor.save(data["img_aug_matrix"].data[0], os.path.join(root, "img_aug_matrix.tensor"), True)
        tensor.save(data["lidar_aug_matrix"].data[0], os.path.join(root, "lidar_aug_matrix.tensor"), True)
        print(f"Dumped {root}")


if __name__ == "__main__":
    main()
