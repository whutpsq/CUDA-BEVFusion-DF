#!/usr/bin/env python3
"""Build an intentionally leaked train/val split for pipeline overfit checks.

This is only for proving that a tiny demo dataset can be memorized.  It must not
be used to report validation performance.
"""

import argparse
import pickle
from pathlib import Path


def load_infos(path: Path):
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "infos" not in payload:
        raise ValueError(f"unexpected info format: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="data_yangluo/yangluo_bevfusion",
        help="Directory containing the normal train/val info PKLs.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_yangluo/yangluo_demo_overfit",
        help="Directory receiving the intentionally leaked info PKLs.",
    )
    parser.add_argument("--info-prefix", default="yangluo")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    train = load_infos(input_dir / f"{args.info_prefix}_infos_train.pkl")
    val = load_infos(input_dir / f"{args.info_prefix}_infos_val.pkl")

    by_token = {}
    for info in train["infos"] + val["infos"]:
        token = info.get("token")
        if token is None:
            raise ValueError("every info must contain a token")
        by_token[token] = info
    infos = sorted(by_token.values(), key=lambda x: (x.get("timestamp", 0), x["token"]))
    if not infos:
        raise ValueError("no samples found")

    metadata = dict(train.get("metadata", {}))
    metadata.update(
        demo_overfit=True,
        warning="train and val intentionally contain the same samples",
        source_train_count=len(train["infos"]),
        source_val_count=len(val["infos"]),
    )
    payload = {"infos": infos, "metadata": metadata}

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        path = output_dir / f"{args.info_prefix}_infos_{split}.pkl"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream)
        temporary.replace(path)
        print(f"wrote {path} samples={len(infos)}")

    print("PASS demo-overfit split: train and val intentionally share all samples")


if __name__ == "__main__":
    main()

