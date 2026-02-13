from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List


def gt_filename(split: str) -> str:
    if split == "train":
        return "train_gt.json"
    if split == "test":
        return "test_gt.json"
    return f"{split}_gt.json"


def collect_usable_tracklets(image_root: Path, labels: Dict[str, int]) -> List[str]:
    usable: List[str] = []
    for tid in sorted(labels.keys()):
        tracklet_dir = image_root / tid
        if not tracklet_dir.exists() or not tracklet_dir.is_dir():
            continue
        num_frames = len([p for p in tracklet_dir.iterdir() if p.is_file()])
        if num_frames == 0:
            continue
        usable.append(tid)
    return usable


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny subset for sanity checks.")
    parser.add_argument("--data_root", required=True, help="Path to jersey-2023 root")
    parser.add_argument("--split", default="train", help="Split to sample from (default: train)")
    parser.add_argument("--num_clips", type=int, default=5, help="Number of tracklets to sample")
    parser.add_argument("--out_dir", required=True, help="Output directory (e.g. ./data/tiny5)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    source_root = Path(args.data_root)
    split_dir = source_root / args.split
    image_root = split_dir / "images"
    gt_path = split_dir / gt_filename(args.split)
    if not image_root.exists():
        raise FileNotFoundError(f"Missing image folder: {image_root}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground-truth file: {gt_path}")

    with gt_path.open("r", encoding="utf-8") as f:
        labels = {str(k): int(v) for k, v in json.load(f).items()}

    usable = collect_usable_tracklets(image_root, labels)
    if len(usable) < args.num_clips:
        raise ValueError(f"Requested {args.num_clips} clips but only found {len(usable)} usable tracklets.")

    rng = random.Random(args.seed)
    selected = rng.sample(usable, k=args.num_clips)

    out_root = Path(args.out_dir)
    out_split = out_root / "train"
    out_images = out_split / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    subset_gt: Dict[str, int] = {}
    for tid in selected:
        src_tracklet = image_root / tid
        dst_tracklet = out_images / tid
        if dst_tracklet.exists():
            shutil.rmtree(dst_tracklet)
        shutil.copytree(src_tracklet, dst_tracklet)
        subset_gt[tid] = labels[tid]

    out_gt = out_split / "train_gt.json"
    with out_gt.open("w", encoding="utf-8") as f:
        json.dump(subset_gt, f, indent=2)

    print(f"Created tiny subset at {out_root}")
    print(f"Selected tracklets: {len(selected)}")
    print(f"Ground-truth file: {out_gt}")


if __name__ == "__main__":
    main()
