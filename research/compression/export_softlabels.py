"""Cache teacher SimCC soft labels over a COCO split.

Asserts the teacher's real output bin counts match config (catches a drifted
input size before training silently learns the wrong target).

Usage:
  uv run python -m compression.export_softlabels --split train --limit 10000
"""
import argparse
from tqdm import tqdm
from . import config
from .coco_dataset import CocoTopDown
from .teacher import Teacher
from .softlabel_store import SoftLabelWriter


def export_softlabels(split: str = "train", limit: int = 0) -> str:
    """Core (notebook-callable): cache teacher soft labels for a COCO split.
    Returns the output prefix path."""
    img_dir = config.COCO_ROOT / f"{split}2017"
    ann = config.COCO_ROOT / "annotations" / f"person_keypoints_{split}2017.json"
    ds = CocoTopDown(str(img_dir), str(ann))
    teacher = Teacher()

    n = len(ds) if limit == 0 else min(limit, len(ds))
    out_prefix = config.SOFTLABEL_DIR / f"{split}"
    writer = SoftLabelWriter(str(out_prefix), config.SIMCC_X_BINS,
                             config.SIMCC_Y_BINS, config.NUM_KEYPOINTS)
    for i in tqdm(range(n), desc=f"teacher {split}"):
        s = ds[i]
        sx, sy = teacher.infer_simcc(s["input"])
        if i == 0:
            assert sx.shape == (config.NUM_KEYPOINTS, config.SIMCC_X_BINS), \
                f"teacher x bins {sx.shape} != config {config.SIMCC_X_BINS}"
            assert sy.shape == (config.NUM_KEYPOINTS, config.SIMCC_Y_BINS), \
                f"teacher y bins {sy.shape} != config {config.SIMCC_Y_BINS}"
        writer.add(ann_id=s["meta"]["ann_id"], sx=sx, sy=sy)
    writer.close()
    print(f"wrote {n} soft labels to {out_prefix}.npy")
    return str(out_prefix)


def main():  # thin CLI wrapper around export_softlabels()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    export_softlabels(split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()
