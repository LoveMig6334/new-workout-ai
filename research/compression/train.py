"""Distillation training loop (MPS-native, no mmcv).

Per batch: student(input) -> (sx, sy). Loss = w_gt * (KLDiscret_x + KLDiscret_y)
                                            + w_kd * (KD_x + KD_y).
GT targets are built on the fly via simcc.encode_simcc; teacher targets come
from the precomputed fp16 store, matched by ann_id.

Usage:
  uv run python -m compression.train --split train --limit 10000 --epochs 20
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import config
from .coco_dataset import CocoTopDown
from .simcc import encode_simcc
from .softlabel_store import SoftLabelReader
from .models.student import StudentPose, save_student
from .losses import kl_discret_loss, kd_simcc_loss


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class TrainSet(Dataset):
    """Wraps CocoTopDown; attaches GT simcc + teacher simcc (by ann_id)."""
    def __init__(self, ds: CocoTopDown, reader: SoftLabelReader, limit: int = 0):
        ids = reader.ann_ids()
        self.items = [i for i in range(len(ds)) if ds.samples[i]["id"] in ids]
        if limit:
            self.items = self.items[:limit]
        self.ds = ds
        self.reader = reader

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        s = self.ds[self.items[k]]
        gx, gy = encode_simcc(s["keypoints"], s["vis"])
        tx, ty = self.reader.get(s["meta"]["ann_id"])
        return {
            "input": torch.from_numpy(s["input"]),
            "gx": torch.from_numpy(gx), "gy": torch.from_numpy(gy),
            "tx": torch.from_numpy(tx.astype(np.float32)),
            "ty": torch.from_numpy(ty.astype(np.float32)),
            "vis": torch.from_numpy(s["vis"]),
        }


def train(split="train", limit=0, epochs=20, batch=32, lr=1e-3, workers=4,
          w_gt=1.0, w_kd=1.0, width=config.STUDENT_WIDTH, depth=config.STUDENT_DEPTH,
          input_scale=1.0, tag=""):
    """Core (notebook-callable) distillation loop. Returns the final ckpt path."""
    dev = _device()
    print(f"device={dev}")
    img_dir = config.COCO_ROOT / f"{split}2017"
    ann = config.COCO_ROOT / "annotations" / f"person_keypoints_{split}2017.json"
    ds = CocoTopDown(str(img_dir), str(ann))
    reader = SoftLabelReader(str(config.SOFTLABEL_DIR / split))
    train_set = TrainSet(ds, reader, limit=limit)
    loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                        num_workers=workers, drop_last=True)

    model = StudentPose(width=width, depth=depth, input_scale=input_scale).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train()
        running = 0.0
        for b in tqdm(loader, desc=f"epoch {ep}"):
            inp = b["input"].to(dev)
            sx, sy = model(inp)
            vis = b["vis"].to(dev)
            l_gt = kl_discret_loss(sx, b["gx"].to(dev), vis) + \
                   kl_discret_loss(sy, b["gy"].to(dev), vis)
            l_kd = kd_simcc_loss(sx, b["tx"].to(dev)) + \
                   kd_simcc_loss(sy, b["ty"].to(dev))
            loss = w_gt * l_gt + w_kd * l_kd
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        print(f"epoch {ep}: mean loss {running / max(1, len(loader)):.4f}")
        save_student(model, config.CHECKPOINT_DIR / f"student{tag}_ep{ep}.pt")
    final = config.CHECKPOINT_DIR / f"student{tag}_final.pt"
    save_student(model, final)
    return str(final)


def main():  # thin CLI wrapper around train()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--w_gt", type=float, default=1.0)
    ap.add_argument("--w_kd", type=float, default=1.0)
    ap.add_argument("--width", type=float, default=config.STUDENT_WIDTH)
    ap.add_argument("--depth", type=float, default=config.STUDENT_DEPTH)
    ap.add_argument("--input_scale", type=float, default=1.0)
    ap.add_argument("--tag", default="")  # checkpoint suffix per ablation variant
    a = ap.parse_args()
    train(split=a.split, limit=a.limit, epochs=a.epochs, batch=a.batch, lr=a.lr,
          workers=a.workers, w_gt=a.w_gt, w_kd=a.w_kd, width=a.width, depth=a.depth,
          input_scale=a.input_scale, tag=a.tag)


if __name__ == "__main__":
    main()
