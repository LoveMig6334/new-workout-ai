"""COCO keypoint AP via pycocotools, using GT bounding boxes.

evaluate_predictions takes the GT ann file + a list of COCO-format keypoint
predictions and returns the standard AP/AR summary dict.
"""
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_predictions(ann_file: str, predictions: list) -> dict:
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, iouType="keypoints")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats  # [AP, AP50, AP75, AP(M), AP(L), AR, AR50, AR75, AR(M), AR(L)]
    return {"AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
            "AR": float(s[5])}


import argparse
import numpy as np
import torch
from .. import config
from ..coco_dataset import CocoTopDown
from ..models.student import load_student
from ..simcc import decode_simcc
from ..transforms import get_warp_matrix, warp_keypoints


def run_val_ap(ckpt: str, limit: int = 0) -> dict:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_student(ckpt, dev)
    img_dir = config.COCO_ROOT / "val2017"
    ann = config.COCO_ROOT / "annotations" / "person_keypoints_val2017.json"
    ds = CocoTopDown(str(img_dir), str(ann))
    n = len(ds) if limit == 0 else min(limit, len(ds))
    preds = []
    with torch.no_grad():
        for i in range(n):
            s = ds[i]
            inp = torch.from_numpy(s["input"])[None].to(dev)
            sx, sy = model(inp)
            # softmax logits so decode_simcc's peak-value confidence is a real
            # probability (argmax position is unaffected either way).
            sx = torch.softmax(sx, dim=-1).cpu().numpy()
            sy = torch.softmax(sy, dim=-1).cpu().numpy()
            kps_in, scores = decode_simcc(sx, sy)
            # input space -> image space
            Minv = get_warp_matrix(s["meta"]["center"], s["meta"]["scale"],
                                   (config.INPUT_W, config.INPUT_H), inverse=True)
            kps_img = warp_keypoints(kps_in[0], Minv)
            flat = np.column_stack([kps_img[:, 0], kps_img[:, 1], scores[0]]).reshape(-1)
            preds.append({"image_id": int(s["meta"]["image_id"]), "category_id": 1,
                          "keypoints": flat.tolist(), "score": float(scores[0].mean())})
    return evaluate_predictions(str(ann), preds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.CHECKPOINT_DIR / "student_final.pt"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    print(run_val_ap(a.ckpt, a.limit))
