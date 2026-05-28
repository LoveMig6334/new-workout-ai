"""COCO keypoints as a top-down (one-person-per-sample) dataset.

Each item crops a GT-bbox person to (INPUT_H, INPUT_W) via the shared affine
transform and returns the GT keypoints in input space. Used by both the teacher
exporter and the student trainer (collate to tensors in train.py).
"""
import os
import json
import cv2
import numpy as np
from .transforms import bbox_to_center_scale, get_warp_matrix, warp_keypoints, warp_image


class CocoTopDown:
    def __init__(self, image_dir: str, ann_file: str, min_keypoints: int = 1):
        self.image_dir = image_dir
        with open(ann_file) as f:
            data = json.load(f)
        self.images = {im["id"]: im for im in data["images"]}
        self.samples = [
            a for a in data["annotations"]
            if a.get("iscrowd", 0) == 0 and a.get("num_keypoints", 0) >= min_keypoints
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ann = self.samples[i]
        im = self.images[ann["image_id"]]
        path = os.path.join(self.image_dir, im["file_name"])
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)

        bbox = np.array(ann["bbox"], dtype=np.float32)
        center, scale = bbox_to_center_scale(bbox)
        inp = warp_image(image, center, scale)

        kps = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)  # (17, 3) x,y,v
        M = get_warp_matrix(center, scale, (inp.shape[2], inp.shape[1]))
        kps_in = warp_keypoints(kps[:, :2], M)
        vis = (kps[:, 2] > 0).astype(np.float32)
        # mark points warped outside the crop as not-visible
        inside = (kps_in[:, 0] >= 0) & (kps_in[:, 0] < inp.shape[2]) & \
                 (kps_in[:, 1] >= 0) & (kps_in[:, 1] < inp.shape[1])
        vis = vis * inside.astype(np.float32)

        return {
            "input": inp,
            "keypoints": kps_in.astype(np.float32),
            "vis": vis,
            "meta": {"image_id": ann["image_id"], "ann_id": ann["id"],
                     "center": center, "scale": scale, "bbox": bbox},
        }
