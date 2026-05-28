import json
import numpy as np
import cv2
from compression.eval.coco_ap import evaluate_predictions


def _mini(tmp_path):
    img = np.full((240, 320, 3), 127, np.uint8)
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "0001.jpg"), img)
    kps = []
    for i in range(17):
        kps += [60 + i * 5, 50 + i * 4, 2]
    ann = {"images": [{"id": 1, "file_name": "0001.jpg", "width": 320, "height": 240}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
                            "num_keypoints": 17, "keypoints": kps,
                            "bbox": [50.0, 40.0, 120.0, 160.0], "area": 19200.0}],
           "categories": [{"id": 1, "name": "person", "keypoints": ["k"] * 17,
                           "skeleton": []}]}
    p = tmp_path / "ann.json"
    p.write_text(json.dumps(ann))
    return p, kps


def test_predicting_ground_truth_scores_high_ap(tmp_path):
    ann_path, kps = _mini(tmp_path)
    arr = np.array(kps, dtype=np.float32).reshape(17, 3)
    preds = [{"image_id": 1, "category_id": 1,
              "keypoints": np.column_stack([arr[:, 0], arr[:, 1], np.ones(17)]).reshape(-1).tolist(),
              "score": 1.0}]
    ap = evaluate_predictions(str(ann_path), preds)
    assert ap["AP"] > 0.99
