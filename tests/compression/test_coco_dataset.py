import json
import numpy as np
import cv2
from compression import config
from compression.coco_dataset import CocoTopDown


def _make_mini_coco(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # one 320x240 image with a single person annotation
    img = np.full((240, 320, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_dir / "0001.jpg"), img)
    kps = []
    for i in range(17):
        kps += [60 + i * 5, 50 + i * 4, 2]  # x, y, v=2 (visible)
    ann = {
        "images": [{"id": 1, "file_name": "0001.jpg", "width": 320, "height": 240}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
            "num_keypoints": 17, "keypoints": kps,
            "bbox": [50.0, 40.0, 120.0, 160.0], "area": 19200.0,
        }],
        "categories": [{"id": 1, "name": "person"}],
    }
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(json.dumps(ann))
    return img_dir, ann_path


def test_dataset_yields_aligned_input_and_keypoints(tmp_path):
    img_dir, ann_path = _make_mini_coco(tmp_path)
    ds = CocoTopDown(str(img_dir), str(ann_path))
    assert len(ds) == 1
    sample = ds[0]
    assert sample["input"].shape == (3, config.INPUT_H, config.INPUT_W)
    assert sample["keypoints"].shape == (17, 2)   # input space
    assert sample["vis"].shape == (17,)
    # all GT points were visible -> all vis == 1, and inside the input crop
    assert np.all(sample["vis"] == 1.0)
    assert np.all(sample["keypoints"][:, 0] >= 0) and np.all(sample["keypoints"][:, 0] <= config.INPUT_W)
    assert "meta" in sample and "image_id" in sample["meta"]
