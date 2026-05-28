# RTMPose Compression

## 1. Get COCO 2017 keypoints (public)

```bash
mkdir -p data/coco && cd data/coco
curl -O http://images.cocodataset.org/zips/train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q train2017.zip && unzip -q val2017.zip && unzip -q annotations_trainval2017.zip
```

Result:
```
data/coco/train2017/*.jpg
data/coco/val2017/*.jpg
data/coco/annotations/person_keypoints_train2017.json
data/coco/annotations/person_keypoints_val2017.json
```

**Subset option (faster iteration on the M5 Max):** train on the first ~10k
person instances first to validate the loop, then scale to full train2017.
`export_softlabels.py --limit 10000` and `train.py --limit 10000` honor this.
