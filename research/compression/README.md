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

## Task-specific eval clip

1. Record a ~30 s desk/neck-stretch clip with the webcam.
2. Sample ~30–50 frames; hand-label the 7 upper-body keypoints
   (nose, ears, shoulders, hips) — e.g. with `labelme` or a tiny click tool.
3. Save as a COCO-format json next to the frames.
4. Compare student vs teacher (and vs your labels) via
   `eval/task_eval.py`: report `upper_body_pck` and `angle_agreement`
   (the head-lateral-tilt / CVA differences the scoring actually consumes).

## Quantization accuracy check

After PTQ, re-run COCO val AP through the CoreML int8 model (load via
`coremltools` and feed the same warped crops as `eval/coco_ap.run_val_ap`,
swapping the torch forward for `mlmodel.predict`). Report AP_int8 vs AP_fp.
A small drop (≤ ~1–2 AP) is the success criterion; if larger, fall back to QAT.
