# RTMPose Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a smaller, faster student pose model by distilling RTMPose-m, specialize it for the desk/upper-body case, quantize it for the ANE, and benchmark it (COCO AP + task-specific angle agreement) against RTMPose tiers / MoveNet / MediaPipe — all trained locally on Apple Silicon (MPS), no `mmpose`/`mmcv`.

**Architecture:** A self-contained research package `research/compression/`. A shared SimCC codec + top-down affine transform are used by both the teacher soft-label exporter (runs the existing RTMPose-m ONNX over COCO, caches `simcc_x`/`simcc_y` as fp16) and the student trainer (pure-PyTorch CSPNeXt-lite backbone + SimCC head, trained on MPS via KL distillation + GT loss). Evaluation computes COCO AP (GT bbox), task-specific keypoint error + downstream angle agreement on a self-recorded clip, and latency/size on MPS + CoreML. The live app is untouched until an optional final integration task.

**Tech Stack:** Python 3.12, `uv`, PyTorch (MPS backend), `rtmlib`/`onnxruntime` (teacher), `pycocotools` (AP), `coremltools` (quantization/export), `numpy`, `opencv-python`, `tqdm`, pytest. The repo's `analysis/angles.py` is reused for the downstream-agreement metric.

**Spec:** `docs/superpowers/specs/2026-05-27-rtmpose-compression-design.md`

---

## Conventions used throughout this plan

- Always run Python via `uv run`.
- New package lives under `research/compression/` (importable as `compression.*` once `research/` is on the pytest path — Task 0).
- Tests live under `tests/compression/` and import as `from compression.<module> import ...`.
- Keypoints are COCO-17. Index constants are imported from `analysis.angles` (`NOSE`, `L_SHOULDER`, …) wherever the upper-body subset is needed.
- "input space" = coordinates in the model's fixed input crop (W=192, H=256 default). "image space" = original image pixels.
- Commit after every task with the message shown in its final step.

---

## File structure

```
research/
  compression/
    __init__.py
    config.py            # paths, input size, simcc bins, student/training hyperparams
    simcc.py             # SimCC encode (kps->target) / decode (simcc->kps,scores)  [PURE]
    transforms.py        # bbox->center/scale, top-down affine warp + inverse       [PURE]
    coco_dataset.py      # COCO top-down crop dataset (GT bbox) -> (input, kps, vis)
    teacher.py           # run RTMPose-m over crops, capture simcc
    export_softlabels.py # CLI: cache teacher simcc over COCO as fp16 shards
    softlabel_store.py   # write/read the fp16 soft-label cache                      [PURE I/O]
    models/
      __init__.py
      backbone.py        # CSPNeXt-lite backbone (pure PyTorch)
      head.py            # SimCC head
      student.py         # backbone + head, configurable width/input
    losses.py            # KLDiscretLoss (GT) + KDSimCCLoss (teacher)                [PURE]
    train.py             # CLI: MPS training loop
    eval/
      __init__.py
      coco_ap.py         # pycocotools AP with GT bbox
      task_eval.py       # keypoint error + downstream angle agreement
      benchmark.py       # size + latency on MPS / CoreML
    quantize/
      __init__.py
      ptq_coreml.py      # export student -> ONNX -> CoreML int8 (calibration)
    compare/
      __init__.py
      frontier.py        # gather all results -> accuracy/size/latency Pareto plots
    README.md            # how to run the full pipeline end-to-end
tests/
  compression/
    __init__.py
    conftest.py          # ensures research/ importable + tiny fixtures
    fixtures/
      mini_coco/         # hand-built 2-image COCO subset for dataset/AP unit tests
    test_simcc.py
    test_transforms.py
    test_coco_dataset.py
    test_softlabel_store.py
    test_losses.py
    test_student.py
    test_coco_ap.py
    test_task_eval.py
```

**Phase independence note:** Phases 0–9 are a strict pipeline (each feeds the next). Phase 10 (baselines/frontier) and Phase 11 (app integration) are independent add-ons that can be done last or skipped without breaking the core deliverable.

---

## Phase 0 — Scaffolding & dependencies

### Task 0: Create the package, wire the test path, add dependencies

**Files:**
- Create: `research/compression/__init__.py` (puts `src/` on sys.path — see Step 4)
- Create: `research/compression/config.py`
- Create: `tests/compression/__init__.py` (empty)
- Create: `tests/compression/conftest.py`
- Modify: `pyproject.toml:33-35` (pytest `pythonpath`)

- [ ] **Step 1: Add new dependencies**

Run:
```bash
uv add pycocotools coremltools
```
Expected: `pyproject.toml` gains `pycocotools` and `coremltools` under `[project] dependencies`; `uv.lock` updates; no error.

- [ ] **Step 2: Put `research/` on the pytest import path**

Edit `pyproject.toml`, change:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```
to:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "research"]
```

- [ ] **Step 3: Create the config module**

Create `research/compression/config.py`:
```python
"""Central config for the RTMPose compression project.

INPUT_H/W and the SIMCC bin counts are the RTMPose-m defaults; they are
re-verified against the teacher's real ONNX output shapes in
export_softlabels.py (which asserts on mismatch) so they can never drift
silently.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# Top-down model input (H, W) and SimCC split ratio (RTMPose-m defaults).
INPUT_H = 256
INPUT_W = 192
SIMCC_SPLIT = 2.0
SIMCC_X_BINS = int(INPUT_W * SIMCC_SPLIT)  # 384
SIMCC_Y_BINS = int(INPUT_H * SIMCC_SPLIT)  # 512
NUM_KEYPOINTS = 17

# Top-down crop padding around the bbox (matches rtmlib's RTMPose default).
BBOX_PADDING = 1.25

# ImageNet-style normalization used by RTMPose.
PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)

# Where artifacts land (all gitignored under models/ or a sibling).
SOFTLABEL_DIR = MODELS_DIR / "compression" / "softlabels"
CHECKPOINT_DIR = MODELS_DIR / "compression" / "checkpoints"
COCO_ROOT = PROJECT_ROOT / "data" / "coco"  # images + annotations live here

# Student defaults (overridable on the train.py CLI).
STUDENT_WIDTH = 0.33   # backbone channel multiplier vs the reference
STUDENT_DEPTH = 0.33   # backbone block-count multiplier
```

- [ ] **Step 4: Create the test conftest with a sanity test**

Create `tests/compression/conftest.py`:
```python
import sys
from pathlib import Path

# pytest's pythonpath already adds research/, but `python -m pytest` from some
# CWDs may not — mirror the src/ belt-and-suspenders pattern in tests/conftest.py.
RESEARCH = Path(__file__).resolve().parents[2] / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))
```

Create `tests/compression/__init__.py` (empty).

Create `research/compression/__init__.py` so the CLIs (`python -m compression.*`) can import the project's `src/` modules (`pose2d`, `analysis.angles`) — `pythonpath=["src"]` only applies under pytest, not to `python -m`:
```python
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
```

- [ ] **Step 5: Verify the package imports under pytest**

Add a temporary check by running:
```bash
uv run python -c "from compression import config; print(config.SIMCC_X_BINS, config.SIMCC_Y_BINS)"
```
Expected: `384 512`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock research/compression/ tests/compression/
git commit -m "chore: scaffold research/compression package + deps"
```

---

## Phase 1 — Shared SimCC codec (pure, TDD)

### Task 1: SimCC encode/decode

**Files:**
- Create: `research/compression/simcc.py`
- Test: `tests/compression/test_simcc.py`

- [ ] **Step 1: Write the failing test**

Create `tests/compression/test_simcc.py`:
```python
import numpy as np
from compression import config
from compression.simcc import encode_simcc, decode_simcc


def test_encode_shapes_and_peak_at_keypoint():
    # one keypoint at input-space (x=96, y=128) -> peak bin = coord * split
    kps = np.array([[96.0, 128.0]], dtype=np.float32)  # (K=1, 2)
    vis = np.array([1.0], dtype=np.float32)
    sx, sy = encode_simcc(kps, vis)
    assert sx.shape == (1, config.SIMCC_X_BINS)
    assert sy.shape == (1, config.SIMCC_Y_BINS)
    assert int(sx[0].argmax()) == round(96.0 * config.SIMCC_SPLIT)
    assert int(sy[0].argmax()) == round(128.0 * config.SIMCC_SPLIT)


def test_invisible_keypoint_is_zero_target():
    kps = np.array([[96.0, 128.0]], dtype=np.float32)
    vis = np.array([0.0], dtype=np.float32)
    sx, sy = encode_simcc(kps, vis)
    assert np.allclose(sx, 0.0) and np.allclose(sy, 0.0)


def test_decode_round_trips_encode():
    kps = np.array([[40.0, 200.0], [150.0, 50.0]], dtype=np.float32)
    vis = np.array([1.0, 1.0], dtype=np.float32)
    sx, sy = encode_simcc(kps, vis)
    out_kps, scores = decode_simcc(sx[None], sy[None])  # add batch dim
    assert out_kps.shape == (1, 2, 2)
    # within half a bin (1 / split) of the input
    assert np.allclose(out_kps[0], kps, atol=1.0)
    assert np.all(scores[0] > 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/compression/test_simcc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compression.simcc'`

- [ ] **Step 3: Implement `simcc.py`**

Create `research/compression/simcc.py`:
```python
"""SimCC label codec: keypoints (input-space) <-> 1D Gaussian distributions.

Mirrors RTMPose's SimCC representation so teacher and student share one codec.
encode_simcc produces Gaussian-smoothed soft labels for the GT loss; decode_simcc
turns either GT targets or model logits into keypoints + a peak-value confidence.
"""
import numpy as np
from . import config

_SIGMA = 6.0  # in bin units; RTMPose-style smoothing


def _gaussian_1d(num_bins: int, mu: np.ndarray, sigma: float) -> np.ndarray:
    """(K, num_bins) Gaussian centered at mu (in bins) for each of K keypoints."""
    x = np.arange(num_bins, dtype=np.float32)[None, :]  # (1, bins)
    mu = mu[:, None]                                     # (K, 1)
    g = np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    return g.astype(np.float32)


def encode_simcc(kps_input: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """kps_input: (K, 2) in input space. vis: (K,) 0/1. Returns (sx (K,Xbins), sy (K,Ybins))."""
    mu_x = kps_input[:, 0] * config.SIMCC_SPLIT
    mu_y = kps_input[:, 1] * config.SIMCC_SPLIT
    sx = _gaussian_1d(config.SIMCC_X_BINS, mu_x, _SIGMA)
    sy = _gaussian_1d(config.SIMCC_Y_BINS, mu_y, _SIGMA)
    mask = (vis > 0).astype(np.float32)[:, None]
    return sx * mask, sy * mask


def decode_simcc(sx: np.ndarray, sy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sx: (B, K, Xbins), sy: (B, K, Ybins). Returns (kps (B,K,2) input-space, scores (B,K))."""
    x_idx = sx.argmax(axis=-1).astype(np.float32)  # (B, K)
    y_idx = sy.argmax(axis=-1).astype(np.float32)
    kps = np.stack([x_idx / config.SIMCC_SPLIT, y_idx / config.SIMCC_SPLIT], axis=-1)
    # confidence = geometric-ish mean of the two peak values (after softmax-free norm)
    sx_n = sx / (sx.sum(axis=-1, keepdims=True) + 1e-9)
    sy_n = sy / (sy.sum(axis=-1, keepdims=True) + 1e-9)
    scores = np.sqrt(sx_n.max(axis=-1) * sy_n.max(axis=-1))
    return kps.astype(np.float32), scores.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/compression/test_simcc.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add research/compression/simcc.py tests/compression/test_simcc.py
git commit -m "feat(compression): shared SimCC encode/decode codec"
```

---

## Phase 2 — Top-down affine transform (pure, TDD)

### Task 2: bbox → crop transform + inverse

**Files:**
- Create: `research/compression/transforms.py`
- Test: `tests/compression/test_transforms.py`

- [ ] **Step 1: Write the failing test**

Create `tests/compression/test_transforms.py`:
```python
import numpy as np
from compression import config
from compression.transforms import bbox_to_center_scale, get_warp_matrix, warp_keypoints


def test_center_scale_centers_the_bbox():
    # bbox xywh
    c, s = bbox_to_center_scale(np.array([100.0, 50.0, 40.0, 80.0]))
    assert np.allclose(c, [120.0, 90.0])  # center of the box
    assert s[0] > 0 and s[1] > 0


def test_warp_maps_bbox_center_to_input_center():
    c, s = bbox_to_center_scale(np.array([100.0, 50.0, 40.0, 80.0]))
    M = get_warp_matrix(c, s, (config.INPUT_W, config.INPUT_H))
    center_in = warp_keypoints(c[None], M)[0]
    assert np.allclose(center_in, [config.INPUT_W / 2, config.INPUT_H / 2], atol=1.0)


def test_inverse_warp_round_trips():
    c, s = bbox_to_center_scale(np.array([10.0, 20.0, 200.0, 100.0]))
    M = get_warp_matrix(c, s, (config.INPUT_W, config.INPUT_H))
    Minv = get_warp_matrix(c, s, (config.INPUT_W, config.INPUT_H), inverse=True)
    pts = np.array([[15.0, 25.0], [120.0, 80.0]], dtype=np.float32)
    pts_in = warp_keypoints(pts, M)
    pts_back = warp_keypoints(pts_in, Minv)
    assert np.allclose(pts_back, pts, atol=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/compression/test_transforms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compression.transforms'`

- [ ] **Step 3: Implement `transforms.py`**

Create `research/compression/transforms.py`:
```python
"""Top-down affine preprocessing matching rtmlib's RTMPose convention:
bbox -> center/scale (with BBOX_PADDING, aspect fixed to the input ratio) ->
affine warp to (INPUT_W, INPUT_H). Used identically by teacher export and the
student dataset so their SimCC outputs are aligned.
"""
import cv2
import numpy as np
from . import config


def bbox_to_center_scale(bbox_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = bbox_xywh
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
    aspect = config.INPUT_W / config.INPUT_H
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    scale = np.array([w, h], dtype=np.float32) * config.BBOX_PADDING
    return center, scale


def get_warp_matrix(center, scale, output_size, inverse: bool = False) -> np.ndarray:
    """2x3 affine. output_size = (W, H). Maps image space -> input space (or inverse)."""
    out_w, out_h = output_size
    src_w, src_h = float(scale[0]), float(scale[1])
    src = np.array([
        center,
        center + np.array([0.0, -src_h * 0.5], dtype=np.float32),
        center + np.array([-src_w * 0.5, 0.0], dtype=np.float32),
    ], dtype=np.float32)
    dst = np.array([
        [out_w * 0.5, out_h * 0.5],
        [out_w * 0.5, out_h * 0.5 - out_h * 0.5],
        [out_w * 0.5 - out_w * 0.5, out_h * 0.5],
    ], dtype=np.float32)
    if inverse:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def warp_keypoints(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine M to (N, 2) points."""
    pts = np.asarray(pts, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([pts, ones], axis=1)  # (N, 3)
    return (homo @ M.T).astype(np.float32)


def warp_image(image_bgr: np.ndarray, center, scale) -> np.ndarray:
    """Warp + normalize -> (3, H, W) float32 tensor (CHW, RGB, normalized)."""
    M = get_warp_matrix(center, scale, (config.INPUT_W, config.INPUT_H))
    crop = cv2.warpAffine(image_bgr, M, (config.INPUT_W, config.INPUT_H), flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - np.array(config.PIXEL_MEAN)) / np.array(config.PIXEL_STD)
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/compression/test_transforms.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add research/compression/transforms.py tests/compression/test_transforms.py
git commit -m "feat(compression): top-down affine crop transform + inverse"
```

---

## Phase 3 — COCO dataset

### Task 3: COCO top-down keypoint dataset

**Files:**
- Create: `tests/compression/fixtures/mini_coco/` (a 2-image hand-built COCO subset)
- Create: `research/compression/coco_dataset.py`
- Test: `tests/compression/test_coco_dataset.py`

- [ ] **Step 1: Build the mini-COCO fixture generator + the failing test**

Create `tests/compression/test_coco_dataset.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/compression/test_coco_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compression.coco_dataset'`

- [ ] **Step 3: Implement `coco_dataset.py`**

Create `research/compression/coco_dataset.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/compression/test_coco_dataset.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add research/compression/coco_dataset.py tests/compression/test_coco_dataset.py
git commit -m "feat(compression): COCO top-down keypoint dataset"
```

### Task 3b: Document COCO acquisition (no code — data prep)

**Files:**
- Create: `research/compression/README.md` (start it here; later tasks append)

- [ ] **Step 1: Write the data-acquisition section of the README**

Create `research/compression/README.md`:
```markdown
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
```

Add `data/` to `.gitignore` if not already ignored:
```bash
grep -qxF 'data/' .gitignore || echo 'data/' >> .gitignore
```

- [ ] **Step 2: Commit**

```bash
git add research/compression/README.md .gitignore
git commit -m "docs(compression): COCO acquisition instructions"
```

---

## Phase 4 — Teacher soft-label export

### Task 4: Teacher wrapper + soft-label store + export CLI

**Files:**
- Create: `research/compression/softlabel_store.py`
- Create: `research/compression/teacher.py`
- Create: `research/compression/export_softlabels.py`
- Test: `tests/compression/test_softlabel_store.py`

- [ ] **Step 1: Write the failing test for the store (pure I/O)**

Create `tests/compression/test_softlabel_store.py`:
```python
import numpy as np
from compression.softlabel_store import SoftLabelWriter, SoftLabelReader


def test_write_then_read_round_trips_fp16(tmp_path):
    w = SoftLabelWriter(str(tmp_path / "sl"), x_bins=384, y_bins=512, num_kpts=17)
    sx = np.random.rand(17, 384).astype(np.float32)
    sy = np.random.rand(17, 512).astype(np.float32)
    w.add(ann_id=42, sx=sx, sy=sy)
    w.close()

    r = SoftLabelReader(str(tmp_path / "sl"))
    assert 42 in r.ann_ids()
    rx, ry = r.get(42)
    # stored as fp16 -> compare with fp16 tolerance
    assert np.allclose(rx, sx, atol=1e-2)
    assert np.allclose(ry, sy, atol=1e-2)
    assert rx.dtype == np.float16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/compression/test_softlabel_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compression.softlabel_store'`

- [ ] **Step 3: Implement `softlabel_store.py`**

Create `research/compression/softlabel_store.py`:
```python
"""On-disk fp16 store for teacher SimCC soft labels, keyed by COCO ann_id.

Layout: a single .npz-per-shard would reload everything; instead use a flat
memmap of fixed-size records + a JSON index of ann_id -> row. fp16 halves the
~30 KB/instance footprint.
"""
import json
import numpy as np
from pathlib import Path


class SoftLabelWriter:
    def __init__(self, prefix: str, x_bins: int, y_bins: int, num_kpts: int):
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self.x_bins, self.y_bins, self.num_kpts = x_bins, y_bins, num_kpts
        self.row_floats = num_kpts * (x_bins + y_bins)
        self._rows = []           # list of fp16 arrays
        self._index = {}          # ann_id -> row idx

    def add(self, ann_id: int, sx: np.ndarray, sy: np.ndarray):
        rec = np.concatenate([sx.reshape(-1), sy.reshape(-1)]).astype(np.float16)
        assert rec.size == self.row_floats
        self._index[int(ann_id)] = len(self._rows)
        self._rows.append(rec)

    def close(self):
        arr = np.stack(self._rows, axis=0) if self._rows else \
            np.zeros((0, self.row_floats), np.float16)
        np.save(str(self.prefix) + ".npy", arr)
        meta = {"x_bins": self.x_bins, "y_bins": self.y_bins,
                "num_kpts": self.num_kpts, "index": self._index}
        Path(str(self.prefix) + ".json").write_text(json.dumps(meta))


class SoftLabelReader:
    def __init__(self, prefix: str):
        meta = json.loads(Path(str(prefix) + ".json").read_text())
        self.x_bins, self.y_bins, self.num_kpts = meta["x_bins"], meta["y_bins"], meta["num_kpts"]
        self._index = {int(k): v for k, v in meta["index"].items()}
        self._arr = np.load(str(prefix) + ".npy", mmap_mode="r")

    def ann_ids(self):
        return set(self._index.keys())

    def get(self, ann_id: int):
        row = self._arr[self._index[int(ann_id)]]
        nx = self.num_kpts * self.x_bins
        sx = np.asarray(row[:nx]).reshape(self.num_kpts, self.x_bins)
        sy = np.asarray(row[nx:]).reshape(self.num_kpts, self.y_bins)
        return sx, sy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/compression/test_softlabel_store.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Implement the teacher wrapper**

Create `research/compression/teacher.py`:
```python
"""RTMPose-m teacher: produce SimCC logits for a preprocessed input crop.

Reuses the project's Pose2D plumbing only for the rtmlib pose model + its simcc
capture hook. We call the underlying ONNX session directly on our own
already-warped crop so the teacher and student see identical inputs.
"""
import numpy as np
from pose2d import Pose2D  # from src/, on the pytest/app path


class Teacher:
    def __init__(self):
        self._p = Pose2D(mode="balanced", accelerator="coreml")
        self._sess = self._p._body.pose_model.session
        self._inp_name = self._sess.get_inputs()[0].name

    def infer_simcc(self, input_chw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """input_chw: (3, H, W) normalized. Returns (sx (K,Xbins), sy (K,Ybins))."""
        batch = input_chw[None].astype(np.float32)  # (1, 3, H, W)
        outs = self._sess.run(None, {self._inp_name: batch})
        sx, sy = outs[0], outs[1]  # (1, K, Xbins), (1, K, Ybins)
        return sx[0].astype(np.float32), sy[0].astype(np.float32)
```

- [ ] **Step 6: Implement the export CLI**

Create `research/compression/export_softlabels.py`:
```python
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    img_dir = config.COCO_ROOT / f"{args.split}2017"
    ann = config.COCO_ROOT / "annotations" / f"person_keypoints_{args.split}2017.json"
    ds = CocoTopDown(str(img_dir), str(ann))
    teacher = Teacher()

    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    out_prefix = config.SOFTLABEL_DIR / f"{args.split}"
    writer = SoftLabelWriter(str(out_prefix), config.SIMCC_X_BINS,
                             config.SIMCC_Y_BINS, config.NUM_KEYPOINTS)
    for i in tqdm(range(n), desc=f"teacher {args.split}"):
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


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Smoke-verify the export against a small slice (requires COCO val + RTMPose weights)**

Run (only if `data/coco/val2017` + RTMPose weights present):
```bash
uv run python -m compression.export_softlabels --split val --limit 50
```
Expected: a tqdm bar to 50, no assertion error, final line `wrote 50 soft labels to .../val.npy`, and `models/compression/softlabels/val.npy` exists.

If COCO isn't downloaded yet, skip this step and run it after Task 3b's download.

- [ ] **Step 8: Commit**

```bash
git add research/compression/softlabel_store.py research/compression/teacher.py \
        research/compression/export_softlabels.py tests/compression/test_softlabel_store.py
git commit -m "feat(compression): teacher soft-label exporter + fp16 store"
```

---

## Phase 5 — Student model

### Task 5: CSPNeXt-lite backbone + SimCC head + student

**Files:**
- Create: `research/compression/models/__init__.py` (empty)
- Create: `research/compression/models/backbone.py`
- Create: `research/compression/models/head.py`
- Create: `research/compression/models/student.py`
- Test: `tests/compression/test_student.py`

- [ ] **Step 1: Write the failing test**

Create `tests/compression/test_student.py`:
```python
import torch
from compression import config
from compression.models.student import StudentPose


def test_student_output_shapes():
    model = StudentPose(width=config.STUDENT_WIDTH, depth=config.STUDENT_DEPTH)
    x = torch.randn(2, 3, config.INPUT_H, config.INPUT_W)
    sx, sy = model(x)
    assert sx.shape == (2, config.NUM_KEYPOINTS, config.SIMCC_X_BINS)
    assert sy.shape == (2, config.NUM_KEYPOINTS, config.SIMCC_Y_BINS)


def test_student_param_count_smaller_than_rtmpose_m():
    model = StudentPose(width=config.STUDENT_WIDTH, depth=config.STUDENT_DEPTH)
    params = sum(p.numel() for p in model.parameters())
    # RTMPose-m is ~13M; the lite student must be clearly smaller.
    assert params < 8_000_000, f"student too big: {params}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/compression/test_student.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compression.models.student'`

- [ ] **Step 3: Implement the backbone**

Create `research/compression/models/backbone.py`:
```python
"""CSPNeXt-lite: a compact conv backbone (pure PyTorch, MPS-friendly ops only).

Five stages, each halving spatial resolution, channels scaled by `width`.
Stem stride 2 + 4 stages stride 2 -> /32 total. Output is the last feature map.
"""
import torch.nn as nn


def _ch(base, width):
    return max(8, int(round(base * width / 8) * 8))


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = ConvBNAct(c, c, 3)
        self.c2 = ConvBNAct(c, c, 3)

    def forward(self, x):
        return x + self.c2(self.c1(x))


class CSPNeXtLite(nn.Module):
    def __init__(self, width=0.33, depth=0.33):
        super().__init__()
        base = [64, 128, 256, 512, 1024]
        nblocks = max(1, int(round(3 * depth)))
        chs = [_ch(b, width) for b in base]
        self.stem = ConvBNAct(3, chs[0], 3, s=2)
        stages = []
        cin = chs[0]
        for cout in chs[1:]:
            layers = [ConvBNAct(cin, cout, 3, s=2)]
            layers += [Bottleneck(cout) for _ in range(nblocks)]
            stages.append(nn.Sequential(*layers))
            cin = cout
        self.stages = nn.ModuleList(stages)
        self.out_channels = chs[-1]

    def forward(self, x):
        x = self.stem(x)
        for st in self.stages:
            x = st(x)
        return x  # (B, out_channels, H/32, W/32)
```

- [ ] **Step 4: Implement the SimCC head**

Create `research/compression/models/head.py`:
```python
"""SimCC head: backbone feature -> per-keypoint 1D logits over x and y bins.

Reduce to K channels, flatten the spatial map per keypoint, then two linear
classifiers (shared across keypoints) to the x/y bin axes.
"""
import torch
import torch.nn as nn


class SimCCHead(nn.Module):
    def __init__(self, in_channels, num_kpts, feat_h, feat_w, x_bins, y_bins):
        super().__init__()
        self.num_kpts = num_kpts
        self.reduce = nn.Conv2d(in_channels, num_kpts, 1)
        flat = feat_h * feat_w
        self.x_fc = nn.Linear(flat, x_bins)
        self.y_fc = nn.Linear(flat, y_bins)

    def forward(self, feat):
        b = feat.shape[0]
        x = self.reduce(feat)                      # (B, K, h, w)
        x = x.flatten(2)                           # (B, K, h*w)
        sx = self.x_fc(x)                          # (B, K, x_bins)
        sy = self.y_fc(x)                          # (B, K, y_bins)
        return sx, sy
```

- [ ] **Step 5: Implement the student**

Create `research/compression/models/student.py`:
```python
"""Student = CSPNeXt-lite backbone + SimCC head, sized from config.

`input_scale` is the resolution-specialization lever: it downsamples the input
crop fed to the backbone while the SimCC bin counts stay fixed at the teacher's
(384/512), so distillation targets stay aligned and decoded keypoints remain in
the canonical 256x192 space. width/depth shrink the backbone.
"""
import torch.nn as nn
import torch.nn.functional as F
from .. import config
from .backbone import CSPNeXtLite
from .head import SimCCHead


class StudentPose(nn.Module):
    def __init__(self, width=config.STUDENT_WIDTH, depth=config.STUDENT_DEPTH,
                 input_scale: float = 1.0):
        super().__init__()
        self.input_scale = input_scale
        self.arch = {"width": width, "depth": depth, "input_scale": input_scale}
        self.backbone = CSPNeXtLite(width=width, depth=depth)
        feat_h = int(config.INPUT_H * input_scale) // 32
        feat_w = int(config.INPUT_W * input_scale) // 32
        self.head = SimCCHead(
            in_channels=self.backbone.out_channels,
            num_kpts=config.NUM_KEYPOINTS,
            feat_h=feat_h, feat_w=feat_w,
            x_bins=config.SIMCC_X_BINS, y_bins=config.SIMCC_Y_BINS,
        )

    def forward(self, x):
        if self.input_scale != 1.0:
            x = F.interpolate(x, scale_factor=self.input_scale,
                              mode="bilinear", align_corners=False)
        return self.head(self.backbone(x))


def save_student(model: "StudentPose", path):
    """Checkpoint the weights AND the arch dict so any variant reloads exactly."""
    import torch
    torch.save({"state_dict": model.state_dict(), "arch": model.arch}, path)


def load_student(path, device="cpu") -> "StudentPose":
    import torch
    blob = torch.load(path, map_location=device)
    if isinstance(blob, dict) and "arch" in blob:
        model = StudentPose(**blob["arch"])
        model.load_state_dict(blob["state_dict"])
    else:  # backward-compat: a bare state_dict
        model = StudentPose()
        model.load_state_dict(blob)
    return model.to(device).eval()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/compression/test_student.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add research/compression/models/ tests/compression/test_student.py
git commit -m "feat(compression): CSPNeXt-lite student with SimCC head"
```

---

## Phase 6 — Losses

### Task 6: GT KLDiscret loss + teacher KD loss

**Files:**
- Create: `research/compression/losses.py`
- Test: `tests/compression/test_losses.py`

- [ ] **Step 1: Write the failing test**

Create `tests/compression/test_losses.py`:
```python
import torch
from compression.losses import kl_discret_loss, kd_simcc_loss


def test_kl_discret_zero_when_pred_matches_target():
    target = torch.softmax(torch.randn(2, 17, 384), dim=-1)
    pred_logits = torch.log(target + 1e-9)  # softmax(log p) == p
    vis = torch.ones(2, 17)
    loss = kl_discret_loss(pred_logits, target, vis)
    assert loss.item() < 1e-4


def test_kl_discret_ignores_invisible_keypoints():
    target = torch.softmax(torch.randn(2, 17, 384), dim=-1)
    pred_logits = torch.randn(2, 17, 384)
    vis = torch.zeros(2, 17)  # nothing visible -> zero loss
    loss = kl_discret_loss(pred_logits, target, vis)
    assert loss.item() == 0.0


def test_kd_loss_decreases_as_student_approaches_teacher():
    t_logits = torch.randn(2, 17, 384)
    far = kd_simcc_loss(torch.zeros_like(t_logits), t_logits, T=1.0)
    near = kd_simcc_loss(t_logits.clone(), t_logits, T=1.0)
    assert near.item() < far.item()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/compression/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compression.losses'`

- [ ] **Step 3: Implement `losses.py`**

Create `research/compression/losses.py`:
```python
"""SimCC training losses.

kl_discret_loss: student logits vs Gaussian GT target (RTMPose's GT loss),
                 masked by per-keypoint visibility.
kd_simcc_loss:   student logits vs teacher logits (knowledge distillation),
                 temperature-softened KL on both axes.
"""
import torch
import torch.nn.functional as F


def _axis_kl(pred_logits, target_prob, vis):
    logp = F.log_softmax(pred_logits, dim=-1)          # (B, K, bins)
    kl = (target_prob * (torch.log(target_prob + 1e-9) - logp)).sum(-1)  # (B, K)
    denom = vis.sum().clamp(min=1.0)
    return (kl * vis).sum() / denom


def kl_discret_loss(pred_logits, target_prob, vis):
    """pred_logits/target on one axis: (B, K, bins). target is a (sums-to-1) prob.
    Here target may be an un-normalized Gaussian; normalize it first."""
    target_prob = target_prob / (target_prob.sum(-1, keepdim=True) + 1e-9)
    return _axis_kl(pred_logits, target_prob, vis)


def kd_simcc_loss(student_logits, teacher_logits, T: float = 1.0):
    """Temperature-softened KL(teacher || student) on one axis. Mean over all kpts."""
    t = F.softmax(teacher_logits / T, dim=-1)
    logs = F.log_softmax(student_logits / T, dim=-1)
    kl = (t * (torch.log(t + 1e-9) - logs)).sum(-1)    # (B, K)
    return (T * T) * kl.mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/compression/test_losses.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add research/compression/losses.py tests/compression/test_losses.py
git commit -m "feat(compression): SimCC GT + distillation losses"
```

---

## Phase 7 — Training loop

### Task 7: MPS training CLI

**Files:**
- Create: `research/compression/train.py`
- Test: (verification runs — no unit test; the components are already covered)

- [ ] **Step 1: Implement `train.py`**

Create `research/compression/train.py`:
```python
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


def train(args):
    dev = _device()
    print(f"device={dev}")
    img_dir = config.COCO_ROOT / f"{args.split}2017"
    ann = config.COCO_ROOT / "annotations" / f"person_keypoints_{args.split}2017.json"
    ds = CocoTopDown(str(img_dir), str(ann))
    reader = SoftLabelReader(str(config.SOFTLABEL_DIR / args.split))
    train_set = TrainSet(ds, reader, limit=args.limit)
    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True)

    model = StudentPose(width=args.width, depth=args.depth,
                        input_scale=args.input_scale).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
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
            loss = args.w_gt * l_gt + args.w_kd * l_kd
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        print(f"epoch {ep}: mean loss {running / max(1, len(loader)):.4f}")
        save_student(model, config.CHECKPOINT_DIR / f"student{args.tag}_ep{ep}.pt")
    save_student(model, config.CHECKPOINT_DIR / f"student{args.tag}_final.pt")


def main():
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
    train(ap.parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-verify the loop overfits a tiny slice on MPS**

After Task 3b (COCO downloaded) and Task 4 Step 7 (a `train` soft-label slice exists — run `uv run python -m compression.export_softlabels --split train --limit 200` first), run:
```bash
uv run python -m compression.train --split train --limit 64 --epochs 5 --batch 8 --workers 0
```
Expected: prints `device=mps`, and the printed `mean loss` **decreases** across the 5 epochs (overfitting a tiny set is the correctness signal). A `student_final.pt` appears in `models/compression/checkpoints/`.

- [ ] **Step 3: Commit**

```bash
git add research/compression/train.py
git commit -m "feat(compression): MPS distillation training loop"
```

- [ ] **Step 4: (Verification, no commit) Full training run**

Once the smoke run looks right, train for real (start with a subset, then scale):
```bash
uv run python -m compression.export_softlabels --split train --limit 20000
uv run python -m compression.train --split train --limit 20000 --epochs 30 --batch 32
```
Record final loss + wall-clock in `research/compression/README.md` under a new "## Training runs" section. Commit that note.

### Task 7b: Specialization ablation variants

**Files:**
- Modify: `research/compression/README.md` (record the variant grid + results)

This realizes the spec's "task specialization" lever. Each variant is just a
training run with different `--width` / `--depth` / `--input_scale` and a `--tag`;
because `save_student`/`load_student` carry the arch dict, every eval CLI reloads
the right shape automatically (pass the variant's `--ckpt`).

- [ ] **Step 1: Train the variant grid**

Run (after the soft-label cache exists for the chosen `--limit`):
```bash
# baseline (already trained as student_final.pt)
uv run python -m compression.train --limit 20000 --epochs 30 --tag _w033
# smaller backbone
uv run python -m compression.train --limit 20000 --epochs 30 --width 0.25 --depth 0.25 --tag _w025
# reduced input resolution (bins stay fixed -> KD still aligned)
uv run python -m compression.train --limit 20000 --epochs 30 --input_scale 0.75 --tag _s075
```
Expected: three checkpoints `student_w033_final.pt`, `student_w025_final.pt`, `student_s075_final.pt`.

- [ ] **Step 2: Evaluate each variant (AP + latency)**

Run for each tag:
```bash
uv run python -m compression.eval.coco_ap --ckpt models/compression/checkpoints/student_w025_final.pt --limit 1000
uv run python -m compression.eval.benchmark --ckpt models/compression/checkpoints/student_w025_final.pt
```
Expected: an AP dict + a latency dict per variant.

- [ ] **Step 3: Record the variant grid in the README + commit**

Append a "## Specialization ablations" table (variant → AP, params, latency_ms) to `research/compression/README.md`, then:
```bash
git add research/compression/README.md
git commit -m "docs(compression): specialization ablation results"
```

---

## Phase 8 — Evaluation

### Task 8: COCO AP (GT bbox)

**Files:**
- Create: `research/compression/eval/__init__.py` (empty)
- Create: `research/compression/eval/coco_ap.py`
- Test: `tests/compression/test_coco_ap.py`

- [ ] **Step 1: Write the failing test (predicting GT yields AP ≈ 1.0)**

Create `tests/compression/test_coco_ap.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/compression/test_coco_ap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compression.eval.coco_ap'`

- [ ] **Step 3: Implement `coco_ap.py`**

Create `research/compression/eval/coco_ap.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/compression/test_coco_ap.py -v`
Expected: PASS (1 passed); pycocotools prints its summary table.

- [ ] **Step 5: Add a student→predictions helper + a val-AP CLI in the same module**

Append to `research/compression/eval/coco_ap.py`:
```python
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
```

- [ ] **Step 6: Verify val AP runs end to end (requires COCO val + a trained ckpt)**

Run:
```bash
uv run python -m compression.eval.coco_ap --limit 200
```
Expected: a pycocotools summary table + a printed dict like `{'AP': 0.5x, ...}`. (Absolute value depends on training; the check here is that it runs and returns a plausible 0–1 AP.)

- [ ] **Step 7: Commit**

```bash
git add research/compression/eval/__init__.py research/compression/eval/coco_ap.py \
        tests/compression/test_coco_ap.py
git commit -m "feat(compression): COCO AP eval (GT bbox) + val CLI"
```

### Task 9: Task-specific eval — keypoint error + downstream angle agreement

**Files:**
- Create: `research/compression/eval/task_eval.py`
- Test: `tests/compression/test_task_eval.py`

- [ ] **Step 1: Write the failing test**

Create `tests/compression/test_task_eval.py`:
```python
import numpy as np
from compression.eval.task_eval import angle_agreement, upper_body_pck


def test_angle_agreement_zero_when_identical():
    kps = np.random.rand(17, 2).astype(np.float32) * 100 + 50
    scores = np.ones(17, dtype=np.float32)
    diffs = angle_agreement(kps, scores, kps, scores)
    assert abs(diffs["head_lateral_tilt_2d"]) < 1e-6


def test_pck_perfect_when_identical():
    kps = np.random.rand(17, 2).astype(np.float32) * 100 + 50
    pck = upper_body_pck(kps, kps, threshold_px=5.0)
    assert pck == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/compression/test_task_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compression.eval.task_eval'`

- [ ] **Step 3: Implement `task_eval.py`**

Create `research/compression/eval/task_eval.py`:
```python
"""Product-relevant evaluation: does the student preserve the *measurements*
the scoring depends on? Compares student vs teacher (or vs GT) keypoints through
analysis.angles, plus PCK on the upper-body subset the rules use.
"""
import numpy as np
from analysis.angles import (
    head_lateral_tilt_2d, craniovertebral_angle_2d,
    NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
)

UPPER_BODY = [NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]


def angle_agreement(kps_a, scores_a, kps_b, scores_b) -> dict:
    """Absolute difference in each scoring angle between two keypoint sets.
    NaN-safe: a metric that is NaN for either set is reported as NaN."""
    def diff(fn):
        va, vb = fn(kps_a, scores_a), fn(kps_b, scores_b)
        if np.isnan(va) or np.isnan(vb):
            return float("nan")
        return abs(va - vb)
    return {
        "head_lateral_tilt_2d": diff(head_lateral_tilt_2d),
        "craniovertebral_angle_2d": diff(craniovertebral_angle_2d),
    }


def upper_body_pck(kps_pred, kps_gt, threshold_px: float = 10.0) -> float:
    """Fraction of upper-body keypoints within threshold_px of GT."""
    sel = np.array(UPPER_BODY)
    d = np.linalg.norm(kps_pred[sel] - kps_gt[sel], axis=1)
    return float((d <= threshold_px).mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/compression/test_task_eval.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Document the self-recorded clip protocol in the README**

Append to `research/compression/README.md`:
```markdown
## Task-specific eval clip

1. Record a ~30 s desk/neck-stretch clip with the webcam.
2. Sample ~30–50 frames; hand-label the 7 upper-body keypoints
   (nose, ears, shoulders, hips) — e.g. with `labelme` or a tiny click tool.
3. Save as a COCO-format json next to the frames.
4. Compare student vs teacher (and vs your labels) via
   `eval/task_eval.py`: report `upper_body_pck` and `angle_agreement`
   (the head-lateral-tilt / CVA differences the scoring actually consumes).
```

- [ ] **Step 6: Commit**

```bash
git add research/compression/eval/task_eval.py tests/compression/test_task_eval.py \
        research/compression/README.md
git commit -m "feat(compression): task-specific keypoint + angle-agreement eval"
```

### Task 10: Size + latency benchmark

**Files:**
- Create: `research/compression/eval/benchmark.py`
- Test: (verification run — timing has no meaningful unit test)

- [ ] **Step 1: Implement `benchmark.py`**

Create `research/compression/eval/benchmark.py`:
```python
"""Model size + latency on MPS (and CPU). ONNX/CoreML latency is measured in
quantize/ptq_coreml.py after export. Usage:
  uv run python -m compression.eval.benchmark --ckpt models/compression/checkpoints/student_final.pt
"""
import argparse
import time
import os
import torch
from .. import config
from ..models.student import load_student


def benchmark(ckpt: str, iters: int = 100) -> dict:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_student(ckpt, dev)
    params = sum(p.numel() for p in model.parameters())
    size_mb = os.path.getsize(ckpt) / 1e6
    x = torch.randn(1, 3, config.INPUT_H, config.INPUT_W, device=dev)
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(x)
        if dev == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if dev == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return {"device": dev, "params": params, "ckpt_size_mb": round(size_mb, 2),
            "latency_ms": round(dt * 1000, 2), "fps": round(1.0 / dt, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.CHECKPOINT_DIR / "student_final.pt"))
    print(benchmark(ap.parse_args().ckpt))
```

- [ ] **Step 2: Verify it runs (requires a trained ckpt)**

Run:
```bash
uv run python -m compression.eval.benchmark
```
Expected: a dict with `device=mps`, `params` < 8M, a `latency_ms`, and an `fps`.

- [ ] **Step 3: Commit**

```bash
git add research/compression/eval/benchmark.py
git commit -m "feat(compression): size + MPS latency benchmark"
```

---

## Phase 9 — Quantization

### Task 11: PTQ to CoreML int8

**Files:**
- Create: `research/compression/quantize/__init__.py` (empty)
- Create: `research/compression/quantize/ptq_coreml.py`
- Test: (verification run — coremltools conversion has no fast unit test)

- [ ] **Step 1: Implement `ptq_coreml.py`**

Create `research/compression/quantize/ptq_coreml.py`:
```python
"""Export the student to CoreML and apply int8 post-training quantization.

Pipeline: torch -> ONNX -> CoreML (mlprogram) -> linear int8 weight
quantization via coremltools. Reports the size before/after.

Usage:
  uv run python -m compression.quantize.ptq_coreml --ckpt models/compression/checkpoints/student_final.pt
"""
import argparse
import os
import torch
import coremltools as ct
import coremltools.optimize.coreml as cto
from .. import config
from ..models.student import load_student


def export(ckpt: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    model = load_student(ckpt, "cpu")  # export from CPU for deterministic tracing
    example = torch.randn(1, 3, config.INPUT_H, config.INPUT_W)
    traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
    )
    fp_path = os.path.join(out_dir, "student_fp16.mlpackage")
    mlmodel.save(fp_path)

    cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    )
    q = cto.linear_quantize_weights(mlmodel, config=cfg)
    int8_path = os.path.join(out_dir, "student_int8.mlpackage")
    q.save(int8_path)

    def _dir_mb(p):
        total = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(p) for f in fs)
        return round(total / 1e6, 2)

    return {"fp16_mb": _dir_mb(fp_path), "int8_mb": _dir_mb(int8_path),
            "fp16_path": fp_path, "int8_path": int8_path}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.CHECKPOINT_DIR / "student_final.pt"))
    ap.add_argument("--out", default=str(config.MODELS_DIR / "compression" / "coreml"))
    args = ap.parse_args()
    print(export(args.ckpt, args.out))
```

- [ ] **Step 2: Verify export + quantization (requires a trained ckpt)**

Run:
```bash
uv run python -m compression.quantize.ptq_coreml
```
Expected: a dict with `int8_mb` meaningfully smaller than `fp16_mb`, and both `.mlpackage` paths exist.

- [ ] **Step 3: Time the CoreML int8 model on the ANE**

Append a CoreML latency helper to `research/compression/quantize/ptq_coreml.py`:
```python
def benchmark_coreml(mlpackage_path: str, iters: int = 100) -> dict:
    import time
    import numpy as np
    import coremltools as ct
    m = ct.models.MLModel(mlpackage_path)
    x = np.random.randn(1, 3, config.INPUT_H, config.INPUT_W).astype(np.float32)
    for _ in range(10):
        m.predict({"input": x})  # warmup / compile
    t0 = time.perf_counter()
    for _ in range(iters):
        m.predict({"input": x})
    dt = (time.perf_counter() - t0) / iters
    return {"latency_ms": round(dt * 1000, 2), "fps": round(1.0 / dt, 1)}
```

Run:
```bash
uv run python -c "from compression.quantize.ptq_coreml import benchmark_coreml; print(benchmark_coreml('models/compression/coreml/student_int8.mlpackage'))"
```
Expected: a `latency_ms` / `fps` dict for the int8 model running on CoreML (compute units = ALL, i.e. ANE/GPU).

- [ ] **Step 4: Document the AP-after-PTQ check in the README**

Append to `research/compression/README.md`:
```markdown
## Quantization accuracy check

After PTQ, re-run COCO val AP through the CoreML int8 model (load via
`coremltools` and feed the same warped crops as `eval/coco_ap.run_val_ap`,
swapping the torch forward for `mlmodel.predict`). Report AP_int8 vs AP_fp.
A small drop (≤ ~1–2 AP) is the success criterion; if larger, fall back to QAT.
```

- [ ] **Step 5: Commit**

```bash
git add research/compression/quantize/ research/compression/README.md
git commit -m "feat(compression): PTQ export to CoreML int8"
```

---

## Phase 10 — Baselines & frontier (independent add-on)

### Task 12: Collect results + plot the accuracy/size/latency frontier

**Files:**
- Create: `research/compression/compare/__init__.py` (empty)
- Create: `research/compression/compare/frontier.py`
- Test: (verification — plotting has no meaningful unit test)

- [ ] **Step 1: Implement `frontier.py`**

Create `research/compression/compare/frontier.py`:
```python
"""Gather {name -> (AP, size_mb, latency_ms)} into a results table + Pareto plots.

Baseline numbers (RTMPose-t/s/m, MoveNet, MediaPipe) are entered here from their
own benchmark runs / published values; this module only tabulates and plots.
Fairness caveat: MoveNet is COCO-17; MediaPipe BlazePose is 33-kpt — compare on
overlapping joints and annotate cross-topology entries.
"""
import argparse
import json
import matplotlib.pyplot as plt


def plot_frontier(results: dict, out_png: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, r in results.items():
        ax1.scatter(r["latency_ms"], r["AP"]); ax1.annotate(name, (r["latency_ms"], r["AP"]))
        ax2.scatter(r["size_mb"], r["AP"]); ax2.annotate(name, (r["size_mb"], r["AP"]))
    ax1.set_xlabel("latency (ms)"); ax1.set_ylabel("COCO AP"); ax1.set_title("Accuracy vs Latency")
    ax2.set_xlabel("model size (MB)"); ax2.set_ylabel("COCO AP"); ax2.set_title("Accuracy vs Size")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="JSON: {name: {AP, size_mb, latency_ms}}")
    ap.add_argument("--out", default="research/compression/frontier.png")
    a = ap.parse_args()
    plot_frontier(json.loads(open(a.results).read()), a.out)
```

- [ ] **Step 2: Verify with a synthetic results file**

Run:
```bash
uv run python -c "import json; json.dump({'student-int8':{'AP':0.62,'size_mb':4.0,'latency_ms':6.0},'rtmpose-m':{'AP':0.75,'size_mb':52.0,'latency_ms':18.6}}, open('/tmp/r.json','w'))"
uv run python -m compression.compare.frontier --results /tmp/r.json --out /tmp/frontier.png
```
Expected: `wrote /tmp/frontier.png` and the file exists.

- [ ] **Step 3: Document the baseline-gathering procedure in the README**

Append to `research/compression/README.md`:
```markdown
## Baselines for the frontier

- RTMPose-t/s/m: run `eval/coco_ap.run_val_ap`-style eval with each rtmlib tier
  as the model; latency from `eval/benchmark` equivalents.
- MoveNet (Lightning/Thunder): TF-Hub / TFLite; map its COCO-17 output directly.
- MediaPipe BlazePose: `mediapipe` package; compare only overlapping joints.
Assemble all rows into a results JSON and run `compare/frontier.py`.
```

- [ ] **Step 4: Commit**

```bash
git add research/compression/compare/ research/compression/README.md
git commit -m "feat(compression): accuracy/size/latency frontier plot"
```

---

## Phase 11 — App integration (stretch, independent add-on)

### Task 13: Optional flag to run the student inside the live app

**Files:**
- Modify: `src/pose2d.py` (add a `StudentPose2D` class — separate from `Pose2D`, same interface)
- Test: `tests/compression/test_student_pose2d_smoke.py` (skips without a ckpt)

- [ ] **Step 1: Write a skip-gated smoke test**

Create `tests/compression/test_student_pose2d_smoke.py`:
```python
import os
import numpy as np
import pytest

CKPT = "models/compression/coreml/student_int8.mlpackage"


@pytest.mark.skipif(not os.path.exists(CKPT), reason="no compressed model yet")
def test_student_pose2d_returns_17_keypoints():
    from pose2d import StudentPose2D
    p = StudentPose2D(CKPT)
    frame = np.full((480, 640, 3), 127, np.uint8)
    kps, scores = p.infer(frame)
    assert kps.shape == (17, 2)
    assert scores.shape == (17,)
```

- [ ] **Step 2: Run test to verify it skips (no ckpt) or fails on missing class**

Run: `uv run pytest tests/compression/test_student_pose2d_smoke.py -v`
Expected: SKIPPED if no `.mlpackage` present; otherwise FAIL with `ImportError: cannot import name 'StudentPose2D'`.

- [ ] **Step 3: Implement `StudentPose2D` in `src/pose2d.py`**

Append to `src/pose2d.py` (keep `Pose2D` unchanged; this is an additive sibling that reuses the project's detector for the bbox, then runs the CoreML student for pose):
```python
class StudentPose2D:
    """Drop-in pose source backed by the compressed CoreML student.

    Same (keypoints, scores) contract as Pose2D.infer. Uses Pose2D's detector
    for the person bbox, then the int8 student for keypoints. Requires
    coremltools + a top-down crop matching the training transform.
    """

    def __init__(self, mlpackage_path: str):
        import coremltools as ct
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "research"))
        from compression.transforms import (
            bbox_to_center_scale, get_warp_matrix, warp_keypoints, warp_image,
        )
        from compression.simcc import decode_simcc

        self._ct_model = ct.models.MLModel(mlpackage_path)
        self._det = Pose2D(mode="balanced", accelerator="coreml")._body.det_model
        self._bbox_to_center_scale = bbox_to_center_scale
        self._get_warp_matrix = get_warp_matrix
        self._warp_keypoints = warp_keypoints
        self._warp_image = warp_image
        self._decode_simcc = decode_simcc

    def infer(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from compression import config as ccfg

        bboxes = self._det(image_bgr)
        if len(bboxes) == 0:
            return np.zeros((17, 2), np.float32), np.zeros((17,), np.float32)
        x1, y1, x2, y2 = bboxes[0][:4]
        bbox = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
        center, scale = self._bbox_to_center_scale(bbox)
        inp = self._warp_image(image_bgr, center, scale)[None]  # (1,3,H,W)
        out = self._ct_model.predict({"input": inp})
        keys = list(out.keys())
        sx, sy = out[keys[0]], out[keys[1]]
        kps_in, scores = self._decode_simcc(np.asarray(sx), np.asarray(sy))
        Minv = self._get_warp_matrix(center, scale, (ccfg.INPUT_W, ccfg.INPUT_H), inverse=True)
        kps_img = self._warp_keypoints(kps_in[0], Minv)
        return kps_img.astype(np.float32), scores[0].astype(np.float32)
```

- [ ] **Step 4: Run the smoke test (passes once a model exists, skips otherwise)**

Run: `uv run pytest tests/compression/test_student_pose2d_smoke.py -v`
Expected: PASS if the int8 `.mlpackage` exists; SKIPPED otherwise.

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

Run: `uv run pytest -k "not smoke"`
Expected: all prior tests still pass; the new compression pure-logic tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pose2d.py tests/compression/test_student_pose2d_smoke.py
git commit -m "feat(compression): StudentPose2D sibling source for the live app"
```

---

## Done criteria

- `uv run pytest tests/compression` is green (all pure-logic tasks).
- A trained `student_final.pt` exists and `eval/coco_ap.py` prints a COCO AP; `eval/task_eval.py` reports angle agreement vs the teacher; `eval/benchmark.py` reports params/latency.
- `quantize/ptq_coreml.py` produces an int8 `.mlpackage` smaller than fp16, with a documented AP delta.
- `compare/frontier.py` plots the student against RTMPose tiers / MoveNet / MediaPipe.
- README documents the full run order: download COCO → export softlabels → train → eval → quantize → frontier.
