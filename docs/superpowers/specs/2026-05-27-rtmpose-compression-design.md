# Compressed, Desk-Specialized RTMPose for On-Device Form Coaching — Design

**Date:** 2026-05-27
**Status:** Draft (pending spec review)
**Type:** Research / model-training deliverable (bachelor's senior project)
**Relation to app:** The trained model targets the existing `Pose2D` slot (`src/pose2d.py`). Training and benchmarking code lives outside the runtime pipeline (`research/compression/`); only the final exported model artifact touches `models/`.

## Goal

Produce a **smaller, faster 2D pose model that matches RTMPose-m's accuracy on the keypoints this project actually uses**, trained and benchmarked end-to-end, and positioned against current on-device pose models ("the market"). The deliverable is the *training + compression methodology and its evaluation* — the academic contribution — not beating a shipping product head-on.

The 2D keypoints are the model that does the real work here: `analysis/angles.py` computes all form measurements (head lateral tilt, CVA, shoulder asymmetry, etc.) directly from 2D COCO keypoints, and the rule-based scorer consumes those. The 3D lifter is visualization-only. So compressing the **2D** model is both the highest-leverage product improvement and a clean, benchmarkable thesis target.

## Decisions (from brainstorming)

| Decision | Choice |
| --- | --- |
| Model under study | **RTMPose** (2D keypoints), not the 2D→3D lifter (lifter dropped — viz-only, not on the scoring path) |
| Objective framing | **Academic deliverable** — contribution + benchmark matter more than beating a product |
| Headline technique | **Knowledge distillation** — RTMPose-m teacher → smaller student |
| Specialization lever | **Task specialization for the desk/upper-body case** — reduced input resolution + slimmer backbone (keeps 17 keypoints for AP comparability); keypoint-subset head as an ablation |
| Stacked technique | **Int8 quantization** (PTQ; QAT if time permits) for the final on-device model |
| Pruning | **Out of scope** for the core (kept as a possible later ablation; trimmed to lower risk) |
| Accuracy metric | **Both** — standard COCO AP (comparability) **and** task-specific keypoint/angle accuracy on a self-recorded clip |
| Training compute | **Local on the M5 Max**, via a **pure-PyTorch MPS-native loop** (NOT `mmpose`/`mmcv`, which is CUDA-bound and breaks on MPS) |
| Data | **Public only** — COCO 2017 for training/AP; a tiny self-recorded, hand-labeled clip for task-specific eval |

## Background: why this is the right target, and the compute constraint

- **RTMPose-m is already SOTA-class.** We will not beat it on general COCO accuracy. The defensible contribution is **compression + domain specialization**: same task-relevant accuracy at a fraction of the size/latency, on Apple Silicon / ANE.
- **`mmpose`/`mmcv` is the real blocker, not the silicon.** The stock RTMPose training stack relies on custom CUDA ops; on MPS it falls back to CPU or breaks. An M5 Max has ample GPU + unified memory, so the methodology is built around a **pure-PyTorch student trained on MPS** with a **frozen teacher whose soft labels are precomputed offline** via the existing ONNX model. This sidesteps CUDA entirely.

## Method

### 1. Teacher and soft-label precomputation

- **Teacher:** the existing RTMPose-m ONNX model (already in `models/rtmlib_cache/`), run via the current ONNX/CoreML path.
- RTMPose's head is **SimCC** (coordinate classification: separate 1D distributions over binned x and y per keypoint). These `simcc_x` / `simcc_y` tensors are already reachable — `pose2d.infer_with_heatmaps` reconstructs heatmaps from them today.
- **Precompute once:** run the teacher over the COCO train set (GT person crops) and **cache `simcc_x` / `simcc_y` as fp16** soft labels. This decouples training from the teacher (no live teacher inference, no CUDA), and makes the MPS training loop fast.
- Storage estimate: ~17 kpts × (≈384 + ≈512) bins × 2 bytes ≈ 30 KB/crop fp16. Full COCO train (~150k person instances) ≈ 4–5 GB. Mitigation if needed: subset COCO or store argmax + a few neighbors.

### 2. Student architecture (pure PyTorch)

- **Backbone:** a slimmed CSPNeXt-style CNN (reduced width/depth), reimplemented in plain PyTorch — no `mmcv`. Target params well below RTMPose-m (~13M); aim near RTMPose-t (~3.5M) or smaller after specialization.
- **Head:** SimCC head (1D classification over x/y bins), matching the teacher's representation so distillation is direct.
- Reimplementing (rather than running an mmpose config) is intentional: it runs natively on MPS **and** demonstrates architectural understanding for the thesis.

### 3. Distillation training

- **Loss:** `L = α · KL(student_simcc ‖ teacher_simcc) + β · KLDiscret(student_simcc ‖ gt_simcc)`, where `gt_simcc` is the Gaussian-smoothed ground-truth label RTMPose uses (`KLDiscretLoss`). Optionally add a feature-distillation term on an intermediate map.
- Teacher targets come from the fp16 cache (step 1); GT comes from COCO annotations.
- **Environment:** PyTorch MPS backend, `PYTORCH_ENABLE_MPS_FALLBACK=1` for any rare unsupported op. Standard ops only.

### 4. Task specialization (the "fit our case" lever)

Two specializations that **keep all 17 keypoints** (so COCO AP stays comparable):
- **Reduced input resolution:** RTMPose-m uses 256×192. Train student variants at e.g. 192×144 and 160×120 — large latency wins for a desk framing where the person is large and centered.
- **Slimmer backbone width/depth** tuned to the accuracy floor the scoring needs.

One specialization explored as an **ablation only** (breaks 17-kpt AP, so reported separately): a **keypoint-subset head** predicting just the ~9 upper-body joints the rules use (nose, ears, eyes, shoulders, hips) — shows the extra size/latency headroom available for the deployed model.

### 5. Quantization (stacked)

- **PTQ to int8** on the trained student: export → ONNX → CoreML, apply `coremltools` quantization/palettization with a calibration set; per-channel where supported.
- Measure the accuracy↔size↔latency change PTQ adds on top of distillation.
- **QAT** in PyTorch only if PTQ's accuracy drop is unacceptable and time permits (stretch).

## Datasets

- **Training / COCO AP:** COCO 2017 keypoints (public). Train split for distillation; val2017 for AP.
- **Task-specific eval:** a short self-recorded desk/stretch clip; hand-label upper-body keypoints on ~30–50 frames. A tiny eval set like this is consistent with "public-only **training**" — it is evaluation, not a collected training corpus.

## Evaluation protocol

1. **COCO AP** — val2017, **GT bounding boxes** (isolates pose accuracy from detection), `pycocotools` OKS AP/AR. Standard, comparable to the literature.
2. **Task-specific keypoint accuracy** — on the labeled clip: PCK and mean pixel error (normalized by shoulder width) for the upper-body joints the rules use.
3. **Downstream agreement (most product-relevant)** — feed student vs. teacher keypoints through `analysis/angles.py` and compare the **resulting angle measurements** (e.g. `head_lateral_tilt_2d`, `craniovertebral_angle_2d`). "Same performance" for *this app* ultimately means the scoring inputs don't change.
4. **Efficiency** — params, model size (MB), latency (ms/frame, fps) and CPU% on the **M5 Max (MPS)** and **CoreML/ANE**, matching the app's real config.

## Comparison to "the market"

Plot the **accuracy↔latency** and **accuracy↔size** Pareto frontiers for:
- The compressed student(s) (this work),
- RTMPose-t / -s / -m (the teacher is the accuracy ceiling),
- **MoveNet** (Lightning / Thunder),
- **MediaPipe BlazePose**.

Fairness caveat to document: MediaPipe/BlazePose use a different keypoint topology (33 pts) and MoveNet uses COCO-17 — compare on overlapping joints, and treat cross-topology numbers as indicative, not identical-metric.

## Deliverables

- `research/compression/` — teacher soft-label exporter, student model, MPS training loop, eval scripts, quantization scripts.
- Trained student checkpoint(s) + ONNX/CoreML exports.
- Benchmark tables + Pareto plots (AP / task error / size / latency).
- Thesis writeup material (method, results, discussion).
- **Stretch:** wire the exported student into `Pose2D` behind a flag so the live app can run the compressed model.

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| COCO training time on a single Mac | Distillation converges faster than from-scratch; precomputed teacher targets; smaller input res; subset COCO; fewer epochs |
| MPS op gaps even in pure PyTorch | Standard ops only; `PYTORCH_ENABLE_MPS_FALLBACK=1`; CPU fallback for rare ops |
| PTQ accuracy drop on ANE | Per-channel quant + calibration set; QAT as fallback |
| Teacher soft-label storage size | fp16 cache; subset COCO; store argmax + neighbors |
| Cross-model comparison fairness | Compare on overlapping joints; document topology differences |
| Specialization breaks COCO AP comparability | Resolution/width specializations keep 17 kpts; keypoint-subset only as a separately-reported ablation |

## Out of scope (YAGNI)

- Retraining the YOLOX **detector** (use existing detector or GT bbox for eval).
- The **2D→3D lifter** (dropped — viz-only, off the scoring path).
- **Multi-person** pose.
- **Pruning** as a core technique (possible later ablation only).
- **Streaming-server** integration (future direction; unaffected).
- Any **new large-scale data collection**.

## Open questions

- Exact student backbone config (width/depth multipliers) — pin down during planning vs. an initial sweep.
- Whether to precompute the full COCO teacher cache or train against a COCO subset first to validate the loop, then scale.
- Final on-device export target: ONNX→CoreML via `coremltools` vs. keeping ORT's CoreML EP path (matches current app).
