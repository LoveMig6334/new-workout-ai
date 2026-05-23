# Workout AI — Real-Time Form Coach (Thai)

Real-time webcam-based form coach for macOS Apple Silicon. The launcher opens an exercise selector, captures a short per-user calibration, then runs a hold-based session for the chosen exercise. Joint angles are measured **directly from 2D keypoints** (nose / ears / shoulders) — robust to the seated, desk-camera framing where the 3D lift fails (see `CLAUDE.md` → "2D-direct measurement"); the 3D rig on screen is visualization-only. **Currently registered**: `neck_stretch_left` — a 20-second left-neck lateral-tilt hold (vertical slice of the office-syndrome plug-in architecture). **Coming next** (content-only): right-neck, both shoulders, chest-and-shoulder, both hands, neck-flexion — the 2D measurement cookbook for these (CVA, forward-head, neck-flexion, shoulder asymmetry, wrist extension) already ships in `analysis/angles.py`. **Squat coaching code** is preserved in the codebase but is not currently wired to the launcher; see `CLAUDE.md` for how to re-enable it.

## Setup

```bash
uv sync --all-extras
uv run python scripts/download_models.py     # ~6 GB, one-time
```

## Run

```bash
uv run python main.py
```

Keys:

- **Selector**: number key next to the exercise (currently `1` for `neck_stretch_left`). `q` or Esc cancels.
- **Calibration**: each session opens with a ~5 s "sit naturally" capture that records your neutral posture; targets are then scored as a delta from your own neutral.
- **During a hold session**: `q` quits the session. The session also ends automatically after the 20s hold completes (Thai summary shown for ~3s, then back to the selector).

Diagnostic / debugging:

```bash
uv run python src/test_2D_3D.py            # live 3-panel view: camera / 2D pose + metrics / 3D rig, with profiling
uv run python scripts/profile_pipeline.py  # headless per-stage timing (CPU%, ms/stage)
```

## Stack

| Layer | Model | Framework |
|---|---|---|
| 2D pose | YOLOX-m + RTMPose-m (balanced; default) | rtmlib (ONNX) on **CoreML / Neural Engine** |
| 3D lift (visualization only) | MotionBERT-Lite | PyTorch on MPS |
| Form analysis | **2D-direct** joint angles + per-user calibration + camera-view gating + phase FSM | pure Python |
| Thai feedback | Qwen3.5-4B (mxfp4) | mlx-vlm |

The 2D model defaults to the balanced tier on the Neural Engine (54 fps; far more accurate keypoints than the old lightweight+CPU path). CPU stays cool — ONNX threads are pinned to 2 and inference is throttled to ~15 Hz. See `docs/perf/2026-05-23-coreml-experiment.md`.

## Acceptance criteria

Shared pipeline (both modes):

- [x] Skeleton overlay ≥ 25 FPS on M-series (balanced + CoreML ≈ 54 fps; full per-frame infer+lift ≈ 23 ms).
- [x] 3D rig updates at ≥ 5 Hz (6 Hz in `app`, up to 30 Hz in the diagnostic).
- [x] Models live in `./models/` and load from disk on subsequent runs.

### Stretches (current launcher mode)

- [x] Selector at startup lists registered exercises; number key picks one.
- [x] Hold FSM tracks `IDLE / ENTERING / HOLDING / DRIFTED / COMPLETE` with pause-on-drift timing and a stability penalty on drift count.
- [x] 100-pt score on completion (50 duration / 30 precision / 20 stability) + Thai summary from the on-device VLM.
- [x] Live Thai nudges during the hold (throttled to ~1 every 3–4 s by Qwen latency).
- [x] **2D-direct measurement** (nose+shoulders), robust to seated/desk-camera framing where the 3D lift is structurally invalid.
- [x] **Per-user calibration** — `neck_stretch_left` is scored as a delta from the user's captured neutral (`_NECK_TILT_TARGET_LEFT_DEG = -35°` is now a delta-from-neutral; still a rough value pending real-user tuning).
- [x] **Camera-view gating** — exercise declares `valid_views`; FSM stays IDLE with a "rotate" coaching message in an unsupported view.
- [ ] Remaining 9 exercises registered (right-neck, shoulders, chest, hands, neck-flexion). The 2D measurement functions they need already ship in `analysis/angles.py`.

### Squat (preserved code, not currently exposed)

The original `SquatFSM` + `rules_squat.score_rep` + `RepAnalysis` flow is intact in the codebase. The Thai LLM still handles `RepAnalysis` payloads via the dispatch in `ThaiCoachLLM.generate`. The acceptance bar from the original spec — ±1 rep over a 10-rep set, 2–3 sentence Thai feedback within 3 s of rep completion — is unchanged on the code path, but no entry point currently routes to it.

Design specs:

- [Office syndrome stretches](docs/superpowers/specs/2026-05-22-office-syndrome-stretches-design.md) — current launcher mode.
- [3D scoring upgrade](docs/superpowers/specs/2026-05-19-3d-scoring-design.md).
- [Original squat coach](docs/superpowers/specs/2026-05-18-pose-form-coach-design.md).
