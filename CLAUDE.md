# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Real-time webcam-based squat form coach for macOS Apple Silicon. Streams from the webcam, runs 2D pose → 3D pose → rule-based form analysis, and produces Thai-language coaching feedback from an on-device VLM at rep boundaries.

## Commands

This project uses `uv` (Python 3.12) — always invoke Python via `uv run`.

```bash
uv sync --all-extras                          # install + dev deps
uv run python scripts/download_models.py      # one-time, ~6 GB into ./models/
uv run python main.py                         # run the app (webcam window opens; q to quit, a to toggle attention overlay)

uv run pytest                                 # run full test suite
uv run pytest tests/test_phases.py            # single file
uv run pytest tests/test_phases.py::test_name # single test
uv run ruff check                             # lint
uv run ruff format                            # format
```

`pytest.ini_options` sets `pythonpath = ["src"]`, so tests import as `from workout_ai.* import ...` without an editable install.

## Architecture

Single-process pipeline driven by `src/workout_ai/app.py::run()`. The main loop reads webcam frames, runs them through pose inference and a finite-state machine, scores each completed rep, and ships the score to a background LLM thread for narration. The display is composed of the camera feed plus a right-hand panel that shows the 3D rig and Thai coaching text.

Data flow per frame:

```
WebcamCapture (bg thread)
   → Pose2D (YOLOX-tiny + RTMPose-s via rtmlib/ONNX, CPU) kps:(17,2) + scores + optional simcc heatmaps
       → coco17_to_h36m17 → Pose3DBuffer (27-frame window)
           → Pose3D (MotionBERT-Lite, PyTorch on MPS)      kps_3d:(17,3) — lifted every 5 frames
   → SquatFSM.update(kps, ts)                              STANDING → DESCENT → BOTTOM → ASCENT → STANDING
       → on rep complete: score_rep(bottom_frame) → RepAnalysis
           → LLMWorker.submit (bg thread, drop-stale)
               → ThaiCoachLLM (Qwen3.5-4B mxfp4 via mlx-vlm)
   → Renderer.compose(...) → cv2.imshow
```

### Coordinate systems & keypoint layouts

- **2D keypoints throughout `analysis/`**: COCO-17 indices (see `analysis/angles.py` for the canonical index constants). Image-pixel coords with y growing downward — that's why `hip_below_knee` checks `hip_y > knee_y`.
- **3D keypoints**: Human3.6M-17 layout. `pose3d.coco17_to_h36m17` synthesizes pelvis, spine, and thorax from COCO joints (indices marked `-1`, `-2`, `-3` in `COCO_TO_H36M`). Normalization in `_normalize_2d` divides both x and y by frame **width** (not height) — this matches MotionBERT's expected input convention; do not "fix" it.
- **Heatmaps for attention overlay**: reconstructed from RTMPose's simcc outputs via outer product (see `pose2d.infer_with_heatmaps`). This is a monkey-patch of `rtmlib`'s pose estimator; if `rtmlib` API shifts, the patch silently degrades and `heatmaps` becomes `None`.

### Rep detection and scoring

`analysis/phases.py::SquatFSM` is the single source of truth for "what counts as a rep." Thresholds (`STAND_THRESHOLD=160°`, `BOTTOM_THRESHOLD=100°`) operate on the average of left/right knee angles. A rep is committed only on the ASCENT→STANDING transition, which fires the `on_rep_complete(meta)` callback with `descent_ms` / `ascent_ms`.

`analysis/rules_squat.py::score_rep` is a pure function from a single `PoseFrame` (the bottom frame) + tempo to a `RepAnalysis`. Component budget totals 100: depth 30 / valgus 25 / torso 20 / symmetry 15 / tempo 10. Each violation also produces a Thai-language hint (`detail_th`) consumed by the LLM prompt — keep wording natural Thai when adding rules.

### LLM feedback

`feedback/llm.py::ThaiCoachLLM` wraps Qwen3.5-4B (mxfp4) via `mlx-vlm`. First `generate()` call is slow (compilation) — `app.run()` calls `warmup()` before entering the loop; preserve that order.

`feedback/worker.py::LLMWorker` runs in a background thread with **drop-stale** semantics: `submit(rep)` replaces any pending rep that hasn't been processed yet. There is no queue. If reps come faster than the LLM, older ones are silently dropped — by design, since stale feedback is worse than no feedback.

### Model artifacts

All models live under `./models/` (gitignored) and are downloaded by `scripts/download_models.py`:

- `models/rtmlib_cache/` — RTMPose-l ONNX (warmed by instantiating `rtmlib.Body`)
- `models/motionbert/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin` — MotionBERT-Lite weights (HF: `walterzhu/MotionBERT`)
- `models/qwen3_5_4b_mxfp4/` — Qwen3.5-4B mxfp4 mlx-vlm snapshot

`vendor/motionbert/` is a checked-out copy of the upstream MotionBERT repo (gitignored as `vendor/`). `pose3d.py` does `sys.path.insert(0, MOTIONBERT_DIR)` and imports `lib.model.DSTformer` + `lib.utils.tools.get_config` from it. The config file at `vendor/motionbert/configs/pose3d/MB_ft_h36m_global_lite.yaml` is required at construction time. If `vendor/` is missing or wiped, `Pose3D` cannot be constructed — re-clone MotionBERT into `vendor/motionbert/`.

### Threading model

Three background threads, all daemonized:

1. **WebcamCapture._loop** — pulls frames, keeps only the latest. `read_latest()` returns a copy with timeout.
2. **LLMWorker._loop** — condition-variable wait, drop-stale submission.
3. **Pose2D / Pose3D** themselves are not threaded; they run inline on the main loop.

The main loop is the only consumer of OpenCV's GUI (`cv2.imshow` / `cv2.waitKey`); don't call those from worker threads.

## Testing notes

- `tests/conftest.py` adds `src/` to `sys.path` (pytest config also does this — both are intentional, removing one breaks `python -m pytest` invocations).
- The `*_smoke.py` tests actually load the heavy models (RTMPose, MotionBERT, Qwen) and will be slow/fail without `scripts/download_models.py` having run. Pure-logic tests (`test_angles`, `test_phases`, `test_rules_squat`, `test_attention`, `test_types`) are fast and have no model dependency — prefer these when iterating on `analysis/`.
- `test_llm_smoke.py` loads ~4 GB of weights; only run when actually touching `feedback/`.

## Development workflow

### Jupyter for per-stage debugging

Treat the pipeline as five separable stages and prefer iterating on each one in a notebook before wiring it back into `app.run()`. The natural cut points match the module boundaries:

1. **Capture** (`workout_ai.capture`) — read a few frames, save as PNGs, sanity-check shape/FPS.
2. **2D pose** (`workout_ai.pose2d`) — feed a still image, scatter the 17 keypoints with `matplotlib`, overlay the COCO skeleton from `render.SKELETON`.
3. **3D lift** (`workout_ai.pose3d`) — build a 27-frame window manually (or replay from a saved `.npy`), call `Pose3D.infer`, plot the result with `mpl_toolkits.mplot3d`.
4. **Analysis** (`workout_ai.analysis.*`) — these are pure functions; feed synthetic `PoseFrame`s and inspect `RepAnalysis` / FSM transitions step-by-step.
5. **LLM** (`workout_ai.feedback.llm`) — call `ThaiCoachLLM.generate(rep)` directly with a hand-built `RepAnalysis`; useful for prompt tweaking without running the camera loop.

Notebook conventions:

- Launch via `uv run jupyter lab` (add `jupyter` with `uv add --dev jupyter` if it isn't present yet) so the kernel sees the project's `.venv`.
- Keep exploratory notebooks under `notebooks/` (gitignored if they balloon with model outputs); promote stable cells into `tests/` as pure-logic tests once the behavior is settled.
- For attention/heatmap visualizations use `matplotlib.imshow` with `cmap="jet"` and alpha-blend over the frame — mirrors `Renderer._overlay_attention` so visuals stay consistent with the live app.
- For the 3D rig, draw bones with `SKELETON` from `render.py` so the topology matches what the user sees in the panel.
- Heavy models survive across cells — don't re-instantiate `Pose2D` / `Pose3D` / `ThaiCoachLLM` unless you actually need to; reuse the existing object to keep MPS memory pressure down.

### Tools available beyond the standard set

- **Python LSP** — use it to jump to definitions, find references, and rename symbols when navigating the codebase. Preferred over `grep` for symbol-level questions ("where is `coco17_to_h36m17` called from?", "what implements `on_rep_complete`?").
- **context7 MCP plugin** — fetch current docs for `mlx-vlm`, `rtmlib`, `torch` (MPS specifics), `mediapipe`/`opencv-python`, `huggingface_hub`, FastAPI/Starlette, WebRTC libs, etc. Use whenever an API surface might have drifted from training data, especially before writing integration code against `mlx-vlm` or MotionBERT internals.

## Long-term direction: streaming server for mobile client

The end-state target is **not** a desktop OpenCV app. The current `app.run()` is a local-debug harness; the production shape is a server that:

- Accepts a streaming video feed from a mobile client (WebRTC / WebSocket / HTTP-multipart — undecided; flag this when relevant).
- Runs the same per-frame pipeline (2D → 3D → FSM → score → LLM-on-rep-boundary) on the server.
- Streams back, per frame or per event:
  - 3D rig keypoints (the `(17, 3)` array currently passed to `Renderer._draw_rig_3d`) for the client to render.
  - Phase + live score updates.
  - Thai coaching text generated on rep completion.

Implications when making changes today:

- Keep the pipeline stages independent of OpenCV's GUI (`cv2.imshow` / `waitKey`) — those calls live only in `app.run()` and `Renderer`. New analysis logic must not assume a display is attached.
- Avoid coupling rep state or score state to the renderer; the renderer should remain a pure consumer.
- Frame I/O abstraction is currently `WebcamCapture`; a future `StreamCapture` (from a socket / WebRTC track) should be drop-in compatible — preserve `start()` / `read_latest(timeout)` / `stop()` semantics on any new capture source.
- `LLMWorker`'s drop-stale design is correct for the streaming case too; preserve it rather than queueing.

## Repo conventions

- Acceptance targets from the README (kept for orientation, not yet automated): 2D overlay ≥25 FPS, 3D rig ≥5 Hz, ±1 rep over a 10-rep set, Thai feedback within 3 s of rep completion.
- Thai feedback wording (`prompt_th.py`, `rules_squat.py` `detail_th`) is intentionally short and polite — match the existing tone.
- Design spec: `docs/superpowers/specs/2026-05-18-pose-form-coach-design.md`. Implementation plan: `docs/superpowers/plans/2026-05-18-pose-form-coach.md`.
