# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Real-time webcam-based form coach for macOS Apple Silicon. Streams from the webcam, runs 2D pose → 3D pose → rule-based form analysis, and produces Thai-language coaching feedback from an on-device VLM.

**Two exercise modes coexist in the codebase:**

- **Timed-hold stretches (current entry point)** — `app.run()` opens an OpenCV selector at startup, then dispatches to one registered `Exercise`. Built on `HoldFSM` (`analysis/phases.py`) + generic `rules_hold.py` scorer + `exercises/` package (one module per exercise, mostly target-angle data). Currently registered: `NeckStretchLeft` (left-neck lateral tilt, 20s timed hold). Remaining 9 entries (neck-right, shoulders, chest-and-shoulder, front-hand, back-hand, neck-flexion) are content-only additions deferred to a follow-up plan — each is a new module under `src/exercises/` + a registry entry.
- **Squat coaching (rep-based, preserved)** — original `SquatFSM` (STANDING → DESCENT → BOTTOM → ASCENT) + `analysis/rules_squat.py::score_rep`. LLM fires on rep completion. The code is intact and the tests still cover it, but **`main.py` no longer routes to it** — the squat flow is parked while the stretch slice is in active development. To re-expose squats from the entry point, register a squat-shaped `Exercise` (or add a CLI flag in `app.run`).

The selector loads heavy weights (Qwen, MotionBERT, RTMPose) once and then loops session-by-session; switching exercises doesn't trigger reloads. Spec for the stretches: `docs/superpowers/specs/2026-05-22-office-syndrome-stretches-design.md`.

## Commands

This project uses `uv` (Python 3.12) — always invoke Python via `uv run`.

```bash
uv sync --all-extras                          # install + dev deps
uv run python scripts/download_models.py      # one-time, ~6 GB into ./models/
uv run python main.py                         # opens selector → pick an exercise (number key) → hold session begins; q/Esc cancels selector, q quits a session

uv run pytest                                 # run full test suite
uv run pytest tests/test_phases.py            # single file
uv run pytest tests/test_phases.py::test_name # single test
uv run ruff check                             # lint
uv run ruff format                            # format
```

`pytest.ini_options` sets `pythonpath = ["src"]`, so tests import as `from analysis.* import ...`, `from feedback.* import ...`, `from app import ...`, etc. without an editable install.

## Architecture

Single-process pipeline driven by `src/app.py::run()`. Top-level `run()` loads pose + LLM weights once, then loops: `choose_exercise()` (selector) → `run_session(exercise)` (one hold) → repeat. `run_session` reads webcam frames, runs them through pose inference and the per-exercise hold FSM, scores in-flight + on completion, and ships scores to a background LLM thread for Thai narration. The display is the camera feed plus a right-hand panel with the 3D rig, hold-state badge, progress bar, and Thai coaching text.

Data flow per frame (hold mode — current entry point):

```
WebcamCapture (bg thread)
   → Pose2D (YOLOX-tiny + RTMPose-s via rtmlib/ONNX, CPU) kps:(17,2) + scores
       → coco17_to_h36m17 → Pose3DBuffer (27-frame window)
           → Pose3D (MotionBERT-Lite, PyTorch on MPS)      kps_3d:(17,3) — lifted every 5 frames
   → exercise.measure(PoseFrame)                           {joint_name: degrees}
       → rules_hold.score_frame(target, measured)          (in_target: bool, violations)
           → HoldFSM.update(in_target, ts)                 IDLE → ENTERING → HOLDING → COMPLETE (with DRIFTED branch)
   → live (throttled ≥ 2.5s): LLMWorker.submit(LiveSnapshot, exercise=...)
   → on COMPLETE: rules_hold.score_hold(...) → LLMWorker.submit(HoldAnalysis, exercise=...)
                  → ThaiCoachLLM.generate(payload, exercise=...) — dispatches on payload type
   → Renderer.compose(..., hold_state=state.value, hold_progress=...) → cv2.imshow
```

Data flow for the parked squat mode (code still intact, no entry-point wiring):

```
... 2D + 3D pose stages identical ...
   → SquatFSM.update(kps, ts)                              STANDING → DESCENT → BOTTOM → ASCENT → STANDING
       → on rep complete: rules_squat.score_rep(bottom_frame) → RepAnalysis
           → LLMWorker.submit(RepAnalysis)
               → ThaiCoachLLM.generate(RepAnalysis) — same dispatch, squat branch
```

### Coordinate systems & keypoint layouts

- **2D keypoints throughout `analysis/`**: COCO-17 indices (see `analysis/angles.py` for the canonical index constants). Image-pixel coords with y growing downward — that's why `hip_below_knee` checks `hip_y > knee_y`.
- **3D keypoints**: Human3.6M-17 layout. `pose3d.coco17_to_h36m17` synthesizes pelvis, spine, and thorax from COCO joints (indices marked `-1`, `-2`, `-3` in `COCO_TO_H36M`). Normalization in `_normalize_2d` divides both x and y by frame **width** (not height) — this matches MotionBERT's expected input convention; do not "fix" it.
- **Heatmaps for attention overlay**: reconstructed from RTMPose's simcc outputs via outer product (see `pose2d.infer_with_heatmaps`). This is a monkey-patch of `rtmlib`'s pose estimator; if `rtmlib` API shifts, the patch silently degrades and `heatmaps` becomes `None`.

### Phase detection and scoring

Two FSMs live in `analysis/phases.py`. They share no state and aren't polymorphic — pick the right one for the exercise shape.

**`HoldFSM` (timed-hold stretches, current entry point).** Takes a single `bool in_target` per frame (computed upstream by `rules_hold.score_frame`) and tracks the lifecycle `IDLE → ENTERING → HOLDING → COMPLETE`, with a `DRIFTED` branch off `HOLDING`. Two parameters with defaults: `stability_window_s=0.5` (gate to filter accidental fly-throughs), `drift_grace_s=0.3` (forgive single-frame jitter). Pause-on-drift semantics — `in_target_ms` advances only while `HOLDING`. Drift past grace transitions back to `ENTERING` (preserving accumulated `in_target_ms` per spec §5.2) and increments `drift_count`, which feeds the stability score. Fires `on_hold_complete(meta)` with `in_target_ms` / `drift_count` / `completed_ts`.

`analysis/rules_hold.py` is the generic, exercise-agnostic scorer:
- `score_frame(target, measured) -> (in_target, violations)`: per-frame predicate driving the FSM. NaN or missing measurement → out-of-target with severity 1.0. Deviation past tolerance ramps severity from 0 at the edge to 1.0 at 2× tolerance.
- `score_hold(...) -> HoldAnalysis`: final 100-pt rubric — duration 50 (clamped `in_target_ms / target_ms`), precision 30 (driven by worst per-joint severity), stability 20 (smooth decay on `drift_count`).

Per-exercise data lives under `src/exercises/`. Each module declares `name`, `display_th`, `target: TargetPose` (joint angle ranges + tolerances + Thai hints), `prompt: PromptTemplate` (Thai live + summary templates), and a `measure(frame) -> dict[str, float]` method. Add a new exercise: drop a new module in, register it in `exercises/__init__.py::EXERCISES`.

**`SquatFSM` (preserved squat mode, not wired to entry point).** Thresholds `STAND_THRESHOLD=160°` and `BOTTOM_THRESHOLD=100°` on the average of left/right knee angles. A rep commits on `ASCENT → STANDING`, firing `on_rep_complete(meta)` with `descent_ms` / `ascent_ms`.

`analysis/rules_squat.py::score_rep` is a pure function from a single `PoseFrame` (the bottom frame) + tempo to a `RepAnalysis`. Component budget totals 100: depth 30 / valgus 25 / torso 20 / symmetry 15 / tempo 10. Each violation also produces a Thai-language hint (`detail_th`) consumed by the LLM prompt — keep wording natural Thai when adding rules.

### LLM feedback

`feedback/llm.py::ThaiCoachLLM` wraps Qwen3.5-4B (mxfp4) via `mlx-vlm`. First `generate()` call is slow (compilation) — `app.run()` calls `warmup()` before entering the loop; preserve that order. `generate(payload, ..., exercise=None)` dispatches on payload type:
- `RepAnalysis` → squat system prompt + `build_user_prompt` (no `exercise` needed).
- `HoldAnalysis` → hold system prompt + `build_hold_summary_prompt(payload, exercise)` — `exercise=` is **required**, else `ValueError`.
- `LiveSnapshot` → hold system prompt + `build_live_prompt(payload, exercise)` — `exercise=` is **required**, else `ValueError`.

`feedback/worker.py::LLMWorker` runs in a background thread with **drop-stale** semantics: `submit(payload, **kwargs)` replaces any pending submission that hasn't been processed yet. There is no queue. Kwargs (including `exercise=`) are stored alongside the payload and forwarded to `generate()`. If submissions arrive faster than the LLM, older ones are silently dropped — by design, since stale feedback is worse than no feedback.

Hold-mode live cadence: `app.run_session` throttles submissions to ≥ 2.5 s apart (Qwen 4B mxfp4 on MPS produces a short Thai phrase in ~1–2 s, so faster submission is wasted work). Expect roughly one fresh nudge every 3–4 s during a 20 s hold.

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
- The `*_smoke.py` tests actually load the heavy models (RTMPose, MotionBERT, Qwen) and will be slow/fail without `scripts/download_models.py` having run. They use `pytest.mark.skipif` against `tests/fixtures/...` and `models/...` paths anchored from `__file__`, so they skip gracefully when assets are missing.
- Pure-logic tests are fast and have no model dependency — prefer these when iterating on `analysis/` or `exercises/`. The pure-logic set: `test_angles`, `test_angles_3d`, `test_phases`, `test_hold_fsm`, `test_rules_squat`, `test_rules_hold`, `test_attention`, `test_types`, `test_exercises_base`, `test_exercises_registry`, `test_neck_stretch`, `test_prompt_th`, `test_render`, `test_worker`.
- `test_llm_smoke.py` loads ~4 GB of weights; only run when actually touching `feedback/`. It contains both squat (`RepAnalysis`) and hold (`HoldAnalysis`) generation checks.
- `test_exercise_pipeline_smoke.py` runs the full 2D → 3D → measure → score_frame glue for `NeckStretchLeft` against a fixture image — gated on RTMPose weights + the fixture being present.

## Development workflow

### Jupyter for per-stage debugging

Treat the pipeline as five separable stages and prefer iterating on each one in a notebook before wiring it back into `app.run()`. The natural cut points match the module boundaries:

1. **Capture** (`capture`) — read a few frames, save as PNGs, sanity-check shape/FPS.
2. **2D pose** (`pose2d`) — feed a still image, scatter the 17 keypoints with `matplotlib`, overlay the COCO skeleton from `render.SKELETON`.
3. **3D lift** (`pose3d`) — build a 27-frame window manually (or replay from a saved `.npy`), call `Pose3D.infer`, plot the result with `mpl_toolkits.mplot3d`.
4. **Analysis** (`analysis.*`) — these are pure functions; feed synthetic `PoseFrame`s and inspect `RepAnalysis` / FSM transitions step-by-step.
5. **LLM** (`feedback.llm`) — call `ThaiCoachLLM.generate(rep)` directly with a hand-built `RepAnalysis`; useful for prompt tweaking without running the camera loop.

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
- Design specs:
  - Original squat coach: `docs/superpowers/specs/2026-05-18-pose-form-coach-design.md` (impl plan: `docs/superpowers/plans/2026-05-18-pose-form-coach.md`).
  - 3D scoring upgrade: `docs/superpowers/specs/2026-05-19-3d-scoring-design.md`.
  - Office syndrome stretches: `docs/superpowers/specs/2026-05-22-office-syndrome-stretches-design.md` (impl plan: `docs/superpowers/plans/2026-05-22-office-syndrome-stretches.md`).
