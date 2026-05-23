# Workout AI — Real-Time Form Coach (Thai)

Real-time webcam-based form coach for macOS Apple Silicon. The launcher opens an exercise selector and runs a hold-based session for the chosen exercise. **Currently registered**: `neck_stretch_left` — a 20-second left-neck lateral-tilt hold (vertical slice of the office-syndrome plug-in architecture). **Coming next** (content-only): right-neck, both shoulders, chest-and-shoulder, both hands, neck-flexion. **Squat coaching code** is preserved in the codebase but is not currently wired to the launcher; see `CLAUDE.md` for how to re-enable it.

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
- **During a hold session**: `q` quits the session. The session also ends automatically after the 20s hold completes (Thai summary shown for ~3s, then back to the selector).

## Stack

| Layer | Model | Framework |
|---|---|---|
| 2D pose | YOLOX-tiny + RTMPose-s | rtmlib (ONNX, CPU) |
| 3D lift | MotionBERT-Lite | PyTorch on MPS |
| Form analysis | Joint angles + phase FSM + attention | pure Python |
| Thai feedback | Qwen3.5-4B (mxfp4) | mlx-vlm |

## Acceptance criteria

Shared pipeline (both modes):

- [ ] Skeleton overlay at ≥ 25 FPS on M2 or better.
- [ ] 3D rig updates at ≥ 5 Hz.
- [ ] Models live in `./models/` and load from disk on subsequent runs.

### Stretches (current launcher mode)

- [x] Selector at startup lists registered exercises; number key picks one.
- [x] Hold FSM tracks `IDLE / ENTERING / HOLDING / DRIFTED / COMPLETE` with pause-on-drift timing and a stability penalty on drift count.
- [x] 100-pt score on completion (50 duration / 30 precision / 20 stability) + Thai summary from the on-device VLM.
- [x] Live Thai nudges during the hold (throttled to ~1 every 3–4 s by Qwen latency).
- [ ] Target angles for `neck_stretch_left` calibrated from a personal reference photo (`_NECK_TILT_TARGET_LEFT_DEG` is a placeholder pending Task 13 of the implementation plan).
- [ ] Remaining 9 exercises registered (right-neck, shoulders, chest, hands, neck-flexion).

### Squat (preserved code, not currently exposed)

The original `SquatFSM` + `rules_squat.score_rep` + `RepAnalysis` flow is intact in the codebase. The Thai LLM still handles `RepAnalysis` payloads via the dispatch in `ThaiCoachLLM.generate`. The acceptance bar from the original spec — ±1 rep over a 10-rep set, 2–3 sentence Thai feedback within 3 s of rep completion — is unchanged on the code path, but no entry point currently routes to it.

Design specs:

- [Office syndrome stretches](docs/superpowers/specs/2026-05-22-office-syndrome-stretches-design.md) — current launcher mode.
- [3D scoring upgrade](docs/superpowers/specs/2026-05-19-3d-scoring-design.md).
- [Original squat coach](docs/superpowers/specs/2026-05-18-pose-form-coach-design.md).
