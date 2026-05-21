# Workout AI — Real-Time Form Coach (Thai)

Real-time webcam-based form coach for macOS Apple Silicon. Currently supports squats; office-syndrome stretches (neck, shoulder, chest, hands, neck flexion) are designed and being added next via a plug-in exercise architecture.

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

- `q` — quit
- `a` — toggle model-attention overlay

## Stack

| Layer | Model | Framework |
|---|---|---|
| 2D pose | YOLOX-tiny + RTMPose-s | rtmlib (ONNX, CPU) |
| 3D lift | MotionBERT-Lite | PyTorch on MPS |
| Form analysis | Joint angles + phase FSM + attention | pure Python |
| Thai feedback | Qwen3.5-4B (mxfp4) | mlx-vlm |

## Acceptance criteria

### Squat (current)

- [ ] Skeleton overlay at ≥ 25 FPS on M2 or better.
- [ ] 3D rig updates at ≥ 5 Hz.
- [ ] Squat reps detected within ±1 over a 10-rep set.
- [ ] 0–100 form score shown per rep + running average.
- [ ] 2–3 sentence Thai feedback within 3 s of rep completion.
- [ ] Models live in `./models/` and load from disk on subsequent runs.

### Office syndrome stretches (designed, not yet implemented)

Six exercises — neck, shoulder, chest-and-shoulder, front-hand, back-hand, neck-flexion — exposed as timed-hold sessions with a selector at startup. Design spec: [`docs/superpowers/specs/2026-05-22-office-syndrome-stretches-design.md`](docs/superpowers/specs/2026-05-22-office-syndrome-stretches-design.md).
