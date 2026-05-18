# Workout AI — Real-Time Squat Form Coach (Thai)

Real-time webcam-based squat form coach for macOS Apple Silicon.

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
| 2D pose | RTMPose-l | rtmlib (ONNX) |
| 3D lift | MotionBERT-Lite | PyTorch on MPS |
| Form analysis | Joint angles + phase FSM + attention | pure Python |
| Thai feedback | Qwen3.5-4B (mxfp4) | mlx-vlm |

## Acceptance criteria

- [ ] Skeleton overlay at ≥ 25 FPS on M2 or better.
- [ ] 3D rig updates at ≥ 5 Hz.
- [ ] Squat reps detected within ±1 over a 10-rep set.
- [ ] 0–100 form score shown per rep + running average.
- [ ] 2–3 sentence Thai feedback within 3 s of rep completion.
- [ ] Models live in `./models/` and load from disk on subsequent runs.
