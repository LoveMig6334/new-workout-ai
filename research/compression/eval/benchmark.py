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
