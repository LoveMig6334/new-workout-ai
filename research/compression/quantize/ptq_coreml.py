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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.CHECKPOINT_DIR / "student_final.pt"))
    ap.add_argument("--out", default=str(config.MODELS_DIR / "compression" / "coreml"))
    args = ap.parse_args()
    print(export(args.ckpt, args.out))
