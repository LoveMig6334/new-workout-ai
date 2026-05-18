import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("RTMLIB_CACHE", str(PROJECT_ROOT / "models" / "rtmlib_cache"))


class Pose2D:
    """Wraps rtmlib's RTMPose-l. Single-person inference: returns the highest-score person."""

    def __init__(self, device: str = "cpu"):
        from rtmlib import Body
        self._body = Body(mode="performance", to_openpose=False, backend="onnxruntime", device=device)

    def infer(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (keypoints (17,2) float32, scores (17,) float32) for the most prominent person."""
        keypoints, scores = self._body(image_bgr)
        if len(keypoints) == 0:
            return (
                np.zeros((17, 2), dtype=np.float32),
                np.zeros((17,), dtype=np.float32),
            )
        idx = np.argmax(scores.sum(axis=1))
        return keypoints[idx].astype(np.float32), scores[idx].astype(np.float32)
