"""RTMPose-m teacher: produce SimCC logits for a preprocessed input crop.

Reuses the project's Pose2D plumbing only for the rtmlib pose model + its simcc
capture hook. We call the underlying ONNX session directly on our own
already-warped crop so the teacher and student see identical inputs.
"""
import numpy as np
from pose2d import Pose2D  # from src/, on the pytest/app path


class Teacher:
    def __init__(self):
        self._p = Pose2D(mode="balanced", accelerator="coreml")
        self._sess = self._p._body.pose_model.session
        self._inp_name = self._sess.get_inputs()[0].name

    def infer_simcc(self, input_chw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """input_chw: (3, H, W) normalized. Returns (sx (K,Xbins), sy (K,Ybins))."""
        batch = input_chw[None].astype(np.float32)  # (1, 3, H, W)
        outs = self._sess.run(None, {self._inp_name: batch})
        sx, sy = outs[0], outs[1]  # (1, K, Xbins), (1, K, Ybins)
        return sx[0].astype(np.float32), sy[0].astype(np.float32)
