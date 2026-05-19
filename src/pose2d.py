import os
from pathlib import Path
from typing import Literal, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("RTMLIB_CACHE", str(PROJECT_ROOT / "models" / "rtmlib_cache"))


class Pose2D:
    """Wraps rtmlib's RTMPose. Single-person inference: returns the highest-score person.
    Optionally returns simcc-decoded heatmaps via `infer_with_heatmaps`.

    `mode="lightweight"` loads YOLOX-tiny + RTMPose-s (~55 FPS on CPU at 720p, M-series).
    `mode="balanced"` loads YOLOX-m + RTMPose-m (~12 FPS).
    `mode="performance"` loads YOLOX-x + RTMPose-x (~3 FPS) — rtmlib's "performance" means
    accuracy-first, not speed-first.

    `device="mps"` (CoreML EP) currently crashes on every YOLOX variant in rtmlib due to a
    dynamic-shape NMS op that fails when zero detections are produced. Stay on CPU until
    rtmlib swaps to RTMO or onnxruntime gets a fix.
    """

    def __init__(
        self,
        device: str = "cpu",
        mode: Literal["lightweight", "balanced", "performance"] = "lightweight",
    ):
        from rtmlib import Body

        self._body = Body(
            mode=mode, to_openpose=False, backend="onnxruntime", device=device
        )

        # rtmlib 0.0.15 exposes the pose estimator as `pose_model`
        self._pose = getattr(self._body, "pose_model", None)

        if self._pose is not None:
            # Monkey-patch inference to capture raw simcc outputs for attention overlay.
            pose_model = self._pose
            _orig_inference = pose_model.inference

            def _capturing_inference(image):
                result = _orig_inference(image)
                try:
                    # inference returns [simcc_x, simcc_y] — each shape (1, N_kpts, simcc_bins)
                    if isinstance(result, (list, tuple)) and len(result) == 2:
                        pose_model._last_simcc = result
                except Exception:
                    pass
                return result

            pose_model.inference = _capturing_inference

    def infer(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (keypoints (17,2) float32, scores (17,) float32) for the most prominent person."""
        keypoints, scores = self._body(image_bgr)
        if len(keypoints) == 0:
            return (
                np.zeros((17, 2), dtype=np.float32),
                np.zeros((17,), dtype=np.float32),
            )
        idx = int(np.argmax(scores.sum(axis=1)))
        return keypoints[idx].astype(np.float32), scores[idx].astype(np.float32)

    def infer_with_heatmaps(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Returns (keypoints, scores, heatmaps) where heatmaps is (17, H, W) reconstructed from simcc.
        If rtmlib version does not expose simcc outputs, heatmaps will be None."""
        kps, scores = self.infer(image_bgr)
        heatmaps = None
        if self._pose is not None:
            try:
                simcc_pair = getattr(self._pose, "_last_simcc", None)
                if simcc_pair is not None:
                    simcc_x, simcc_y = simcc_pair
                    # simcc_x: (1, N_kpts, W_bins), simcc_y: (1, N_kpts, H_bins)
                    sx = simcc_x[0]  # (N_kpts, W_bins)
                    sy = simcc_y[0]  # (N_kpts, H_bins)
                    hms = []
                    for k in range(sx.shape[0]):
                        # outer product: rows=H_bins, cols=W_bins
                        hm = np.outer(sy[k], sx[k])
                        hms.append(hm)
                    heatmaps = np.stack(hms, axis=0).astype(np.float32)
            except Exception:
                heatmaps = None
        return kps, scores, heatmaps
