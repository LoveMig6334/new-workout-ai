"""Product-relevant evaluation: does the student preserve the *measurements*
the scoring depends on? Compares student vs teacher (or vs GT) keypoints through
analysis.angles, plus PCK on the upper-body subset the rules use.
"""
import numpy as np
from analysis.angles import (
    head_lateral_tilt_2d, craniovertebral_angle_2d,
    NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
)

UPPER_BODY = [NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]


def angle_agreement(kps_a, scores_a, kps_b, scores_b) -> dict:
    """Absolute difference in each scoring angle between two keypoint sets.
    NaN-safe: a metric that is NaN for either set is reported as NaN."""
    def diff(fn):
        va, vb = fn(kps_a, scores_a), fn(kps_b, scores_b)
        if np.isnan(va) or np.isnan(vb):
            return float("nan")
        return abs(va - vb)
    return {
        "head_lateral_tilt_2d": diff(head_lateral_tilt_2d),
        "craniovertebral_angle_2d": diff(craniovertebral_angle_2d),
    }


def upper_body_pck(kps_pred, kps_gt, threshold_px: float = 10.0) -> float:
    """Fraction of upper-body keypoints within threshold_px of GT."""
    sel = np.array(UPPER_BODY)
    d = np.linalg.norm(kps_pred[sel] - kps_gt[sel], axis=1)
    return float((d <= threshold_px).mean())
