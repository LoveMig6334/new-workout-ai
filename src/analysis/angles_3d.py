import numpy as np

# H36M-17 indices used in this module.
PELVIS = 0
R_HIP = 1
R_KNEE = 2
R_ANKLE = 3
L_HIP = 4
L_KNEE = 5
L_ANKLE = 6
THORAX = 8


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def body_frame_axes(
    kps_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (body_up, body_lateral, body_forward) as unit vectors in MotionBERT's frame.

    - body_up: pelvis → thorax (head-ward along the torso).
    - body_lateral: L_hip → R_hip projected orthogonal to body_up.
    - body_forward: body_up × body_lateral (sagittal axis, forward through the chest).
    """
    up = _norm(kps_3d[THORAX] - kps_3d[PELVIS])
    hip = kps_3d[R_HIP] - kps_3d[L_HIP]
    lateral_raw = hip - float(np.dot(hip, up)) * up
    lateral = _norm(lateral_raw)
    forward = np.cross(up, lateral)
    return up, lateral, forward
