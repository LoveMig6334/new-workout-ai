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


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in degrees.

    Returns NaN if either vector has near-zero length.
    """
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def knee_flexion_3d(kps_3d: np.ndarray) -> tuple[float, float]:
    """Per-knee flexion in degrees, computed in full 3D. Returns (left, right).

    180° = straight leg, smaller values = more bent. NaN if any bone has zero length.
    """
    left = _angle_deg(
        kps_3d[L_HIP] - kps_3d[L_KNEE],
        kps_3d[L_ANKLE] - kps_3d[L_KNEE],
    )
    right = _angle_deg(
        kps_3d[R_HIP] - kps_3d[R_KNEE],
        kps_3d[R_ANKLE] - kps_3d[R_KNEE],
    )
    return left, right


# Image vertical in MotionBERT's normalized 3D frame (image-y down → up = -y).
# This equals gravity-up when the camera is upright; the spec accepts this
# camera-roll dependency for the current target (phone-on-tripod).
IMAGE_UP = np.array([0.0, -1.0, 0.0], dtype=np.float32)


def torso_lean_3d(kps_3d: np.ndarray) -> float:
    """Angle in degrees between body_up (pelvis → thorax) and image vertical."""
    torso = kps_3d[THORAX] - kps_3d[PELVIS]
    return _angle_deg(torso, IMAGE_UP)
