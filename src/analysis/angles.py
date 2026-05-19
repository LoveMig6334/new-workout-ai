import numpy as np

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def angle_between_3_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees, B is the vertex."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def knee_angles(kps: np.ndarray) -> tuple[float, float]:
    left = angle_between_3_points(kps[L_HIP], kps[L_KNEE], kps[L_ANKLE])
    right = angle_between_3_points(kps[R_HIP], kps[R_KNEE], kps[R_ANKLE])
    return left, right


def torso_lean_deg(kps: np.ndarray) -> float:
    """Angle between the vector (mid-hip -> mid-shoulder) and vertical (up)."""
    mid_shoulder = (kps[L_SHOULDER] + kps[R_SHOULDER]) / 2.0
    mid_hip = (kps[L_HIP] + kps[R_HIP]) / 2.0
    v = mid_shoulder - mid_hip
    vertical = np.array([0.0, -1.0])
    cos = np.dot(v, vertical) / (np.linalg.norm(v) + 1e-9)
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def hip_below_knee(kps: np.ndarray) -> bool:
    """In image coords (y down), hip below knee means hip_y > knee_y."""
    mid_hip_y = (kps[L_HIP, 1] + kps[R_HIP, 1]) / 2.0
    mid_knee_y = (kps[L_KNEE, 1] + kps[R_KNEE, 1]) / 2.0
    return bool(mid_hip_y > mid_knee_y)


def knee_valgus_ratio(kps: np.ndarray) -> tuple[float, float]:
    """Signed normalized distance: (knee_x - ankle_x) / hip_width.
    Positive = knee inward of ankle on that side. Returns (left, right).
    Used as a valgus signal; abs value > 0.15 considered valgus."""
    hip_width = abs(kps[R_HIP, 0] - kps[L_HIP, 0]) + 1e-6
    l = (kps[L_KNEE, 0] - kps[L_ANKLE, 0]) / hip_width
    r = (kps[R_ANKLE, 0] - kps[R_KNEE, 0]) / hip_width
    return float(l), float(r)
