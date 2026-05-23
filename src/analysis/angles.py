import numpy as np

NOSE = 0
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


def head_lateral_tilt_2d(
    kps_2d: np.ndarray,
    scores: np.ndarray | None = None,
    score_threshold: float = 0.3,
) -> float:
    """Signed head-lateral-tilt angle in degrees, computed in image space from
    nose + shoulders only — robust to lower-body occlusion (desk-camera, seated).

    Sign convention matches `angles_3d.head_lateral_tilt_3d`:
    positive = head tilted toward the body's R_shoulder side (body_lateral = R - L);
    negative = tilted toward the L_shoulder side. Independent of camera mirroring
    because the lateral reference is the body's own shoulder vector.

    Up reference is image-vertical (the spec accepts camera-roll dependence
    for the phone-on-tripod target). Returns NaN if nose / either shoulder has
    confidence below `score_threshold`, or shoulders are coincident.
    """
    if scores is not None:
        if (
            scores[NOSE] < score_threshold
            or scores[L_SHOULDER] < score_threshold
            or scores[R_SHOULDER] < score_threshold
        ):
            return float("nan")

    nose = kps_2d[NOSE]
    l_sh = kps_2d[L_SHOULDER]
    r_sh = kps_2d[R_SHOULDER]
    mid_sh = (l_sh + r_sh) / 2.0

    shoulder_dx = float(r_sh[0] - l_sh[0])
    if abs(shoulder_dx) < 1e-6:
        return float("nan")

    head_dx = float(nose[0] - mid_sh[0])
    head_dy = float(nose[1] - mid_sh[1])  # image y grows down

    # Project head vector onto body axes. Lateral direction is sign(shoulder_dx)
    # along image-x (mirror-independent). Up direction is image-up = (0, -1).
    lat_component = head_dx * (1.0 if shoulder_dx > 0 else -1.0)
    up_component = -head_dy

    if lat_component == 0.0 and up_component == 0.0:
        return float("nan")

    return float(np.degrees(np.arctan2(lat_component, up_component)))
