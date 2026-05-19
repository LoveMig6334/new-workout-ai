import numpy as np

from analysis.angles_3d import body_frame_axes, knee_flexion_3d, torso_lean_3d


def _canonical_standing_pose() -> np.ndarray:
    """Pelvis at origin, thorax 1 unit up (image-y down → up = -y),
    hips along ±x. Other H36M joints zero-initialized (unused by this fn)."""
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[0] = (0.0, 0.0, 0.0)       # pelvis
    kps[1] = (0.2, 0.0, 0.0)       # R_hip
    kps[4] = (-0.2, 0.0, 0.0)      # L_hip
    kps[8] = (0.0, -1.0, 0.0)      # thorax
    return kps


def test_body_frame_axes_canonical_pose_is_orthonormal():
    kps = _canonical_standing_pose()
    up, lat, fwd = body_frame_axes(kps)
    np.testing.assert_allclose(up, [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(lat, [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(fwd, [0.0, 0.0, 1.0], atol=1e-6)


def test_body_frame_axes_are_unit_vectors():
    kps = _canonical_standing_pose()
    for v in body_frame_axes(kps):
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_body_frame_axes_lateral_is_orthogonal_to_up_even_when_hipline_tilted():
    kps = _canonical_standing_pose()
    # Tilt the hip line so it isn't already perpendicular to body_up.
    kps[1] = (0.2, -0.1, 0.0)
    kps[4] = (-0.2, 0.1, 0.0)
    up, lat, _ = body_frame_axes(kps)
    assert abs(np.dot(up, lat)) < 1e-6


def _both_legs_at_flexion(theta_deg: float) -> np.ndarray:
    """Symmetric pose with both knees at the given flexion (degrees).

    Geometry: thigh points straight up (knee→hip = (0, -1, 0) direction),
    shin rotated about the lateral axis so the angle between knee→hip and
    knee→ankle equals theta. theta=180° → straight leg; theta=90° → shin
    perpendicular to thigh."""
    import math

    kps = _canonical_standing_pose()
    bone = 0.3
    theta = math.radians(theta_deg)
    # knee→ankle direction in (y, z) sagittal plane:
    # at theta=180°, points (0, +1, 0) (down in image, opposite of knee→hip).
    # at theta=90°,  points (0, 0, +1) (forward).
    # general: (0, -cos(theta), sin(theta))
    shin_dir = np.array([0.0, -math.cos(theta), math.sin(theta)], dtype=np.float32)

    for hip_x in (0.2, -0.2):
        hip = np.array([hip_x, -bone, 0.0], dtype=np.float32)
        knee = np.array([hip_x, 0.0, 0.0], dtype=np.float32)
        ankle = knee + bone * shin_dir
        if hip_x > 0:
            kps[1] = hip  # R_HIP
            kps[2] = knee  # R_KNEE
            kps[3] = ankle  # R_ANKLE
        else:
            kps[4] = hip  # L_HIP
            kps[5] = knee  # L_KNEE
            kps[6] = ankle  # L_ANKLE
    return kps


def test_knee_flexion_3d_straight_leg_is_180():
    kps = _both_legs_at_flexion(180.0)
    left, right = knee_flexion_3d(kps)
    assert abs(left - 180.0) < 0.5
    assert abs(right - 180.0) < 0.5


def test_knee_flexion_3d_right_angle_is_90():
    kps = _both_legs_at_flexion(90.0)
    left, right = knee_flexion_3d(kps)
    assert abs(left - 90.0) < 0.5
    assert abs(right - 90.0) < 0.5


def test_knee_flexion_3d_degenerate_returns_nan():
    kps = _canonical_standing_pose()
    # All knee-related joints at origin — zero-length bones.
    left, right = knee_flexion_3d(kps)
    assert np.isnan(left) or np.isnan(right)


def test_torso_lean_3d_perfectly_upright_is_zero():
    kps = _canonical_standing_pose()
    assert abs(torso_lean_3d(kps)) < 0.5


def test_torso_lean_3d_thirty_degrees_forward():
    import math

    kps = _canonical_standing_pose()
    theta = math.radians(30.0)
    # Lean thorax 30° forward (sagittal +z), unit-length torso.
    kps[8] = (0.0, -math.cos(theta), math.sin(theta))
    assert abs(torso_lean_3d(kps) - 30.0) < 0.5


def test_torso_lean_3d_sixty_degrees_forward():
    import math

    kps = _canonical_standing_pose()
    theta = math.radians(60.0)
    kps[8] = (0.0, -math.cos(theta), math.sin(theta))
    assert abs(torso_lean_3d(kps) - 60.0) < 0.5
