import math
import numpy as np
import pytest
from analysis.angles import (
    angle_between_3_points,
    head_lateral_tilt_2d,
    knee_angles,
    torso_lean_deg,
    hip_below_knee,
    knee_valgus_ratio,
)


def _shoulders_and_nose(nose_dx: float, mirrored: bool = False):
    """Build a (17, 2) keypoint array with shoulders 100 px apart at y=200 and
    nose 100 px above mid-shoulders, shifted laterally by `nose_dx`.

    `mirrored=True` swaps L/R shoulder x-coords to simulate a selfie-mirrored
    camera, where R_shoulder ends up image-left of L_shoulder.
    """
    kps = np.zeros((17, 2), dtype=np.float32)
    if mirrored:
        kps[5] = (200.0, 200.0)  # L_shoulder on image-right
        kps[6] = (100.0, 200.0)  # R_shoulder on image-left
    else:
        kps[5] = (100.0, 200.0)
        kps[6] = (200.0, 200.0)
    kps[0] = (150.0 + nose_dx, 100.0)
    scores = np.ones(17, dtype=np.float32)
    return kps, scores


def test_angle_180_degrees_straight():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([2.0, 0.0])
    assert angle_between_3_points(a, b, c) == pytest.approx(180.0, abs=0.5)


def test_angle_90_degrees():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([1.0, 1.0])
    assert angle_between_3_points(a, b, c) == pytest.approx(90.0, abs=0.5)


def test_knee_angles_standing():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 100)
    kps[13] = (100, 200)
    kps[15] = (100, 300)
    kps[12] = (150, 100)
    kps[14] = (150, 200)
    kps[16] = (150, 300)
    left, right = knee_angles(kps)
    assert left == pytest.approx(180.0, abs=1.0)
    assert right == pytest.approx(180.0, abs=1.0)


def test_knee_angles_squatting():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (50, 200)
    kps[13] = (100, 200)
    kps[15] = (100, 300)
    kps[12] = kps[11]
    kps[14] = kps[13]
    kps[16] = kps[15]
    left, right = knee_angles(kps)
    assert left == pytest.approx(90.0, abs=2.0)


def test_torso_lean_upright():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[5] = (100, 100)
    kps[6] = (150, 100)
    kps[11] = (100, 200)
    kps[12] = (150, 200)
    assert torso_lean_deg(kps) == pytest.approx(0.0, abs=1.0)


def test_torso_lean_45deg_forward():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[5] = (200, 100)
    kps[6] = (250, 100)
    kps[11] = (100, 200)
    kps[12] = (150, 200)
    assert torso_lean_deg(kps) == pytest.approx(45.0, abs=2.0)


def test_hip_below_knee_true():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 250)
    kps[12] = (150, 250)
    kps[13] = (100, 200)
    kps[14] = (150, 200)
    assert hip_below_knee(kps) is True


def test_hip_below_knee_false():
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 100)
    kps[12] = (150, 100)
    kps[13] = (100, 200)
    kps[14] = (150, 200)
    assert hip_below_knee(kps) is False


def test_knee_valgus_ratio_neutral_stance():
    # Knees directly above ankles -> zero valgus on both sides
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 200)  # L hip
    kps[12] = (200, 200)  # R hip (hip_width = 100)
    kps[13] = (100, 250)  # L knee directly above L ankle
    kps[14] = (200, 250)  # R knee directly above R ankle
    kps[15] = (100, 300)  # L ankle
    kps[16] = (200, 300)  # R ankle
    left, right = knee_valgus_ratio(kps)
    assert left == pytest.approx(0.0, abs=0.001)
    assert right == pytest.approx(0.0, abs=0.001)


def test_knee_valgus_ratio_caved_inward():
    # Both knees shifted toward each other (caved in). Should produce positive values on both sides.
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, 200)  # L hip
    kps[12] = (200, 200)  # R hip (hip_width = 100)
    kps[13] = (130, 250)  # L knee shifted right by 30 (inward)
    kps[14] = (170, 250)  # R knee shifted left by 30 (inward)
    kps[15] = (100, 300)  # L ankle
    kps[16] = (200, 300)  # R ankle
    left, right = knee_valgus_ratio(kps)
    assert left == pytest.approx(0.3, abs=0.01)  # (130-100)/100 = 0.3
    assert right == pytest.approx(0.3, abs=0.01)  # (200-170)/100 = 0.3


# ---- head_lateral_tilt_2d ----


def test_head_tilt_2d_upright_is_zero():
    kps, scores = _shoulders_and_nose(nose_dx=0.0)
    assert head_lateral_tilt_2d(kps, scores) == pytest.approx(0.0, abs=0.5)


def test_head_tilt_2d_left_lean_is_negative():
    """Body-lateral = R - L = +x. Nose shifted to -x leans toward L_shoulder side."""
    kps, scores = _shoulders_and_nose(nose_dx=-30.0)
    assert head_lateral_tilt_2d(kps, scores) < -10.0


def test_head_tilt_2d_right_lean_is_positive():
    kps, scores = _shoulders_and_nose(nose_dx=+30.0)
    assert head_lateral_tilt_2d(kps, scores) > 10.0


def test_head_tilt_2d_45_degrees():
    """Nose 100 px above mid-shoulders and 100 px to image-right → atan2(100, 100) = 45°.
    With L on left and R on right (body_lateral = +x), this is a +45° tilt."""
    kps, scores = _shoulders_and_nose(nose_dx=+100.0)
    assert head_lateral_tilt_2d(kps, scores) == pytest.approx(45.0, abs=0.5)


def test_head_tilt_2d_mirror_invariant():
    """Selfie-mirrored camera flips R/L shoulder x but the user's tilt-direction
    label (left/right of the body) must stay the same. A real left-side tilt
    in a mirrored view: nose drifts toward image-LEFT, and R_shoulder is also
    on image-left, so body_lateral = R - L is negative → tilt stays negative."""
    # Non-mirrored: nose at -x → body_lateral = +x → negative angle.
    kps_a, scores_a = _shoulders_and_nose(nose_dx=-30.0, mirrored=False)
    angle_a = head_lateral_tilt_2d(kps_a, scores_a)

    # Mirrored selfie of the SAME physical pose: nose at +x (mirror flips it),
    # shoulders swapped so R is at image-left. body_lateral = R - L = -x.
    # The sign convention should still report left-side tilt as negative.
    kps_b, scores_b = _shoulders_and_nose(nose_dx=+30.0, mirrored=True)
    angle_b = head_lateral_tilt_2d(kps_b, scores_b)

    assert angle_a < 0
    assert angle_b < 0
    assert angle_a == pytest.approx(angle_b, abs=0.5)


def test_head_tilt_2d_nan_on_low_nose_score():
    kps, scores = _shoulders_and_nose(nose_dx=-30.0)
    scores[0] = 0.05
    assert math.isnan(head_lateral_tilt_2d(kps, scores))


def test_head_tilt_2d_nan_on_low_shoulder_score():
    kps, scores = _shoulders_and_nose(nose_dx=-30.0)
    scores[5] = 0.05  # L_shoulder
    assert math.isnan(head_lateral_tilt_2d(kps, scores))


def test_head_tilt_2d_nan_on_coincident_shoulders():
    """If RTMPose puts both shoulders at the same x, there is no usable lateral
    reference — must NaN out rather than divide by zero."""
    kps, scores = _shoulders_and_nose(nose_dx=-30.0)
    kps[6] = kps[5]  # R_shoulder collapsed onto L_shoulder
    assert math.isnan(head_lateral_tilt_2d(kps, scores))


def test_head_tilt_2d_works_without_scores():
    """`scores=None` skips gating — used when the caller already validated input."""
    kps, _ = _shoulders_and_nose(nose_dx=-30.0)
    assert head_lateral_tilt_2d(kps, scores=None) < 0
