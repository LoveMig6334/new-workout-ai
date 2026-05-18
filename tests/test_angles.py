import numpy as np
import pytest
from workout_ai.analysis.angles import (
    angle_between_3_points,
    knee_angles,
    torso_lean_deg,
    hip_below_knee,
    knee_valgus_ratio,
)


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
