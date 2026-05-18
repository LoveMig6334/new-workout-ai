import numpy as np
import pytest
from workout_ai.analysis.angles import (
    angle_between_3_points,
    knee_angles,
    torso_lean_deg,
    hip_below_knee,
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
