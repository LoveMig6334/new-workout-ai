import numpy as np
from analysis.rules_squat import score_rep
from analysis.types import PoseFrame


def make_pose_frame(
    knee_angle: float,
    hip_y: float,
    knee_y: float,
    valgus: float = 0.0,
    lean: float = 30.0,
) -> PoseFrame:
    """Manually construct a PoseFrame with hand-tuned keypoints for a given posture."""
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100, hip_y)
    kps[12] = (150, hip_y)
    kps[13] = (100 + valgus * 50, knee_y)
    kps[14] = (150 - valgus * 50, knee_y)
    kps[15] = (100, knee_y + 100)
    kps[16] = (150, knee_y + 100)
    import math

    dx = math.tan(math.radians(lean)) * 100
    kps[5] = (100 + dx, hip_y - 100)
    kps[6] = (150 + dx, hip_y - 100)
    scores = np.ones((17,), dtype=np.float32)
    return PoseFrame(
        timestamp=1.0, keypoints_2d=kps, scores=scores, frame_shape=(480, 640)
    )


def test_perfect_rep_high_score():
    bottom = make_pose_frame(
        knee_angle=85.0, hip_y=260, knee_y=240, valgus=0.0, lean=35.0
    )
    result = score_rep(bottom_frame=bottom, descent_ms=1200, ascent_ms=1000)
    assert result.score >= 90
    assert result.violations == []


def test_shallow_squat_loses_depth_points():
    bottom = make_pose_frame(knee_angle=120.0, hip_y=180, knee_y=240, lean=30.0)
    result = score_rep(bottom_frame=bottom, descent_ms=900, ascent_ms=900)
    assert result.components["depth"] == 0
    assert any(v.name == "shallow_depth" for v in result.violations)


def test_knee_valgus_detected():
    bottom = make_pose_frame(
        knee_angle=85.0, hip_y=260, knee_y=240, valgus=0.4, lean=30.0
    )
    result = score_rep(bottom_frame=bottom, descent_ms=1000, ascent_ms=1000)
    assert any(v.name == "knee_valgus" for v in result.violations)
    assert result.components["valgus"] < 25


def test_excessive_forward_lean():
    bottom = make_pose_frame(knee_angle=85.0, hip_y=260, knee_y=240, lean=70.0)
    result = score_rep(bottom_frame=bottom, descent_ms=1000, ascent_ms=1000)
    assert any(v.name == "excessive_forward_lean" for v in result.violations)
    assert result.components["torso"] < 20


def test_tempo_penalty_when_ascent_longer():
    bottom = make_pose_frame(knee_angle=85.0, hip_y=260, knee_y=240, lean=30.0)
    result = score_rep(bottom_frame=bottom, descent_ms=500, ascent_ms=1500)
    assert result.components["tempo"] < 10
