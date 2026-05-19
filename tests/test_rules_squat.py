import math

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


def _make_pose_at_knee_angle(knee_deg: float, lean_deg: float = 35.0) -> PoseFrame:
    """Construct a symmetric pose with the given mean knee flexion (degrees)."""
    knee_y = 240.0
    bone = 100.0
    theta = math.radians(knee_deg)
    hip_dx = bone * math.sin(theta)
    hip_dy = bone * math.cos(theta)
    lean = math.radians(lean_deg)
    sh_dx = bone * math.sin(lean)
    sh_dy = -bone * math.cos(lean)
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[11] = (100 + hip_dx, knee_y + hip_dy)
    kps[12] = (150 + hip_dx, knee_y + hip_dy)
    kps[13] = (100, knee_y)
    kps[14] = (150, knee_y)
    kps[15] = (100, knee_y + bone)
    kps[16] = (150, knee_y + bone)
    kps[5] = (kps[11][0] + sh_dx, kps[11][1] + sh_dy)
    kps[6] = (kps[12][0] + sh_dx, kps[12][1] + sh_dy)
    return PoseFrame(
        timestamp=1.0,
        keypoints_2d=kps,
        scores=np.ones((17,), dtype=np.float32),
        frame_shape=(480, 640),
    )


def test_near_parallel_gets_partial_depth_credit():
    """Mean knee ~110° (above parallel but not standing) should give partial depth credit."""
    bottom = _make_pose_at_knee_angle(knee_deg=110.0)
    result = score_rep(bottom_frame=bottom, descent_ms=1200, ascent_ms=1000)
    # 110° → ratio = (130-110)/40 = 0.5 → depth = 15
    assert 12 <= result.components["depth"] <= 18
    shallow = next(v for v in result.violations if v.name == "shallow_depth")
    assert 0.3 < shallow.severity < 0.7


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


def test_score_rep_uses_2d_fallback_when_no_3d_present():
    """Existing 2D-only fixtures should now report metric_source='2d'."""
    bottom = make_pose_frame(
        knee_angle=85.0, hip_y=260, knee_y=240, valgus=0.0, lean=35.0
    )
    # bottom.keypoints_3d is None by construction (make_pose_frame doesn't set it).
    result = score_rep(bottom_frame=bottom, descent_ms=1200, ascent_ms=1000)
    assert result.metric_source == "2d"
