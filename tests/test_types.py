import numpy as np
from workout_ai.analysis.types import PoseFrame, PhaseState, RepAnalysis


def test_pose_frame_construction():
    keypoints = np.zeros((17, 2), dtype=np.float32)
    scores = np.zeros((17,), dtype=np.float32)
    pf = PoseFrame(
        timestamp=1.0, keypoints_2d=keypoints, scores=scores, frame_shape=(480, 640)
    )
    assert pf.timestamp == 1.0
    assert pf.keypoints_2d.shape == (17, 2)


def test_phase_state_enum():
    assert PhaseState.STANDING.value == "standing"
    assert PhaseState.BOTTOM.value == "bottom"


def test_rep_analysis_defaults():
    ra = RepAnalysis(
        rep_index=0,
        score=85,
        components={"depth": 25, "valgus": 20, "torso": 18, "symmetry": 14, "tempo": 8},
        violations=[],
        descent_ms=900,
        ascent_ms=800,
    )
    assert ra.score == 85
    assert ra.violations == []
