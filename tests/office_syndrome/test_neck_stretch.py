import math
import numpy as np

from analysis.types import PoseFrame
from exercises.neck_stretch import NeckStretchLeft


# Image-plane fixture: nose above mid-shoulders, shoulders horizontal. The
# `head_dx` parameter shifts the nose laterally; head_dx < 0 = head tilted toward
# L_shoulder (the body's -lateral direction), matching the NeckStretchLeft target.
def _kps2d_with_head_shifted(head_dx: float) -> tuple[np.ndarray, np.ndarray]:
    kps = np.zeros((17, 2), dtype=np.float32)
    # Shoulders 100 px apart, at image-y = 200; L is at x=100 (image-left), R at x=200.
    kps[5] = (100.0, 200.0)  # L_shoulder
    kps[6] = (200.0, 200.0)  # R_shoulder
    # Nose 100 px above mid-shoulders (smaller y = up in image coords).
    kps[0] = (150.0 + head_dx, 100.0)
    scores = np.ones(17, dtype=np.float32)
    return kps, scores


def _make_frame(kps_2d: np.ndarray, scores: np.ndarray) -> PoseFrame:
    return PoseFrame(
        timestamp=0.0,
        keypoints_2d=kps_2d,
        scores=scores,
        frame_shape=(720, 1280),
        keypoints_3d=None,
    )


def test_metadata():
    ex = NeckStretchLeft()
    assert ex.name == "neck_stretch_left"
    assert ex.target.side == "left"
    assert ex.target.hold_seconds == 20.0
    assert len(ex.target.joints) == 1
    assert ex.target.joints[0].name == "head_lateral_tilt"


def test_measure_returns_declared_joints():
    ex = NeckStretchLeft()
    kps, scores = _kps2d_with_head_shifted(-30.0)
    measured = ex.measure(_make_frame(kps, scores))
    assert set(measured.keys()) == {"head_lateral_tilt"}


def test_measure_nan_when_nose_confidence_low():
    """If RTMPose can't see the nose, the measurement must surface as NaN so the
    FSM stays IDLE rather than acting on garbage."""
    ex = NeckStretchLeft()
    kps, scores = _kps2d_with_head_shifted(-30.0)
    scores[0] = 0.05  # nose below threshold
    measured = ex.measure(_make_frame(kps, scores))
    assert math.isnan(measured["head_lateral_tilt"])


def test_measure_nan_when_shoulder_confidence_low():
    ex = NeckStretchLeft()
    kps, scores = _kps2d_with_head_shifted(-30.0)
    scores[5] = 0.05  # L_shoulder below threshold
    measured = ex.measure(_make_frame(kps, scores))
    assert math.isnan(measured["head_lateral_tilt"])


def test_prompt_template_renders_summary_without_keyerror():
    ex = NeckStretchLeft()
    rendered = ex.prompt.summary.format(
        exercise_th=ex.display_th,
        score=87,
        duration=50,
        precision=25,
        stability=12,
        violations="(none)",
    )
    assert "87" in rendered


def test_prompt_template_renders_live_without_keyerror():
    ex = NeckStretchLeft()
    rendered = ex.prompt.live.format(
        exercise_th=ex.display_th,
        state="holding",
        progress_pct=60,
        violations="(none)",
    )
    assert "60" in rendered


def test_measure_sign_for_left_tilt_is_negative():
    """Catch a regression if the head-tilt sign convention ever flips.

    Fixture has R_shoulder.x > L_shoulder.x (body_lateral = +x). A nose shifted
    to -x is leaning toward the body's L_shoulder side, which must produce a
    NEGATIVE tilt to match the NeckStretchLeft target of -35°.
    """
    ex = NeckStretchLeft()
    kps, scores = _kps2d_with_head_shifted(-30.0)
    result = ex.measure(_make_frame(kps, scores))
    assert result["head_lateral_tilt"] < 0, (
        f"expected negative tilt for left-leaning head, got {result['head_lateral_tilt']}"
    )
