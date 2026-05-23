import math
import numpy as np

from analysis.types import PoseFrame
from exercises.neck_stretch import NeckStretchLeft


def _make_frame(kps_3d: np.ndarray | None) -> PoseFrame:
    return PoseFrame(
        timestamp=0.0,
        keypoints_2d=np.zeros((17, 2), dtype=np.float32),
        scores=np.ones(17, dtype=np.float32),
        frame_shape=(720, 1280),
        keypoints_3d=kps_3d,
    )


def _h36m_with_head_at(lat_offset: float) -> np.ndarray:
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[0] = (0.0, 0.0, 0.0)
    kps[1] = (0.5, 0.0, 0.0)   # R_HIP — matches the fixture convention in test_angles_3d
    kps[4] = (-0.5, 0.0, 0.0)  # L_HIP
    kps[8] = (0.0, -1.0, 0.0)
    kps[10] = (lat_offset, -2.0, 0.0)
    return kps


def test_metadata():
    ex = NeckStretchLeft()
    assert ex.name == "neck_stretch_left"
    assert ex.target.side == "left"
    assert ex.target.hold_seconds == 20.0
    assert len(ex.target.joints) == 1
    assert ex.target.joints[0].name == "head_lateral_tilt"


def test_measure_returns_declared_joints():
    ex = NeckStretchLeft()
    measured = ex.measure(_make_frame(_h36m_with_head_at(-0.5)))
    assert set(measured.keys()) == {"head_lateral_tilt"}


def test_measure_nan_when_no_3d_keypoints():
    ex = NeckStretchLeft()
    measured = ex.measure(_make_frame(None))
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
    """Catch a regression if the head-tilt sign convention ever flips."""
    ex = NeckStretchLeft()
    # Negative lat_offset = head tilted toward body's -lateral direction.
    # With the fixture's hip orientation (R_HIP at +x, L_HIP at -x), body_lateral = +x,
    # so a -x head offset yields a negative tilt angle.
    result = ex.measure(_make_frame(_h36m_with_head_at(-0.5)))
    assert result["head_lateral_tilt"] < 0, (
        f"expected negative tilt for left-leaning head, got {result['head_lateral_tilt']}"
    )
