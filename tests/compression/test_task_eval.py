import numpy as np
from compression.eval.task_eval import angle_agreement, upper_body_pck


def test_angle_agreement_zero_when_identical():
    kps = np.random.rand(17, 2).astype(np.float32) * 100 + 50
    scores = np.ones(17, dtype=np.float32)
    diffs = angle_agreement(kps, scores, kps, scores)
    assert abs(diffs["head_lateral_tilt_2d"]) < 1e-6


def test_pck_perfect_when_identical():
    kps = np.random.rand(17, 2).astype(np.float32) * 100 + 50
    pck = upper_body_pck(kps, kps, threshold_px=5.0)
    assert pck == 1.0
