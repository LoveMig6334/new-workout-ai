import os
import numpy as np
import pytest

CKPT = "models/compression/coreml/student_int8.mlpackage"


@pytest.mark.skipif(not os.path.exists(CKPT), reason="no compressed model yet")
def test_student_pose2d_returns_17_keypoints():
    from pose2d import StudentPose2D
    p = StudentPose2D(CKPT)
    frame = np.full((480, 640, 3), 127, np.uint8)
    kps, scores = p.infer(frame)
    assert kps.shape == (17, 2)
    assert scores.shape == (17,)
