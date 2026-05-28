import torch
from compression import config
from compression.models.student import StudentPose


def test_student_output_shapes():
    model = StudentPose(width=config.STUDENT_WIDTH, depth=config.STUDENT_DEPTH)
    x = torch.randn(2, 3, config.INPUT_H, config.INPUT_W)
    sx, sy = model(x)
    assert sx.shape == (2, config.NUM_KEYPOINTS, config.SIMCC_X_BINS)
    assert sy.shape == (2, config.NUM_KEYPOINTS, config.SIMCC_Y_BINS)


def test_student_param_count_smaller_than_rtmpose_m():
    model = StudentPose(width=config.STUDENT_WIDTH, depth=config.STUDENT_DEPTH)
    params = sum(p.numel() for p in model.parameters())
    # RTMPose-m is ~13M; the lite student must be clearly smaller.
    assert params < 8_000_000, f"student too big: {params}"
