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


def test_student_input_scale_075_forward():
    """input_scale=0.75 is a planned ablation value (Task 7b). Stride-2 convs
    with padding produce ceil-style feature sizes, so the head must be sized
    from a real backbone pass — not a divisible-by-32 formula."""
    model = StudentPose(width=config.STUDENT_WIDTH, depth=config.STUDENT_DEPTH,
                        input_scale=0.75)
    x = torch.randn(1, 3, config.INPUT_H, config.INPUT_W)
    sx, sy = model(x)
    assert sx.shape == (1, config.NUM_KEYPOINTS, config.SIMCC_X_BINS)
    assert sy.shape == (1, config.NUM_KEYPOINTS, config.SIMCC_Y_BINS)


def test_student_input_scale_05_forward():
    """input_scale=0.5 (input divisible by 32) should also forward cleanly."""
    model = StudentPose(width=config.STUDENT_WIDTH, depth=config.STUDENT_DEPTH,
                        input_scale=0.5)
    x = torch.randn(1, 3, config.INPUT_H, config.INPUT_W)
    sx, sy = model(x)
    assert sx.shape == (1, config.NUM_KEYPOINTS, config.SIMCC_X_BINS)
    assert sy.shape == (1, config.NUM_KEYPOINTS, config.SIMCC_Y_BINS)
