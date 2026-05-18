import numpy as np
from workout_ai.render import Renderer


def test_draw_skeleton_does_not_crash():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    kps = np.array([[100 + i * 5, 100 + i * 5] for i in range(17)], dtype=np.float32)
    scores = np.ones((17,), dtype=np.float32) * 0.9
    r = Renderer(panel_width=320)
    out = r.draw_skeleton(frame, kps, scores)
    assert out.shape == frame.shape


def test_compose_with_panel_returns_wider_image():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r = Renderer(panel_width=320)
    out = r.compose(frame, score=85, running_avg=72.5, rep_count=4, phase="bottom", thai_text="")
    assert out.shape == (480, 640 + 320, 3)
