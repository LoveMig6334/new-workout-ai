import cv2
import numpy as np
from workout_ai.capture import WebcamCapture


def test_capture_thread_yields_frames(monkeypatch):
    class FakeVC:
        def __init__(self, idx):
            self.opened = True

        def isOpened(self):
            return self.opened

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            self.opened = False

        def set(self, *args, **kwargs):
            return True

    monkeypatch.setattr(cv2, "VideoCapture", FakeVC)
    cap = WebcamCapture(device=0)
    cap.start()
    try:
        frame = cap.read_latest(timeout=1.0)
        assert frame is not None
        assert frame.shape == (480, 640, 3)
    finally:
        cap.stop()
