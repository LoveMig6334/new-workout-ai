import threading
import time
from typing import Optional
import cv2
import numpy as np


class WebcamCapture:
    """Background thread that pulls frames from a webcam and keeps the latest one available."""

    def __init__(self, device: int = 0, width: int = 1280, height: int = 720):
        self._device = device
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._running = False

    def start(self):
        self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera device {self._device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = frame
                self._latest_ts = time.monotonic()

    def read_latest(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self._latest.copy()
            time.sleep(0.005)
        return None

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
