import threading
from typing import Optional
from workout_ai.analysis.types import RepAnalysis


class LLMWorker:
    """Background thread that processes the most recent RepAnalysis at a time, dropping older ones."""

    def __init__(self, llm):
        self._llm = llm
        self._lock = threading.Lock()
        self._pending: Optional[RepAnalysis] = None
        self._latest_text: Optional[str] = None
        self._cv = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, rep: RepAnalysis):
        with self._cv:
            self._pending = rep  # newer overwrites older
            self._cv.notify()

    def latest(self) -> Optional[str]:
        with self._lock:
            return self._latest_text

    def _loop(self):
        while self._running:
            with self._cv:
                while self._running and self._pending is None:
                    self._cv.wait(timeout=0.1)
                if not self._running:
                    return
                rep = self._pending
                self._pending = None
            try:
                text = self._llm.generate(rep)
            except Exception as e:
                text = f"[LLM error: {e}]"
            with self._lock:
                self._latest_text = text

    def stop(self):
        with self._cv:
            self._running = False
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
