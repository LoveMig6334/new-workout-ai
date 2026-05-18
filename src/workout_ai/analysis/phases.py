from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from workout_ai.analysis.angles import knee_angles
from workout_ai.analysis.types import PhaseState


STAND_THRESHOLD = 160.0
BOTTOM_THRESHOLD = 100.0


@dataclass
class SquatFSM:
    state: PhaseState = PhaseState.STANDING
    descent_start_ts: Optional[float] = None
    bottom_ts: Optional[float] = None
    on_rep_complete: Optional[Callable[[dict], None]] = None

    def update(self, kps: np.ndarray, timestamp: float) -> PhaseState:
        l, r = knee_angles(kps)
        angle = (l + r) / 2.0

        if self.state == PhaseState.STANDING:
            if angle < STAND_THRESHOLD:
                self.state = PhaseState.DESCENT
            else:
                self.descent_start_ts = timestamp  # track last confirmed standing frame
        elif self.state == PhaseState.DESCENT:
            if angle < BOTTOM_THRESHOLD:
                self.state = PhaseState.BOTTOM
                self.bottom_ts = timestamp
            elif angle >= STAND_THRESHOLD:
                self.state = PhaseState.STANDING
                self.descent_start_ts = None
        elif self.state == PhaseState.BOTTOM:
            if angle >= BOTTOM_THRESHOLD:
                self.state = PhaseState.ASCENT
        elif self.state == PhaseState.ASCENT:
            if angle >= STAND_THRESHOLD:
                meta = {
                    "descent_ms": int((self.bottom_ts - self.descent_start_ts) * 1000),
                    "ascent_ms": int((timestamp - self.bottom_ts) * 1000),
                    "bottom_ts": self.bottom_ts,
                    "completed_ts": timestamp,
                }
                self.state = PhaseState.STANDING
                self.descent_start_ts = None
                self.bottom_ts = None
                if self.on_rep_complete:
                    self.on_rep_complete(meta)
        return self.state
