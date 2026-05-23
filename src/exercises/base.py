from dataclasses import dataclass
from typing import Protocol

from analysis.types import PoseFrame


@dataclass(frozen=True)
class JointTarget:
    """One joint angle that must be within range during the hold."""

    name: str
    target_deg: float
    tolerance_deg: float
    detail_th: str  # Thai hint shown if out-of-target


@dataclass(frozen=True)
class PromptTemplate:
    """Per-exercise Thai prompt templates."""

    live: str  # rendered with LiveSnapshot fields
    summary: str  # rendered with HoldAnalysis fields


@dataclass(frozen=True)
class TargetPose:
    joints: tuple[JointTarget, ...]
    hold_seconds: float = 20.0
    side: str | None = None  # "left" / "right" / None for symmetric


class Exercise(Protocol):
    name: str
    display_th: str
    target: TargetPose
    prompt: PromptTemplate

    def measure(self, frame: PoseFrame) -> dict[str, float]:
        """Return {joint_name: degrees} for every joint in self.target.joints.
        Pure function of one frame. May return NaN if keypoints unavailable."""
        ...
