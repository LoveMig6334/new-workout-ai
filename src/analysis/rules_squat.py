from analysis.angles import (
    knee_angles,
    torso_lean_deg,
    hip_below_knee,
    knee_valgus_ratio,
)
from analysis.types import PoseFrame, RepAnalysis, RuleViolation

VALGUS_THRESHOLD = 0.15


def score_rep(
    bottom_frame: PoseFrame, descent_ms: int, ascent_ms: int, rep_index: int = 0
) -> RepAnalysis:
    kps = bottom_frame.keypoints_2d
    violations: list[RuleViolation] = []
    components: dict[str, int] = {}

    # --- Depth (30 pts) ---
    if hip_below_knee(kps):
        components["depth"] = 30
    else:
        components["depth"] = 0
        violations.append(
            RuleViolation(
                name="shallow_depth",
                severity=1.0,
                detail_th="ลงไม่ลึกพอ สะโพกยังสูงกว่าหัวเข่า",
            )
        )

    # --- Valgus (25 pts) ---
    l_v, r_v = knee_valgus_ratio(kps)
    worst_valgus = max(abs(l_v), abs(r_v))
    if worst_valgus < VALGUS_THRESHOLD:
        components["valgus"] = 25
    else:
        severity = min(1.0, (worst_valgus - VALGUS_THRESHOLD) / 0.3)
        components["valgus"] = int(25 * (1.0 - severity))
        violations.append(
            RuleViolation(
                name="knee_valgus",
                severity=severity,
                detail_th="หัวเข่าเข้าด้านใน ควรกางหัวเข่าออกตามแนวปลายเท้า",
            )
        )

    # --- Torso (20 pts) ---
    lean = torso_lean_deg(kps)
    if 20.0 <= lean <= 55.0:
        components["torso"] = 20
    elif lean > 55.0:
        components["torso"] = max(0, int(20 * (1 - (lean - 55) / 30)))
        violations.append(
            RuleViolation(
                name="excessive_forward_lean",
                severity=min(1.0, (lean - 55) / 30),
                detail_th="ลำตัวโน้มไปข้างหน้ามากเกินไป",
            )
        )
    else:
        components["torso"] = max(0, int(20 * (1 - (20 - lean) / 20)))
        violations.append(
            RuleViolation(
                name="too_upright",
                severity=min(1.0, (20 - lean) / 20),
                detail_th="ลำตัวตั้งตรงเกินไป ควรโน้มไปข้างหน้าเล็กน้อย",
            )
        )

    # --- Symmetry (15 pts) ---
    l_ang, r_ang = knee_angles(kps)
    delta = abs(l_ang - r_ang)
    if delta < 10.0:
        components["symmetry"] = 15
    else:
        severity = min(1.0, (delta - 10.0) / 20.0)
        components["symmetry"] = int(15 * (1.0 - severity))
        violations.append(
            RuleViolation(
                name="asymmetric",
                severity=severity,
                detail_th="ซ้ายและขวาไม่สมมาตร",
            )
        )

    # --- Tempo (10 pts) ---
    if descent_ms >= ascent_ms:
        components["tempo"] = 10
    else:
        ratio = descent_ms / max(1, ascent_ms)
        components["tempo"] = int(10 * ratio)
        violations.append(
            RuleViolation(
                name="fast_descent",
                severity=1.0 - ratio,
                detail_th="ลงเร็วกว่าขึ้น ควรลงช้าๆ ควบคุมการเคลื่อนไหว",
            )
        )

    total = sum(components.values())
    return RepAnalysis(
        rep_index=rep_index,
        score=total,
        components=components,
        violations=violations,
        descent_ms=descent_ms,
        ascent_ms=ascent_ms,
        bottom_frame_keypoints_2d=kps.copy(),
        bottom_frame_keypoints_3d=bottom_frame.keypoints_3d.copy()
        if bottom_frame.keypoints_3d is not None
        else None,
    )
