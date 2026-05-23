from analysis.angles import head_lateral_tilt_2d
from analysis.types import PoseFrame
from exercises.base import JointTarget, PromptTemplate, TargetPose


# Initial guess. Calibrate per spec §11.3 (see Task 13). Calibration is against
# the 2D measurement (nose + shoulders in image plane), since MotionBERT-Lite's
# H36M training does not cover the seated / upper-body-only camera framing this
# project targets — see src/test_2D_3D.py for the comparison harness.
_NECK_TILT_TARGET_LEFT_DEG = -35.0
_NECK_TILT_TOLERANCE_DEG = 10.0


_LIVE_TH = (
    "ผู้ใช้กำลังทำท่า {exercise_th} (สถานะ: {state}, ความคืบหน้า {progress_pct}%) "
    "ข้อสังเกตล่าสุด: {violations} "
    "ให้คำแนะนำสั้น ๆ 1 ประโยคเป็นภาษาไทยเพื่อช่วยให้ผู้ใช้ปรับท่าให้ถูกต้อง"
)

_SUMMARY_TH = (
    "ผลการทำท่า {exercise_th}: คะแนนรวม {score}/100\n"
    "  ระยะเวลา: {duration}/50\n"
    "  ความแม่นยำ: {precision}/30\n"
    "  ความนิ่ง: {stability}/20\n"
    "ข้อสังเกต: {violations}\n"
    "ช่วยสรุปสั้น ๆ เป็นภาษาไทยว่าทำดีตรงไหน ควรปรับตรงไหน"
)


class NeckStretchLeft:
    name = "neck_stretch_left"
    display_th = "ยืดคอด้านซ้าย"
    target = TargetPose(
        joints=(
            JointTarget(
                name="head_lateral_tilt",
                target_deg=_NECK_TILT_TARGET_LEFT_DEG,
                tolerance_deg=_NECK_TILT_TOLERANCE_DEG,
                detail_th="เอียงศีรษะไปทางซ้ายมากขึ้นอีกนิด",
            ),
        ),
        hold_seconds=20.0,
        side="left",
    )
    prompt = PromptTemplate(live=_LIVE_TH, summary=_SUMMARY_TH)

    def measure(self, frame: PoseFrame) -> dict[str, float]:
        return {
            "head_lateral_tilt": head_lateral_tilt_2d(
                frame.keypoints_2d, frame.scores
            )
        }
