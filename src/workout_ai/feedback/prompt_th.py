from workout_ai.analysis.types import RepAnalysis

SYSTEM_TH = (
    "คุณเป็นโค้ชฟิตเนสที่ให้คำแนะนำเกี่ยวกับท่าสควอตอย่างกระชับและสุภาพ "
    "ตอบเป็นภาษาไทย 2-3 ประโยค ระบุสิ่งที่ทำถูกและสิ่งที่ควรแก้ไข"
)


def build_user_prompt(rep: RepAnalysis) -> str:
    lines = [
        f"ผลสควอตรอบที่ {rep.rep_index + 1}: คะแนนรวม {rep.score}/100",
        f"  ความลึก: {rep.components['depth']}/30",
        f"  หัวเข่า: {rep.components['valgus']}/25",
        f"  ลำตัว: {rep.components['torso']}/20",
        f"  สมมาตร: {rep.components['symmetry']}/15",
        f"  จังหวะ: {rep.components['tempo']}/10",
        f"เวลาลง: {rep.descent_ms} ms | เวลาขึ้น: {rep.ascent_ms} ms",
    ]
    if rep.violations:
        lines.append("ข้อสังเกต:")
        for v in rep.violations:
            lines.append(f"  - {v.detail_th} (ระดับ {v.severity:.2f})")
    else:
        lines.append("ไม่พบข้อสังเกตที่ต้องแก้ไข")
    lines.append("ช่วยสรุปสั้นๆ ว่าทำดีตรงไหน ควรปรับตรงไหน")
    return "\n".join(lines)
