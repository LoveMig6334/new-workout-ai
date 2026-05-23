"""Real-time guided neck-stretch demo.

Flow: selector -> setup (Start button) -> positioning (outline + calibration)
-> [countdown -> 25s hold -> transition] x4 -> summary. Spoken Thai coaching via
Gemini TTS (with macOS say fallback). The routine logic lives in the pure
RoutineFSM (src/routine.py); this module owns the camera loop, pose inference,
scoring accumulation, rendering, and audio.
"""
from __future__ import annotations

import time

import cv2

import screens
from analysis.angles import L_HIP, L_SHOULDER, NOSE, R_HIP, R_SHOULDER
from analysis.camera_view import classify_view
from analysis.rules_hold import score_frame, score_hold
from analysis.types import HoldAnalysis, HoldState, LiveSnapshot, PoseFrame, RuleViolation
from calibration import BaselinePose, CalibrationError, calibrate_from_samples
from capture import WebcamCapture
from exercises.neck_stretch import NeckStretchLeft, NeckStretchRight
from feedback.llm import ThaiCoachLLM
from feedback.tts import GeminiTTS, TTSWorker
from feedback.worker import LLMWorker
from pose2d import Pose2D
from routine import (
    EV_COUNTDOWN,
    EV_POSITION_OK,
    EV_ROUTINE_COMPLETE,
    EV_SET_COMPLETE,
    EV_SET_STARTED,
    EV_SWITCH_SIDES,
    RoutineConfig,
    RoutineFSM,
    RoutinePhase,
)
from selector import choose_routine

_REQUIRED_KPS = (NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
_KP_FLOOR = 0.3
_LIVE_SUBMIT_INTERVAL_S = 2.5
# Derived from NeckStretchLeft; update if the routine adds a different exercise type.
_VALID_VIEWS = NeckStretchLeft().target.valid_views
_WINDOW = "Workout AI"

# Fixed Thai cue phrases, pre-synthesized at startup so they play instantly.
_CUE_PHRASES = {
    "count_3": "สาม",
    "count_2": "สอง",
    "count_1": "หนึ่ง",
    "start_left": "เริ่มยืดคอไปทางซ้าย ค่อย ๆ เอียงศีรษะ",
    "start_right": "เริ่มยืดคอไปทางขวา ค่อย ๆ เอียงศีรษะ",
    "switch_left": "เปลี่ยนไปยืดด้านซ้าย",
    "switch_right": "เปลี่ยนไปยืดด้านขวา",
    "done": "เยี่ยมมาก ทำครบทุกเซ็ตแล้ว",
    "face_camera": "กรุณาหันหน้าเข้าหากล้อง",
}


def _pose_ready(scores, view) -> bool:
    return (
        all(float(scores[i]) >= _KP_FLOOR for i in _REQUIRED_KPS)
        and view in _VALID_VIEWS
    )


def run():
    cap = WebcamCapture(device=0, width=1280, height=720)
    pose = Pose2D()

    print("Loading Qwen3.5-4B (this takes ~10 seconds the first time)...")
    llm = ThaiCoachLLM()
    print("Warming up LLM...")
    llm.warmup()
    worker = LLMWorker(llm)
    worker.start()

    print("Initializing TTS + pre-caching cues...")
    tts_worker = TTSWorker(GeminiTTS())
    tts_worker.precache(_CUE_PHRASES)
    tts_worker.start()
    print("Ready.")

    side_exercises = {"left": NeckStretchLeft(), "right": NeckStretchRight()}
    cap.start()
    try:
        while True:
            try:
                choose_routine()
            except SystemExit:
                break
            run_neck_stretch_routine(cap, pose, worker, tts_worker, side_exercises)
    finally:
        worker.stop()
        tts_worker.stop()
        cap.stop()
        cv2.destroyAllWindows()


def _aggregate(set_analyses: list[HoldAnalysis], exercise_name: str) -> HoldAnalysis:
    n = max(1, len(set_analyses))
    comp = {
        k: round(sum(a.components.get(k, 0) for a in set_analyses) / n)
        for k in ("duration", "precision", "stability")
    }
    worst: dict[str, RuleViolation] = {}
    for a in set_analyses:
        for v in a.violations:
            if v.name not in worst or v.severity > worst[v.name].severity:
                worst[v.name] = v
    return HoldAnalysis(
        exercise_name=exercise_name,
        score=sum(comp.values()),
        components=comp,
        violations=list(worst.values()),
        in_target_ms=sum(a.in_target_ms for a in set_analyses),
        drift_count=sum(a.drift_count for a in set_analyses),
    )


def run_neck_stretch_routine(cap, pose, worker, tts_worker, side_exercises, config=None):
    fsm = RoutineFSM(config or RoutineConfig())
    cv2.namedWindow(_WINDOW)
    click = {"start": False}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and screens.point_in_rect(
            x, y, screens.SETUP_BUTTON_RECT
        ):
            click["start"] = True

    cv2.setMouseCallback(_WINDOW, _on_mouse)

    baseline: BaselinePose | None = None
    calib_samples: list = []
    set_analyses: list[HoldAnalysis] = []
    cur: dict | None = None
    last_frame_ts = time.monotonic()
    last_live_submit = 0.0
    last_spoken = ""
    summary_submit_ts = 0.0
    spoke_summary = False
    view_bad_since: float | None = None
    view_nudge_ts = 0.0

    while True:
        frame = cap.read_latest(timeout=2.0)
        if frame is None:
            return
        now = time.monotonic()

        kps, scores = pose.infer(frame)
        view = classify_view(kps, scores)
        pose_ready = _pose_ready(scores, view)

        side = fsm.current_side
        in_target = False
        violations: list[RuleViolation] = []
        view_ok = view in _VALID_VIEWS
        if fsm.phase is RoutinePhase.HOLD and side is not None and view_ok:
            ex = side_exercises[side]
            pf = PoseFrame(now, kps, scores, frame.shape[:2])
            measured = ex.measure(pf, baseline=baseline)
            in_target, violations = score_frame(ex.target, measured)

        if fsm.phase is RoutinePhase.POSITIONING and pose_ready:
            calib_samples.append((kps, scores))

        if click["start"]:
            click["start"] = False
            fsm.start(now)

        for ev in fsm.update(now, pose_ready, in_target):
            if ev.kind == EV_POSITION_OK:
                try:
                    baseline = calibrate_from_samples(calib_samples, min_clean_frames=30)
                    print(f"[calibration] OK tilt={baseline.head_lateral_tilt_deg:+.1f}")
                except CalibrationError as e:
                    print(f"[calibration] FAILED ({e}); absolute-angle mode")
                    baseline = BaselinePose(0.0, 0.0, 0.0, 0, now)
            elif ev.kind == EV_COUNTDOWN:
                tts_worker.play_cue(f"count_{ev.value}")
            elif ev.kind == EV_SET_STARTED:
                tts_worker.play_cue(f"start_{ev.value}")
                last_frame_ts = now
                cur = {
                    "in_target_ms": 0,
                    "drift_count": 0,
                    "max_sev": {j.name: 0.0 for j in side_exercises[ev.value].target.joints},
                    "was_in_target": False,
                }
                last_live_submit = 0.0
                last_spoken = ""
            elif ev.kind == EV_SET_COMPLETE and cur is not None:
                done_side = fsm.config.order[ev.value]
                ex = side_exercises[done_side]
                meta = {
                    "in_target_ms": cur["in_target_ms"],
                    "drift_count": cur["drift_count"],
                    "completed_ts": now,
                }
                set_analyses.append(score_hold(ex.name, meta, ex.target, cur["max_sev"]))
                cur = None
            elif ev.kind == EV_SWITCH_SIDES:
                tts_worker.play_cue(f"switch_{ev.value}")
            elif ev.kind == EV_ROUTINE_COMPLETE:
                tts_worker.play_cue("done")
                agg = _aggregate(set_analyses, side_exercises["left"].name)
                worker.submit(agg, exercise=side_exercises["left"])
                summary_submit_ts = now
                spoke_summary = False

        # HOLD accumulation + live feedback.
        if fsm.phase is RoutinePhase.HOLD and cur is not None:
            dt = now - last_frame_ts
            if in_target:
                cur["in_target_ms"] += int(dt * 1000)
            if cur["was_in_target"] and not in_target:
                cur["drift_count"] += 1
            cur["was_in_target"] = in_target
            for v in violations:
                cur["max_sev"][v.name] = max(cur["max_sev"].get(v.name, 0.0), v.severity)

            if view_ok and side is not None and now - last_live_submit >= _LIVE_SUBMIT_INTERVAL_S:
                ex = side_exercises[side]
                progress = min(1.0, fsm.hold_elapsed_s / fsm.config.hold_s)
                snap = LiveSnapshot(
                    exercise_name=ex.name,
                    state=HoldState.HOLDING if in_target else HoldState.DRIFTED,
                    progress_ratio=progress,
                    current_violations=violations,
                )
                worker.submit(snap, exercise=ex)
                last_live_submit = now
            txt = worker.latest()
            if txt and txt != last_spoken and not txt.startswith("[LLM error"):
                tts_worker.submit_feedback(txt)
                last_spoken = txt

            # Spoken view nudge if framing stays invalid > 3s.
            if not view_ok:
                if view_bad_since is None:
                    view_bad_since = now
                elif now - view_bad_since > 3.0 and now - view_nudge_ts > 5.0:
                    tts_worker.play_cue("face_camera")
                    view_nudge_ts = now
            else:
                view_bad_since = None

        # Speak the LLM summary once, shortly after it's submitted.
        if (
            fsm.phase is RoutinePhase.SUMMARY
            and not spoke_summary
            and now - summary_submit_ts > 1.0
        ):
            t = worker.latest()
            if t and not t.startswith("[LLM error"):
                tts_worker.submit_feedback(t)
                spoke_summary = True

        last_frame_ts = now

        # Render the active screen for this phase (pure — no audio side-effects).
        canvas = _compose(fsm, frame, in_target, view_ok, worker, set_analyses)
        cv2.imshow(_WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or fsm.phase is RoutinePhase.DONE:
            return


def _compose(fsm, frame, in_target, view_ok, worker, set_analyses):
    """Pick + draw the screen for the current phase. Pure — no audio side-effects."""
    phase = fsm.phase
    if phase is RoutinePhase.SETUP:
        return screens.draw_setup()
    if phase is RoutinePhase.POSITIONING:
        return screens.draw_positioning(frame, fsm.position_progress,
                                        ok=fsm.position_progress > 0.0)
    if phase is RoutinePhase.COUNTDOWN:
        return screens.draw_countdown(frame, fsm.countdown_value, fsm.current_side)
    if phase is RoutinePhase.HOLD:
        thai = worker.latest() or ""
        return screens.draw_hold(frame, fsm.current_side, fsm.hold_remaining_s,
                                 in_target, view_ok, thai)
    if phase is RoutinePhase.TRANSITION:
        return screens.draw_transition(frame, fsm.next_side)
    # SUMMARY / DONE
    scores = [a.score for a in set_analyses]
    overall = round(sum(scores) / len(scores)) if scores else 0
    summary_txt = worker.latest() or ""
    return screens.draw_summary(scores, overall, summary_txt)


if __name__ == "__main__":
    run()
