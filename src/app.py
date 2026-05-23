import time
import cv2
import numpy as np

from calibration import BaselinePose, CalibrationError, calibrate_from_samples
from capture import WebcamCapture
from pose2d import Pose2D
from pose3d import Pose3D, Pose3DBuffer, coco17_to_h36m17
from render import Renderer
from selector import choose_exercise
from analysis.camera_view import CameraView, classify_view
from analysis.phases import HoldFSM
from analysis.rules_hold import score_frame, score_hold
from analysis.types import HoldState, LiveSnapshot, PoseFrame
from exercises.base import Exercise
from feedback.llm import ThaiCoachLLM
from feedback.worker import LLMWorker


_LIVE_SUBMIT_INTERVAL_S = 2.5  # see spec §8.1

# Camera capture runs at its native FPS (~30 Hz). These knobs throttle the
# heavy stages so CPU/GPU is only spent when fresh data would actually change
# the FSM or the visualized rig. HoldFSM updates still run every camera frame
# (timing is timestamp-based, not call-count-based), so the lower inference
# rate doesn't affect `in_target_ms` accuracy — only how often the input
# transitions are re-detected.
_POSE_INFERENCE_HZ = 15  # 2D pose detection target rate
_LIFT_HZ = 6  # 3D lift target rate (≥ 5 Hz per README acceptance criteria)
_INFERENCE_INTERVAL_S = 1.0 / _POSE_INFERENCE_HZ
_LIFT_INTERVAL_S = 1.0 / _LIFT_HZ

# Calibration phase duration (seconds). 5 s feels long enough to settle without
# being tedious; matches the NeckWatcher / Zen / pose-nudge prior-art pattern.
_CALIBRATION_DURATION_S = 5.0


def run():
    """Top-level: load heavy weights once, then loop selector → session."""
    cap = WebcamCapture(device=0, width=1280, height=720)
    pose = Pose2D()
    lifter = Pose3D()
    renderer = Renderer(panel_width=320)

    print("Loading Qwen3.5-4B (this takes ~10 seconds the first time)...")
    llm = ThaiCoachLLM()
    print("Warming up LLM...")
    llm.warmup()
    worker = LLMWorker(llm)
    worker.start()
    print("Ready.")

    cap.start()
    try:
        while True:
            try:
                exercise = choose_exercise()
            except SystemExit:
                break
            run_session(cap, pose, lifter, renderer, worker, exercise)
    finally:
        worker.stop()
        cap.stop()
        cv2.destroyAllWindows()


def run_session(
    cap: WebcamCapture,
    pose: Pose2D,
    lifter: Pose3D,
    renderer: Renderer,
    worker: LLMWorker,
    exercise: Exercise,
) -> None:
    """One held-pose session. Returns when the hold completes or the user quits."""
    # Capture the user's neutral pose before the FSM starts so exercise
    # targets are interpreted as deltas from the user's own posture rather
    # than absolute population-average angles.
    baseline = _calibration_phase(cap, pose, renderer, exercise)
    if baseline is None:
        return  # user quit during calibration

    buf3d = Pose3DBuffer(lifter)
    fsm = HoldFSM(target_seconds=exercise.target.hold_seconds)
    max_severity_seen: dict[str, float] = {j.name: 0.0 for j in exercise.target.joints}
    last_live_submit_ts = 0.0
    last_pose_ts = 0.0
    last_lift_ts = 0.0
    last_kps: np.ndarray | None = None
    last_scores: np.ndarray | None = None
    last_rig_3d: np.ndarray | None = None

    completion_holder: dict = {}

    def on_complete(meta: dict) -> None:
        completion_holder["meta"] = meta

    fsm.on_hold_complete = on_complete

    while True:
        frame = cap.read_latest(timeout=2.0)
        if frame is None:
            return

        ts = time.monotonic()

        # Throttle 2D pose inference to _POSE_INFERENCE_HZ. Camera-frame rate is
        # higher (~30 Hz native); on skipped frames we reuse the previous (kps,
        # scores) so downstream FSM / score_frame still get values, but we don't
        # pay the ONNX cost again. The HoldFSM timer is timestamp-based — it
        # still accumulates correctly between fresh inferences.
        if last_kps is None or (ts - last_pose_ts) >= _INFERENCE_INTERVAL_S:
            last_kps, last_scores, _ = pose.infer_with_heatmaps(frame)
            last_pose_ts = ts
            buf3d.push(coco17_to_h36m17(last_kps, last_scores))
        kps, scores = last_kps, last_scores
        assert kps is not None and scores is not None

        pf = PoseFrame(
            timestamp=ts,
            keypoints_2d=kps,
            scores=scores,
            frame_shape=frame.shape[:2],
        )

        # 3D lift is also gated on wall-clock so it stays ≥ _LIFT_HZ.
        if buf3d.ready() and (ts - last_lift_ts) >= _LIFT_INTERVAL_S:
            try:
                last_rig_3d = buf3d.lift(frame.shape[0], frame.shape[1])
                pf.keypoints_3d = last_rig_3d
                last_lift_ts = ts
            except Exception as e:
                print(f"3D lift error: {e}")

        # Camera-view gate: if the user is in a framing this exercise doesn't
        # support, skip scoring entirely and keep the FSM at IDLE while a
        # coaching message asks them to rotate. Forces in_target = False so the
        # FSM tracks "not ready to hold yet".
        current_view = classify_view(kps, scores)
        view_ok = current_view in exercise.target.valid_views

        if not view_ok:
            in_target = False
            violations = []
        else:
            measured = exercise.measure(pf, baseline=baseline)
            in_target, violations = score_frame(exercise.target, measured)
            for v in violations:
                max_severity_seen[v.name] = max(
                    max_severity_seen.get(v.name, 0.0), v.severity
                )

        state = fsm.update(in_target, ts)

        # Live LLM submission, throttled.
        if (
            state in (HoldState.HOLDING, HoldState.DRIFTED)
            and (ts - last_live_submit_ts) >= _LIVE_SUBMIT_INTERVAL_S
        ):
            target_ms = int(exercise.target.hold_seconds * 1000) or 1
            snap = LiveSnapshot(
                exercise_name=exercise.name,
                state=state,
                progress_ratio=min(1.0, fsm.in_target_ms / target_ms),
                current_violations=violations,
            )
            worker.submit(snap, exercise=exercise)
            last_live_submit_ts = ts

        if state is HoldState.COMPLETE:
            meta = completion_holder.get("meta") or {
                "in_target_ms": fsm.in_target_ms,
                "drift_count": fsm.drift_count,
                "completed_ts": ts,
            }
            analysis = score_hold(
                exercise.name, meta, exercise.target, max_severity_seen
            )
            worker.submit(analysis, exercise=exercise)
            print(
                f"[hold {exercise.name}] score={analysis.score} components={analysis.components}"
            )
            # Display the final state for ~3 seconds before returning.
            end_ts = ts + 3.0
            while time.monotonic() < end_ts:
                _show_frame(cap, renderer, exercise, fsm, worker, last_rig_3d)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            return

        # Render this frame.
        target_ms = int(exercise.target.hold_seconds * 1000) or 1
        if not view_ok:
            coaching = _view_coaching_message(current_view, exercise.target.valid_views)
            thai_text = coaching
        else:
            thai_text = worker.latest() or ""
        frame_drawn = renderer.draw_skeleton(frame, kps, scores)
        display = renderer.compose(
            frame_drawn,
            score=None,
            running_avg=0.0,
            rep_count=0,
            phase=exercise.display_th,
            thai_text=thai_text,
            rig_3d_kps=last_rig_3d,
            hold_state=state.value,
            hold_progress=min(1.0, fsm.in_target_ms / target_ms),
        )
        cv2.imshow("Workout AI", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return


_VIEW_TH_NAME = {
    CameraView.FRONT: "ตรง",
    CameraView.THREE_QUARTER: "เฉียง",
    CameraView.SIDE: "ด้านข้าง",
    CameraView.UNKNOWN: "(ตรวจไม่พบ)",
}


def _view_coaching_message(
    current: CameraView,
    valid_views: tuple[CameraView, ...],
) -> str:
    """Build a short Thai coaching message that tells the user to rotate into
    one of the views the active exercise supports. Cheap and stateless — no
    LLM call needed for a UI hint."""
    valid_names = " หรือ ".join(_VIEW_TH_NAME[v] for v in valid_views)
    if current is CameraView.UNKNOWN:
        return f"ขยับให้กล้องเห็นไหล่ทั้งสองข้าง (ต้องการมุม: {valid_names})"
    return f"หันมา{valid_names}เล็กน้อย"


def _calibration_phase(
    cap: WebcamCapture,
    pose: Pose2D,
    renderer: Renderer,
    exercise: Exercise,
) -> BaselinePose | None:
    """Capture the user's neutral pose for `_CALIBRATION_DURATION_S` seconds.

    Renders a "sit naturally" message in the same `Renderer.compose` shell as
    the main session so the transition feels continuous. Returns the resulting
    `BaselinePose`, or `None` if the user pressed 'q' during calibration.

    If calibration fails (too few clean frames — e.g. user is out of frame),
    we surface the failure in the console and return a `None`-equivalent
    "zero baseline" so the session still runs in absolute-angle mode. This
    preserves backward compatibility for users who walk away during the
    calibration step.
    """
    end_ts = time.monotonic() + _CALIBRATION_DURATION_S
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    while time.monotonic() < end_ts:
        frame = cap.read_latest(timeout=0.5)
        if frame is None:
            continue
        kps, scores = pose.infer(frame)
        samples.append((kps, scores))

        remaining = max(0.0, end_ts - time.monotonic())
        frame_drawn = renderer.draw_skeleton(frame, kps, scores)
        display = renderer.compose(
            frame_drawn,
            score=None,
            running_avg=0.0,
            rep_count=0,
            phase=f"{exercise.display_th} — calibrating {remaining:.1f}s",
            thai_text="นั่งตรงตามธรรมชาติ มองตรงไปข้างหน้า",
            rig_3d_kps=None,
            hold_state="idle",
            hold_progress=1.0 - remaining / _CALIBRATION_DURATION_S,
        )
        cv2.imshow("Workout AI", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

    try:
        baseline = calibrate_from_samples(samples, min_clean_frames=30)
        print(
            f"[calibration] OK  tilt={baseline.head_lateral_tilt_deg:+.1f}°  "
            f"shoulder_w={baseline.shoulder_width_px:.0f}px  samples={baseline.sample_count}"
        )
        return baseline
    except CalibrationError as e:
        print(f"[calibration] FAILED: {e}")
        print(
            "[calibration] continuing in absolute-angle mode (no baseline subtraction)"
        )
        return BaselinePose(
            shoulder_width_px=0.0,
            head_lateral_tilt_deg=0.0,
            shoulder_y_delta_norm=0.0,
            sample_count=0,
            captured_ts=time.monotonic(),
        )


def _show_frame(cap, renderer, exercise, fsm, worker, last_rig_3d):
    """Render-only helper used during the post-completion delay."""
    frame = cap.read_latest(timeout=0.5)
    if frame is None:
        return
    thai_text = worker.latest() or ""
    target_ms = int(exercise.target.hold_seconds * 1000) or 1
    display = renderer.compose(
        frame,
        score=None,
        running_avg=0.0,
        rep_count=0,
        phase=exercise.display_th,
        thai_text=thai_text,
        rig_3d_kps=last_rig_3d,
        hold_state="complete",
        hold_progress=min(1.0, fsm.in_target_ms / target_ms),
    )
    cv2.imshow("Workout AI", display)


if __name__ == "__main__":
    run()
