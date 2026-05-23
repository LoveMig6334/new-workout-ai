import time
import cv2
import numpy as np

from capture import WebcamCapture
from pose2d import Pose2D
from pose3d import Pose3D, Pose3DBuffer, coco17_to_h36m17
from render import Renderer
from selector import choose_exercise
from analysis.phases import HoldFSM
from analysis.rules_hold import score_frame, score_hold
from analysis.types import HoldState, LiveSnapshot, PoseFrame
from exercises.base import Exercise
from feedback.llm import ThaiCoachLLM
from feedback.worker import LLMWorker


_LIVE_SUBMIT_INTERVAL_S = 2.5  # see spec §8.1


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
    buf3d = Pose3DBuffer(lifter)
    fsm = HoldFSM(target_seconds=exercise.target.hold_seconds)
    max_severity_seen: dict[str, float] = {j.name: 0.0 for j in exercise.target.joints}
    last_live_submit_ts = 0.0
    frame_idx = 0
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
        kps, scores, _hms = pose.infer_with_heatmaps(frame)
        pf = PoseFrame(
            timestamp=ts,
            keypoints_2d=kps,
            scores=scores,
            frame_shape=frame.shape[:2],
        )

        h36m = coco17_to_h36m17(kps, scores)
        buf3d.push(h36m)
        if frame_idx % 5 == 0 and buf3d.ready():
            try:
                last_rig_3d = buf3d.lift(frame.shape[0], frame.shape[1])
                pf.keypoints_3d = last_rig_3d
            except Exception as e:
                print(f"3D lift error: {e}")

        measured = exercise.measure(pf)
        in_target, violations = score_frame(exercise.target, measured)
        for v in violations:
            max_severity_seen[v.name] = max(max_severity_seen.get(v.name, 0.0), v.severity)

        state = fsm.update(in_target, ts)

        # Live LLM submission, throttled.
        if state in (HoldState.HOLDING, HoldState.DRIFTED) and \
           (ts - last_live_submit_ts) >= _LIVE_SUBMIT_INTERVAL_S:
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
            analysis = score_hold(exercise.name, meta, exercise.target, max_severity_seen)
            worker.submit(analysis, exercise=exercise)
            print(f"[hold {exercise.name}] score={analysis.score} components={analysis.components}")
            # Display the final state for ~3 seconds before returning.
            end_ts = ts + 3.0
            while time.monotonic() < end_ts:
                _show_frame(cap, renderer, exercise, fsm, worker, last_rig_3d)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            return

        # Render this frame.
        target_ms = int(exercise.target.hold_seconds * 1000) or 1
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
        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return


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
