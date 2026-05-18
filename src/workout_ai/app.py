import time
import cv2
import numpy as np

from workout_ai.capture import WebcamCapture
from workout_ai.pose2d import Pose2D
from workout_ai.render import Renderer
from workout_ai.analysis.phases import SquatFSM
from workout_ai.analysis.rules_squat import score_rep
from workout_ai.analysis.types import PoseFrame, PhaseState


def run():
    cap = WebcamCapture(device=0, width=1280, height=720)
    pose = Pose2D()
    renderer = Renderer(panel_width=320)
    fsm = SquatFSM()

    rep_count = 0
    running_sum = 0
    last_score: int | None = None

    last_bottom_frame: PoseFrame | None = None

    def on_rep_complete(meta: dict):
        nonlocal rep_count, running_sum, last_score
        if last_bottom_frame is None:
            return
        analysis = score_rep(last_bottom_frame, meta["descent_ms"], meta["ascent_ms"], rep_index=rep_count)
        rep_count += 1
        running_sum += analysis.score
        last_score = analysis.score
        print(f"[rep {analysis.rep_index}] score={analysis.score} components={analysis.components}")
        for v in analysis.violations:
            print(f"    violation: {v.name} severity={v.severity:.2f}")

    fsm.on_rep_complete = on_rep_complete
    cap.start()

    try:
        while True:
            frame = cap.read_latest(timeout=2.0)
            if frame is None:
                break

            ts = time.monotonic()
            kps, scores = pose.infer(frame)
            pf = PoseFrame(timestamp=ts, keypoints_2d=kps, scores=scores, frame_shape=frame.shape[:2])
            prev_state = fsm.state
            state = fsm.update(kps, ts)

            if state == PhaseState.BOTTOM:
                last_bottom_frame = pf

            frame = renderer.draw_skeleton(frame, kps, scores)
            avg = (running_sum / rep_count) if rep_count else 0.0
            display = renderer.compose(
                frame,
                score=last_score,
                running_avg=avg,
                rep_count=rep_count,
                phase=state.value,
                thai_text="",
            )
            cv2.imshow("Workout AI", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
