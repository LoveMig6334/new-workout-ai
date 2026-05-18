import time
import cv2

from workout_ai.capture import WebcamCapture
from workout_ai.pose2d import Pose2D
from workout_ai.render import Renderer


def run():
    cap = WebcamCapture(device=0, width=1280, height=720)
    pose = Pose2D()
    renderer = Renderer(panel_width=320)
    cap.start()

    rep_count = 0
    running_avg = 0.0
    last_score = None

    try:
        while True:
            frame = cap.read_latest(timeout=2.0)
            if frame is None:
                break

            kps, scores = pose.infer(frame)
            frame = renderer.draw_skeleton(frame, kps, scores)
            display = renderer.compose(
                frame,
                score=last_score,
                running_avg=running_avg,
                rep_count=rep_count,
                phase="standing",
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
