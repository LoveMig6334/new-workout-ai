"""Live 3-panel diagnostic app for the 2D → 3D pose pipeline.

Three panels side-by-side:

  [ Camera (raw) ]  [ 2D pose overlay ]  [ 3D rig (lifted)  ]

Each panel carries a measurement readout so you can see — in real time, on your
own desk-camera framing — whether the 2D nose+shoulders neck-tilt (the new
canonical measurement, per recommendation 2 of the seated-pipeline review) and
the 3D pelvis-rooted neck-tilt (the old measurement, still computed for
comparison) agree or diverge.

Run:

    uv run python src/test_2D_3D.py

Keys:
    q / Esc — quit

Why this exists
---------------
MotionBERT-Lite was trained on Human3.6M with full-body visible. For a desk
camera that frames only the head/shoulders/torso, the lifted hips and legs
are out-of-distribution and the body axes derived from them (pelvis → thorax,
L_hip → R_hip) are unreliable. The 2D head-lateral-tilt uses only COCO-17
nose + shoulders and does not depend on the lifted lower body at all. Watch
the two numbers in the panels: if 2D is stable while 3D jitters or is biased,
that's the seated-OOD effect in action.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis.angles import (  # noqa: E402
    NOSE,
    L_HIP,
    L_SHOULDER,
    R_HIP,
    R_SHOULDER,
    head_lateral_tilt_2d,
)
from analysis.angles_3d import head_lateral_tilt_3d  # noqa: E402
from capture import WebcamCapture  # noqa: E402
from exercises.neck_stretch import NeckStretchLeft  # noqa: E402
from pose2d import Pose2D  # noqa: E402
from render import SKELETON  # noqa: E402

CAM_WIDTH = 640
CAM_HEIGHT = 480
PANEL_PAD = 80  # extra vertical room below each panel for readouts
LIFT_EVERY_N_FRAMES = 5  # match app.run_session cadence; full lift every frame is wasteful

# H36M-17 bones used by the 3D rig panel (matches the notebooks' topology).
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def _draw_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = (240, 240, 240),
    scale: float = 0.55,
    thick: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _make_panel(w: int, h: int, title: str) -> np.ndarray:
    panel = np.full((h + PANEL_PAD, w, 3), 22, dtype=np.uint8)
    _draw_text(panel, title, (10, 22), color=(255, 255, 255), scale=0.7, thick=2)
    return panel


def _draw_skeleton_2d(
    img: np.ndarray, kps: np.ndarray, scores: np.ndarray, threshold: float = 0.3
) -> None:
    for a, b in SKELETON:
        if scores[a] < threshold or scores[b] < threshold:
            continue
        pa = (int(kps[a, 0]), int(kps[a, 1]))
        pb = (int(kps[b, 0]), int(kps[b, 1]))
        cv2.line(img, pa, pb, (255, 200, 0), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(kps):
        if scores[i] < threshold:
            continue
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1, cv2.LINE_AA)


def _draw_head_tilt_2d_overlay(
    img: np.ndarray, kps: np.ndarray, scores: np.ndarray, threshold: float = 0.3
) -> None:
    """Visualize the inputs to head_lateral_tilt_2d: mid-shoulders → nose line
    and the body-lateral reference (L_shoulder → R_shoulder). Makes it obvious
    which keypoints the measurement is actually reading from."""
    if (
        scores[NOSE] < threshold
        or scores[L_SHOULDER] < threshold
        or scores[R_SHOULDER] < threshold
    ):
        return
    l = kps[L_SHOULDER]
    r = kps[R_SHOULDER]
    n = kps[NOSE]
    mid = (l + r) / 2.0
    cv2.line(img, (int(l[0]), int(l[1])), (int(r[0]), int(r[1])), (0, 180, 255), 2, cv2.LINE_AA)
    cv2.line(img, (int(mid[0]), int(mid[1])), (int(n[0]), int(n[1])), (60, 60, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (int(mid[0]), int(mid[1])), 5, (0, 180, 255), -1, cv2.LINE_AA)


def _render_rig_3d(
    panel: np.ndarray,
    rig: Optional[np.ndarray],
    box: tuple[int, int, int, int],
) -> None:
    """Render an H36M-17 rig inside the given bounding box (x0, y0, w, h).
    Uses the same view convention as the notebooks (X horizontal, Y-vertical = -3D_y)."""
    x0, y0, w, h = box
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
    if rig is None:
        _draw_text(panel, "warming up (need 27 frames)", (x0 + 12, y0 + h // 2),
                   color=(160, 160, 160), scale=0.5)
        return

    pts = np.stack([rig[:, 0], -rig[:, 1]], axis=1)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = max((maxs - mins).max(), 1e-3)
    pts = (pts - mins) / span
    # Centre + leave a 10 px margin.
    pts[:, 0] = x0 + 10 + pts[:, 0] * (w - 20)
    pts[:, 1] = y0 + 10 + pts[:, 1] * (h - 20)

    for a, b in H36M_SKELETON:
        pa = (int(pts[a, 0]), int(pts[a, 1]))
        pb = (int(pts[b, 0]), int(pts[b, 1]))
        cv2.line(panel, pa, pb, (0, 200, 255), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(panel, (int(x), int(y)), 3, (0, 200, 255), -1, cv2.LINE_AA)


def _try_load_pose3d():
    """Optional — Pose3D needs MotionBERT weights and vendor code. Don't crash
    the diagnostic app if either is missing; just disable the 3D panel."""
    try:
        from pose3d import Pose3D, Pose3DBuffer, coco17_to_h36m17
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"[test_2D_3D] 3D lifter unavailable: {exc}")
        return None, None, None

    try:
        lifter = Pose3D()
    except Exception as exc:  # pragma: no cover - weights missing etc.
        print(f"[test_2D_3D] Pose3D() construction failed: {exc}")
        return None, None, None

    return lifter, Pose3DBuffer(lifter), coco17_to_h36m17


def _tilt_status(tilt: float, target: float, tol: float) -> tuple[str, tuple[int, int, int]]:
    if np.isnan(tilt):
        return "no measurement", (160, 160, 160)
    if abs(tilt - target) <= tol:
        return "IN TARGET", (90, 220, 90)
    return "out of target", (90, 200, 220)


def run() -> None:
    exercise = NeckStretchLeft()
    target = exercise.target.joints[0].target_deg
    tol = exercise.target.joints[0].tolerance_deg

    print(f"[test_2D_3D] Loading 2D pose (RTMPose-s)...")
    pose2d = Pose2D(device="cpu", mode="lightweight")

    print(f"[test_2D_3D] Loading 3D lifter (MotionBERT-Lite)...")
    lifter, buf, coco_to_h36m = _try_load_pose3d()
    have_3d = lifter is not None

    print(f"[test_2D_3D] Opening webcam at {CAM_WIDTH}x{CAM_HEIGHT}...")
    cam = WebcamCapture(device=0, width=CAM_WIDTH, height=CAM_HEIGHT)
    cam.start()

    last_rig_3d: Optional[np.ndarray] = None
    frame_counter = 0

    # Smooth FPS over a small window.
    frame_times: list[float] = []

    window_name = "test_2D_3D"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            t0 = time.time()
            frame_bgr = cam.read_latest(timeout=0.5)
            if frame_bgr is None:
                continue

            kps, scores = pose2d.infer(frame_bgr)
            tilt_2d = head_lateral_tilt_2d(kps, scores)

            tilt_3d = float("nan")
            if have_3d and coco_to_h36m is not None and buf is not None and lifter is not None:
                h36m = coco_to_h36m(kps, scores)
                buf.push(h36m)
                if buf.ready() and frame_counter % LIFT_EVERY_N_FRAMES == 0:
                    last_rig_3d = buf.lift(
                        frame_h=frame_bgr.shape[0], frame_w=frame_bgr.shape[1]
                    )
                if last_rig_3d is not None:
                    tilt_3d = head_lateral_tilt_3d(last_rig_3d)

            # ----- compose panels -----
            cam_panel = _make_panel(CAM_WIDTH, CAM_HEIGHT, "1. Camera (raw)")
            cam_panel[30 : 30 + CAM_HEIGHT, :CAM_WIDTH] = frame_bgr

            two_d_panel = _make_panel(CAM_WIDTH, CAM_HEIGHT, "2. 2D pose + neck-tilt")
            overlay = frame_bgr.copy()
            _draw_skeleton_2d(overlay, kps, scores)
            _draw_head_tilt_2d_overlay(overlay, kps, scores)
            two_d_panel[30 : 30 + CAM_HEIGHT, :CAM_WIDTH] = overlay

            three_d_panel = _make_panel(CAM_WIDTH, CAM_HEIGHT, "3. 3D rig (MotionBERT-Lite)")
            _render_rig_3d(three_d_panel, last_rig_3d, box=(20, 40, CAM_WIDTH - 40, CAM_HEIGHT - 20))

            # ----- readouts -----
            # Camera panel: per-keypoint confidences for the ones the 2D measurement uses,
            # plus the hips that the 3D measurement secretly leans on.
            base_y = 30 + CAM_HEIGHT + 22
            for i, (name, idx) in enumerate(
                [
                    ("nose", NOSE),
                    ("L_sh", L_SHOULDER),
                    ("R_sh", R_SHOULDER),
                    ("L_hip", L_HIP),
                    ("R_hip", R_HIP),
                ]
            ):
                s = float(scores[idx])
                color = (90, 220, 90) if s >= 0.5 else (90, 200, 220) if s >= 0.3 else (90, 90, 220)
                _draw_text(cam_panel, f"{name}: {s:.2f}", (12 + i * 120, base_y), color=color)

            now = time.time()
            frame_times.append(now)
            frame_times = [t for t in frame_times if now - t < 1.0]
            fps = len(frame_times)
            _draw_text(
                cam_panel,
                f"fps={fps}  press q to quit",
                (12, 30 + CAM_HEIGHT + 50),
                color=(180, 180, 180),
                scale=0.5,
            )

            # 2D panel: the canonical measurement.
            label_2d, color_2d = _tilt_status(tilt_2d, target, tol)
            tilt_str_2d = f"{tilt_2d:+.1f}°" if not np.isnan(tilt_2d) else "--"
            _draw_text(
                two_d_panel,
                f"head_lateral_tilt_2d = {tilt_str_2d}    target {target:+.0f}° ± {tol:.0f}°    [{label_2d}]",
                (12, 30 + CAM_HEIGHT + 22),
                color=color_2d,
                scale=0.55,
                thick=1,
            )
            _draw_text(
                two_d_panel,
                "uses: nose, L_shoulder, R_shoulder (COCO-17). desk-camera safe.",
                (12, 30 + CAM_HEIGHT + 50),
                color=(170, 170, 170),
                scale=0.45,
            )

            # 3D panel: legacy comparison + which lifted joints feed into it.
            if not have_3d:
                _draw_text(
                    three_d_panel,
                    "3D lifter unavailable (see console).",
                    (12, 30 + CAM_HEIGHT + 22),
                    color=(170, 170, 170),
                )
            else:
                tilt_str_3d = f"{tilt_3d:+.1f}°" if not np.isnan(tilt_3d) else "--"
                label_3d, color_3d = _tilt_status(tilt_3d, target, tol)
                _draw_text(
                    three_d_panel,
                    f"head_lateral_tilt_3d = {tilt_str_3d}    [{label_3d}]",
                    (12, 30 + CAM_HEIGHT + 22),
                    color=color_3d,
                )
                _draw_text(
                    three_d_panel,
                    "uses lifted PELVIS + hips. unreliable when lower body cropped.",
                    (12, 30 + CAM_HEIGHT + 50),
                    color=(170, 170, 170),
                    scale=0.45,
                )

            canvas = np.hstack([cam_panel, two_d_panel, three_d_panel])
            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            frame_counter += 1
            # Cap UI loop ~60 fps; 2D inference is the bottleneck anyway.
            elapsed = time.time() - t0
            if elapsed < 1 / 60:
                time.sleep(1 / 60 - elapsed)
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
