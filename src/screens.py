"""OpenCV screen rendering for the neck-stretch demo.

Each draw_* function returns a BGR canvas to imshow. Kept separate from the
pure RoutineFSM and from app's control flow so the visuals can be iterated on
in isolation. Thai text uses render.put_thai_text.
"""
from __future__ import annotations

import cv2
import numpy as np

from render import put_thai_text

W, H = 1280, 720
SETUP_BUTTON_RECT = (490, 410, 300, 90)  # x, y, w, h

_BG = (24, 24, 28)
_GREEN = (90, 220, 90)
_AMBER = (90, 200, 220)
_GREY = (160, 160, 160)
_WHITE = (245, 245, 245)

_SIDE_TH = {"left": "ซ้าย", "right": "ขวา"}


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def _blank() -> np.ndarray:
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:] = _BG
    return canvas


def _mirror(frame: np.ndarray) -> np.ndarray:
    """Selfie-view mirror, resized to the demo canvas."""
    f = cv2.resize(frame, (W, H))
    return cv2.flip(f, 1)


def draw_setup() -> np.ndarray:
    canvas = _blank()
    put_thai_text(canvas, "ยืดคอ คลายออฟฟิศซินโดรม", (W // 2, 150),
                  font_size=46, color=_WHITE, center=True)
    put_thai_text(canvas, "ใช้เวลา 2 นาที · ยืดคอสลับซ้าย-ขวา 4 เซ็ต",
                  (W // 2, 230), font_size=26, color=_GREY, center=True)
    x, y, w, h = SETUP_BUTTON_RECT
    cv2.rectangle(canvas, (x, y), (x + w, y + h), _GREEN, -1, cv2.LINE_AA)
    put_thai_text(canvas, "เริ่ม", (x + w // 2, y + 22), font_size=40,
                  color=(20, 20, 20), center=True)
    put_thai_text(canvas, "(คลิกปุ่มเพื่อเริ่ม · กด q เพื่อออก)", (W // 2, 560),
                  font_size=22, color=_GREY, center=True)
    return canvas


def _draw_outline(img: np.ndarray, ok: bool) -> None:
    """Semi-transparent humanoid guide centered on the canvas."""
    color = _GREEN if ok else _GREY
    overlay = img.copy()
    cx = W // 2
    cv2.circle(overlay, (cx, 180), 60, color, 3, cv2.LINE_AA)          # head
    cv2.line(overlay, (cx - 120, 300), (cx + 120, 300), color, 3, cv2.LINE_AA)  # shoulders
    cv2.line(overlay, (cx - 90, 520), (cx + 90, 520), color, 3, cv2.LINE_AA)    # hips
    cv2.line(overlay, (cx - 120, 300), (cx - 90, 520), color, 3, cv2.LINE_AA)   # torso L
    cv2.line(overlay, (cx + 120, 300), (cx + 90, 520), color, 3, cv2.LINE_AA)   # torso R
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0.0, dst=img)


def draw_positioning(frame: np.ndarray, progress: float, ok: bool) -> np.ndarray:
    canvas = _mirror(frame)
    _draw_outline(canvas, ok)
    msg = "ค้างไว้..." if ok else "ยืนให้กล้องเห็นหัว ไหล่ และสะโพก"
    put_thai_text(canvas, msg, (W // 2, 40), font_size=30,
                  color=_GREEN if ok else _AMBER, center=True)
    bw, bx, by = 600, (W - 600) // 2, H - 60
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + 24), _GREY, 2, cv2.LINE_AA)
    fill = int(bw * max(0.0, min(1.0, progress)))
    cv2.rectangle(canvas, (bx, by), (bx + fill, by + 24), _GREEN, -1, cv2.LINE_AA)
    return canvas


def draw_countdown(frame: np.ndarray, number: int, side: str | None) -> np.ndarray:
    canvas = _mirror(frame)
    if side:
        put_thai_text(canvas, f"เตรียมยืดคอด้าน{_SIDE_TH.get(side, side)}",
                      (W // 2, 120), font_size=36, color=_WHITE, center=True)
    cv2.putText(canvas, str(number), (W // 2 - 40, H // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 6.0, _WHITE, 12, cv2.LINE_AA)
    return canvas


def draw_hold(
    frame: np.ndarray,
    side: str | None,
    remaining_s: float,
    in_target: bool,
    view_ok: bool,
    thai_text: str,
) -> np.ndarray:
    canvas = _mirror(frame)
    border = _GREEN if in_target else _AMBER
    cv2.rectangle(canvas, (4, 4), (W - 4, H - 4), border, 8)
    put_thai_text(canvas, f"ยืดคอด้าน{_SIDE_TH.get(side or '', '')}", (30, 30),
                  font_size=34, color=_WHITE)
    cv2.putText(canvas, f"{int(np.ceil(remaining_s)):02d}", (W - 150, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, _WHITE, 5, cv2.LINE_AA)
    if not view_ok:
        put_thai_text(canvas, "หันหน้าเข้าหากล้อง", (W // 2, H // 2),
                      font_size=34, color=_AMBER, center=True)
    elif thai_text:
        put_thai_text(canvas, thai_text, (40, H - 120), font_size=28,
                      color=_WHITE, max_width=W - 80)
    return canvas


def draw_transition(frame: np.ndarray, next_side: str | None) -> np.ndarray:
    canvas = _mirror(frame)
    put_thai_text(canvas, "พักสักครู่", (W // 2, H // 2 - 40), font_size=40,
                  color=_WHITE, center=True)
    if next_side:
        put_thai_text(canvas, f"ต่อไปยืดด้าน{_SIDE_TH.get(next_side, next_side)}",
                      (W // 2, H // 2 + 30), font_size=32, color=_GREEN, center=True)
    return canvas


def draw_summary(set_scores: list[int], overall: int, thai_text: str) -> np.ndarray:
    canvas = _blank()
    put_thai_text(canvas, "จบการฝึก", (W // 2, 90), font_size=46,
                  color=_WHITE, center=True)
    put_thai_text(canvas, f"คะแนนรวม {overall}/100", (W // 2, 170), font_size=34,
                  color=_GREEN, center=True)
    for i, sc in enumerate(set_scores):
        put_thai_text(canvas, f"เซ็ต {i + 1}: {sc}/100", (W // 2, 240 + i * 44),
                      font_size=26, color=_GREY, center=True)
    if thai_text:
        put_thai_text(canvas, thai_text, (W // 2 - 400, 470), font_size=26,
                      color=_WHITE, max_width=800)
    put_thai_text(canvas, "(กด q เพื่อออก)", (W // 2, H - 50), font_size=22,
                  color=_GREY, center=True)
    return canvas
