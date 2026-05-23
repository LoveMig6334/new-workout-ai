import cv2
import numpy as np

# COCO-17 skeleton connections (pairs of keypoint indices)
SKELETON = [
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 6),  # shoulders
    (5, 11),
    (6, 12),  # torso
    (11, 12),  # hips
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]


_THAI_FONT_PATH = "/System/Library/Fonts/Supplemental/Ayuthaya.ttf"


def _load_thai_font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(_THAI_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def put_thai_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_size: int = 22,
    color: tuple[int, int, int] = (255, 255, 255),
    max_width: int | None = None,
    center: bool = False,
) -> None:
    """Draw Thai text onto a BGR ndarray in place.

    `color` is BGR (OpenCV convention). `max_width` (px) word-wraps; otherwise
    splits on '\\n'. `center=True` horizontally centers each line on org[0].
    """
    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_thai_font(font_size)
    rgb = (color[2], color[1], color[0])
    x, y = org

    if max_width:
        lines, line = [], ""
        for word in text.split():
            test = (line + " " + word).strip()
            if draw.textbbox((0, 0), test, font=font)[2] > max_width and line:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)
    else:
        lines = text.split("\n")

    line_h = int(font_size * 1.3)
    cy = y
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        w = bbox[2] - bbox[0]
        lx = x - w // 2 if center else x
        draw.text((lx, cy), ln, fill=rgb, font=font)
        cy += line_h

    np.copyto(img, cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))


class Renderer:
    def __init__(self, panel_width: int = 320):
        self.panel_width = panel_width

    def draw_skeleton(
        self,
        frame: np.ndarray,
        kps: np.ndarray,
        scores: np.ndarray,
        threshold: float = 0.3,
    ) -> np.ndarray:
        out = frame.copy()
        for i, (x, y) in enumerate(kps):
            if scores[i] < threshold:
                continue
            cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)
        for a, b in SKELETON:
            if scores[a] < threshold or scores[b] < threshold:
                continue
            pa = (int(kps[a, 0]), int(kps[a, 1]))
            pb = (int(kps[b, 0]), int(kps[b, 1]))
            cv2.line(out, pa, pb, (255, 200, 0), 2)
        return out

    def compose(
        self,
        frame: np.ndarray,
        score: int | None,
        running_avg: float,
        rep_count: int,
        phase: str,
        thai_text: str,
        rig_3d_kps: np.ndarray | None = None,
        attention: np.ndarray | None = None,
        hold_state: str | None = None,
        hold_progress: float | None = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w + self.panel_width, 3), dtype=np.uint8)

        # Overlay attention on frame if provided
        display_frame = frame
        if attention is not None:
            display_frame = self._overlay_attention(frame, attention)

        canvas[:, :w] = display_frame

        # HUD on the frame
        cv2.putText(
            canvas,
            f"Reps: {rep_count}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"Avg: {running_avg:.1f}",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"Phase: {phase}",
            (12, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 255, 200),
            2,
        )
        if score is not None:
            cv2.putText(
                canvas,
                f"Last: {score}",
                (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3,
            )

        # Right panel
        cv2.rectangle(canvas, (w, 0), (w + self.panel_width, h), (30, 30, 30), -1)
        cv2.putText(
            canvas,
            "Coach",
            (w + 12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        if rig_3d_kps is not None:
            self._draw_rig_3d(
                canvas,
                rig_3d_kps,
                top_left=(w + 10, 40),
                size=(self.panel_width - 20, 220),
            )

        if thai_text:
            self._draw_thai(
                canvas,
                thai_text,
                top_left=(w + 12, 280),
                max_width=self.panel_width - 24,
            )

        if hold_state is not None:
            panel_x = w + 12
            y = h - 90
            cv2.putText(
                canvas,
                f"Hold: {hold_state}",
                (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            if hold_progress is not None:
                bar_y = y + 14
                bar_w = self.panel_width - 24
                cv2.rectangle(
                    canvas,
                    (panel_x, bar_y),
                    (panel_x + bar_w, bar_y + 14),
                    (60, 60, 70),
                    1,
                )
                fill = int(bar_w * max(0.0, min(1.0, hold_progress)))
                color = (90, 220, 90) if hold_state == "holding" else (220, 200, 90)
                if hold_state == "drifted":
                    color = (90, 150, 220)
                cv2.rectangle(
                    canvas, (panel_x, bar_y), (panel_x + fill, bar_y + 14), color, -1
                )

        return canvas

    def _overlay_attention(
        self, frame: np.ndarray, attention: np.ndarray
    ) -> np.ndarray:
        att = (attention * 255).astype(np.uint8)
        att = cv2.resize(att, (frame.shape[1], frame.shape[0]))
        heat = cv2.applyColorMap(att, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.7, heat, 0.3, 0)

    def _draw_rig_3d(
        self,
        canvas: np.ndarray,
        kps_3d: np.ndarray,
        top_left: tuple[int, int],
        size: tuple[int, int],
    ):
        x0, y0 = top_left
        w, h = size
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (50, 50, 50), 1)

        xy = kps_3d[:, :2].copy()
        mins = xy.min(axis=0)
        maxs = xy.max(axis=0)
        span = max((maxs - mins).max(), 1e-3)
        xy = (xy - mins) / span
        xy[:, 0] = x0 + 10 + xy[:, 0] * (w - 20)
        xy[:, 1] = y0 + 10 + xy[:, 1] * (h - 20)

        for i, (x, y) in enumerate(xy):
            cv2.circle(canvas, (int(x), int(y)), 3, (0, 200, 255), -1)
        for a, b in SKELETON:
            if a >= len(xy) or b >= len(xy):
                continue
            pa = (int(xy[a, 0]), int(xy[a, 1]))
            pb = (int(xy[b, 0]), int(xy[b, 1]))
            cv2.line(canvas, pa, pb, (0, 200, 255), 1)

    def _draw_thai(
        self, canvas: np.ndarray, text: str, top_left: tuple[int, int], max_width: int
    ):
        from PIL import Image, ImageDraw, ImageFont

        x, y = top_left
        pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_canvas)
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Ayuthaya.ttf", 16
            )
        except OSError:
            font = ImageFont.load_default()
        words = text.split()
        line = ""
        cy = y
        for word in words:
            test = (line + " " + word).strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            w = bbox[2] - bbox[0]
            if w > max_width and line:
                draw.text((x, cy), line, fill=(255, 255, 255), font=font)
                cy += 20
                line = word
            else:
                line = test
        if line:
            draw.text((x, cy), line, fill=(255, 255, 255), font=font)
        np.copyto(canvas, cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR))
