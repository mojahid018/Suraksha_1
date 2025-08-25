from typing import Dict, List

import cv2

from .utils import center_of_box, distance


COLORS = {
    "M": (60, 220, 60),     # green
    "F": (255, 80, 180),    # magenta/pink
    "U": (200, 200, 200),   # gray
}
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RISK_COLORS = {
    "LOW": (0, 200, 0),
    "MEDIUM": (0, 165, 255),
    "HIGH": (0, 0, 255),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_transparent_rect(img, x1, y1, x2, y2, color, alpha=0.85):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_chip(img, x, y, text, bg_color, text_color=(255, 255, 255), font_scale=0.7, thickness=2, pad_x=10, pad_y=6, alpha=0.85):
    # Measure text
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    w, h = tw + 2 * pad_x, th + 2 * pad_y

    # Background
    _draw_transparent_rect(img, x, y, x + w, y + h, bg_color, alpha=alpha)

    # Text baseline inside chip
    tx = x + pad_x
    ty = y + pad_y + th
    cv2.putText(img, text, (int(tx), int(ty)), FONT, font_scale, text_color, thickness, cv2.LINE_AA)

    return x + w + int(pad_x * 0.8)  # new x (for next chip)


def _text_size(text, font_scale, thickness):
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    return tw, th


def draw_frame(
    frame,
    tracks: Dict[int, dict],
    genders: Dict[int, str],
    male_count: int,
    female_count: int,
    events: List[str],
    level: str,
    score: int,
    fps: float = None,
    compact: bool = True,  # compact HUD by default (less clutter)
):
    H, W = frame.shape[:2]
    frame_vis = frame.copy()

    # Scale UI based on width
    s = max(0.6, min(1.3, W / 1280.0))
    font_scale = 0.7 * s
    small_font = 0.6 * s
    thick = max(1, int(2 * s))
    pad = int(10 * s)

    # 1) Draw person boxes and compact labels
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = [int(v) for v in tr["xyxy"]]
        g = genders.get(tid, "U")
        color = COLORS.get(g, (200, 200, 200))
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
        # Shorter label: "ID 12 (F/M/U)"
        label = f"ID {tid} ({g})"
        (tw, th), _ = cv2.getTextSize(label, FONT, small_font, thick)
        # Semi-transparent label background
        bx2 = x1 + tw + pad
        by1 = max(0, y1 - th - int(8 * s))
        by2 = by1 + th + int(6 * s)
        _draw_transparent_rect(frame_vis, x1, by1, bx2, by2, (30, 30, 30), alpha=0.6)
        cv2.putText(frame_vis, label, (x1 + int(5 * s), by2 - int(6 * s)), FONT, small_font, color, thick, cv2.LINE_AA)

    # 2) Compact top-left chips (Men, Women, Ratio)
    x = pad
    y = pad
    x = _draw_chip(frame_vis, x, y, f"Men {male_count}", COLORS["M"], WHITE, font_scale, thick, pad_x=int(12*s), pad_y=int(6*s), alpha=0.75)
    x = _draw_chip(frame_vis, x, y, f"Women {female_count}", COLORS["F"], WHITE, font_scale, thick, pad_x=int(12*s), pad_y=int(6*s), alpha=0.75)

    # Optional ratio chip (only if either count > 0)
    if male_count > 0 or female_count > 0:
        ratio_text = f"{male_count}:{female_count}"
        x = _draw_chip(frame_vis, x, y, f"Ratio {ratio_text}", (40, 40, 40), WHITE, font_scale, thick, pad_x=int(12*s), pad_y=int(6*s), alpha=0.6)

    # 3) Risk badge at top-right
    risk_text = f"RISK: {level}  (score={score})"
    rt_w, rt_h = _text_size(risk_text, font_scale, thick)
    rx2 = W - pad
    rx1 = rx2 - (rt_w + int(20 * s))
    ry1 = pad
    ry2 = ry1 + rt_h + int(12 * s)
    _draw_transparent_rect(frame_vis, rx1, ry1, rx2, ry2, RISK_COLORS.get(level, (80, 80, 80)), alpha=0.7)
    cv2.putText(frame_vis, risk_text, (rx1 + int(10 * s), ry2 - int(8 * s)), FONT, font_scale, WHITE, thick, cv2.LINE_AA)

    # 4) Event panel at bottom-left (max 3 lines)
    if events:
        lines = events[:3]
        line_w = 0
        line_h = 0
        for e in lines:
            w, h = _text_size(e, small_font, max(1, thick - 1))
            line_w = max(line_w, w)
            line_h = max(line_h, h)
        panel_w = line_w + int(24 * s)
        row_h = line_h + int(10 * s)
        panel_h = row_h * len(lines) + int(10 * s)

        ex1 = pad
        ey2 = H - pad
        ey1 = ey2 - panel_h
        ex2 = ex1 + panel_w
        _draw_transparent_rect(frame_vis, ex1, ey1, ex2, ey2, (20, 20, 20), alpha=0.55)

        y_cursor = ey1 + row_h - int(4 * s)
        for e in lines:
            cv2.putText(frame_vis, e, (ex1 + int(12 * s), y_cursor), FONT, small_font, WHITE, max(1, thick - 1), cv2.LINE_AA)
            y_cursor += row_h

    # 5) FPS bottom-right (small)
    if fps is not None:
        fps_text = f"{fps:.1f} FPS"
        fw, fh = _text_size(fps_text, small_font, max(1, thick - 1))
        fx2 = W - pad
        fx1 = fx2 - (fw + int(18 * s))
        fy2 = H - pad
        fy1 = fy2 - (fh + int(12 * s))
        _draw_transparent_rect(frame_vis, fx1, fy1, fx2, fy2, (30, 30, 30), alpha=0.45)
        cv2.putText(frame_vis, fps_text, (fx1 + int(9 * s), fy2 - int(8 * s)), FONT, small_font, WHITE, max(1, thick - 1), cv2.LINE_AA)

    # 6) Optional proximity lines (turned off in compact mode to reduce clutter)
    if not compact:
        tids = list(tracks.keys())
        centers = {tid: center_of_box(tracks[tid]["xyxy"]) for tid in tids}
        frame_diag = (W**2 + H**2) ** 0.5
        for fid in [t for t in tids if genders.get(t) == "F"]:
            for mid in [t for t in tids if genders.get(t) == "M"]:
                d = distance(centers[fid], centers[mid])
                if d < 0.06 * frame_diag:
                    cv2.line(frame_vis, centers[fid], centers[mid], (0, 140, 255), 2)

    return frame_vis