"""Drawing helpers: emoji overlay, boxes, labels, probability bars and FPS."""

import cv2
import numpy as np

from .config import CLASS_COLORS

OVERLAY_MODES = ("mask", "side", "off")
SIDEBAR_W = 300


def overlay_bgra(frame, image_bgra, x, y, size):
    """Alpha-blend a BGRA image onto `frame` with its top-left corner at (x, y)."""
    img = cv2.resize(image_bgra, (size, size), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + size, w), min(y + size, h)
    if x0 >= x1 or y0 >= y1:
        return frame
    crop = img[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = crop[..., 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (alpha * crop[..., :3] + (1 - alpha) * roi).astype(np.uint8)
    return frame


def _text(frame, text, org, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_face(frame, result, emojis, mode="mask", emoji_scale=1.3, show_box=True):
    x, y, w, h = result.box
    color = CLASS_COLORS.get(result.label, (255, 255, 255))
    emoji = emojis.get(result.label)

    if mode == "mask":
        size = int(max(w, h) * emoji_scale)
        overlay_bgra(frame, emoji, x + w // 2 - size // 2, y + h // 2 - size // 2, size)
    elif mode == "side":
        size = max(int(h * 0.6), 40)
        overlay_bgra(frame, emoji, x + w + 8, y, size)

    if show_box and mode != "mask":
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    _text(frame, f"{result.label} {result.confidence * 100:.0f}%", (x, max(y - 10, 18)), 0.7, color, 2)
    return frame


def build_sidebar(height, result, classes, emojis, width=SIDEBAR_W):
    """Panel beside the video: emoji, expression, confidence and per-class probabilities
    for the primary (largest) face."""
    panel = np.full((height, width, 3), 32, dtype=np.uint8)
    if result is None:
        _text(panel, "no face detected", (16, 40), 0.6, (200, 200, 200))
        return panel
    size = min(width - 60, max(height // 3, 60))
    overlay_bgra(panel, emojis.get(result.label), (width - size) // 2, 16, size)
    color = CLASS_COLORS.get(result.label, (255, 255, 255))
    y = 16 + size + 32
    _text(panel, result.label.upper(), (16, y), 0.8, color, 2)
    _text(panel, f"{result.confidence * 100:.0f}% confident", (16, y + 26), 0.5)
    bar_w = width - 140
    for i, (name, p) in enumerate(zip(classes, result.probs)):
        by = y + 50 + i * 22
        if by + 14 > height - 24:
            break
        cv2.rectangle(panel, (84, by), (84 + int(bar_w * p), by + 14), CLASS_COLORS.get(name, (200, 200, 200)), -1)
        _text(panel, name, (16, by + 12), 0.45)
        _text(panel, f"{p * 100:3.0f}%", (90 + bar_w, by + 12), 0.4)
    return panel


def draw_hud(frame, fps, timings, threshold, mode):
    h = frame.shape[0]
    _text(frame, f"FPS {fps:5.1f} | detect {timings.detect_ms:5.1f} ms | classify {timings.classify_ms:5.1f} ms",
          (10, 22), 0.55, (0, 255, 0))
    _text(frame, f"threshold {threshold:.2f} | overlay: {mode} | [q]uit [m]ode [+/-]thr [p]anel [s]napshot",
          (10, h - 12), 0.45, (220, 220, 220))
    return frame


def render(frame, results, emojis, classes, mode="mask", show_panel=True):
    """Draw results on `frame` in place; returns the frame with the sidebar attached."""
    for r in results:
        draw_face(frame, r, emojis, mode)
    if not show_panel:
        return frame
    sidebar = build_sidebar(frame.shape[0], results[0] if results else None, classes, emojis)
    return np.hstack([frame, sidebar])
