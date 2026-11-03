"""Emoji image assets (pipeline stage 6).

The proposal notes that Unicode emoji render differently on every platform (Sec. III-D,
[6]), so the overlay uses PNG images instead. Default images are drawn procedurally here
and written to ``assets/emojis/``. Dropping your own ``<expression>.png`` (RGBA) into an
emoji folder replaces the default for that expression, which makes custom emoji packs
possible via ``--emoji-dir``.
"""

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import EMOJI_DIR, EMOJI_MAP

S = 1024            # drawing canvas (supersampled for anti-aliasing)
OUT = 256           # saved asset size
DARK = (70, 42, 20, 255)
WHITE = (255, 255, 255, 255)
LEFT_X, RIGHT_X = 360, 664


def _face(top=(255, 222, 90), bottom=(250, 170, 40)):
    grad = np.zeros((S, S, 4), dtype=np.uint8)
    t = np.linspace(0, 1, S)[:, None]
    for ch in range(3):
        grad[..., ch] = (top[ch] * (1 - t) + bottom[ch] * t).astype(np.uint8)
    grad[..., 3] = 255
    img = Image.fromarray(grad, "RGBA")

    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse((32, 32, S - 32, S - 32), fill=255)
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    canvas.paste(img, (0, 0), mask)
    draw = ImageDraw.Draw(canvas)
    outline = tuple(int(c * 0.75) for c in bottom) + (255,)
    draw.ellipse((32, 32, S - 32, S - 32), outline=outline, width=22)
    return canvas, draw


def _eyes(draw, y=430, w=80, h=120, highlight=True):
    for x in (LEFT_X, RIGHT_X):
        draw.ellipse((x - w // 2, y - h // 2, x + w // 2, y + h // 2), fill=DARK)
        if highlight:
            r = max(w // 6, 8)
            draw.ellipse((x - w // 4 - r, y - h // 4 - r, x - w // 4 + r, y - h // 4 + r), fill=WHITE)


def _brows(draw, outer_y, inner_y, width=36, span=(270, 440)):
    x0, x1 = span
    draw.line([(x0, outer_y), (x1, inner_y)], fill=DARK, width=width, joint="curve")
    draw.line([(S - x0, outer_y), (S - x1, inner_y)], fill=DARK, width=width, joint="curve")


def _round_caps(draw, points, width):
    for x, y in (points[0], points[-1]):
        draw.ellipse((x - width / 2, y - width / 2, x + width / 2, y + width / 2), fill=DARK)


def _arc_mouth(draw, box, start, end, width=32):
    draw.arc(box, start, end, fill=DARK, width=width)
    x0, y0, x1, y1 = box
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    ends = []
    for a in (start, end):
        rad = math.radians(a)
        ends.append((cx + (rx - width / 2) * math.cos(rad), cy + (ry - width / 2) * math.sin(rad)))
    _round_caps(draw, ends, width)


def _happy():
    img, draw = _face()
    _eyes(draw, y=400, w=90, h=150)
    box = (240, 420, 784, 860)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).chord(box, 0, 180, fill=255)
    mouth = Image.new("RGBA", (S, S), (90, 30, 20, 255))
    md = ImageDraw.Draw(mouth)
    md.rectangle((0, 630, S, 700), fill=WHITE)
    md.ellipse((380, 760, 644, 940), fill=(230, 80, 90, 255))
    img.paste(mouth, (0, 0), mask)
    draw.chord(box, 0, 180, outline=DARK, width=16)
    return img


def _sad():
    img, draw = _face()
    _brows(draw, outer_y=330, inner_y=280, width=30)
    _eyes(draw, y=450)
    _arc_mouth(draw, (350, 660, 674, 900), 205, 335)
    draw.ellipse((292, 560, 372, 660), fill=(90, 170, 240, 255))
    draw.polygon([(296, 595), (332, 505), (368, 595)], fill=(90, 170, 240, 255))
    return img


def _angry():
    img, draw = _face(top=(225, 70, 40), bottom=(250, 165, 60))
    _brows(draw, outer_y=300, inner_y=380, width=44)
    _eyes(draw, y=460, w=80, h=90)
    _arc_mouth(draw, (350, 660, 674, 860), 205, 335, width=36)
    return img


def _surprise():
    img, draw = _face()
    for x in (LEFT_X, RIGHT_X):
        draw.arc((x - 90, 230, x + 90, 340), 200, 340, fill=DARK, width=28)
    _eyes(draw, y=440, w=100, h=140)
    draw.ellipse((432, 620, 592, 840), fill=(90, 30, 20, 255), outline=DARK, width=14)
    return img


def _fear():
    img, draw = _face(top=(140, 170, 225), bottom=(255, 205, 70))
    _brows(draw, outer_y=310, inner_y=250, width=30)
    for x in (LEFT_X, RIGHT_X):
        draw.ellipse((x - 75, 360, x + 75, 530), fill=WHITE, outline=DARK, width=12)
        draw.ellipse((x - 34, 420, x + 34, 488), fill=DARK)
    draw.rounded_rectangle((310, 650, 714, 800), radius=60, fill=WHITE, outline=DARK, width=18)
    draw.line([(320, 725), (704, 725)], fill=DARK, width=12)
    for x in (410, 512, 614):
        draw.line([(x, 660), (x, 790)], fill=DARK, width=10)
    return img


def _disgust():
    img, draw = _face(top=(150, 200, 90), bottom=(245, 205, 70))
    for x, sign in ((LEFT_X, 1), (RIGHT_X, -1)):
        pts = [(x - 60 * sign, 380), (x + 50 * sign, 430), (x - 60 * sign, 480)]
        draw.line(pts, fill=DARK, width=30, joint="curve")
        _round_caps(draw, pts, 30)
    pts = [(330 + i * 4, 730 + 34 * math.sin(i * 4 / 364 * 3 * math.pi)) for i in range(92)]
    draw.line(pts, fill=DARK, width=30, joint="curve")
    _round_caps(draw, pts, 30)
    return img


def _neutral():
    img, draw = _face()
    _eyes(draw, y=430)
    draw.rounded_rectangle((350, 690, 674, 730), radius=20, fill=DARK)
    return img


def _uncertain():
    img, draw = _face()
    draw.arc((RIGHT_X - 90, 230, RIGHT_X + 90, 340), 200, 340, fill=DARK, width=28)
    pts = [(290, 340), (430, 340)]
    draw.line(pts, fill=DARK, width=28)
    _round_caps(draw, pts, 28)
    _eyes(draw, y=440)
    pts = [(380, 730), (640, 690)]
    draw.line(pts, fill=DARK, width=32)
    _round_caps(draw, pts, 32)
    try:
        font = ImageFont.load_default(size=320)
        draw.text((760, 40), "?", font=font, fill=(90, 90, 110, 255), stroke_width=10, stroke_fill=WHITE)
    except TypeError:  # Pillow < 10.1 has no sized default font
        pass
    return img


DRAWERS = {
    "angry": _angry, "disgust": _disgust, "fear": _fear, "happy": _happy,
    "sad": _sad, "surprise": _surprise, "neutral": _neutral, "uncertain": _uncertain,
}


def render_emoji(name, size=OUT):
    return DRAWERS[name]().resize((size, size), Image.LANCZOS)


def generate_emoji_assets(out_dir=EMOJI_DIR, overwrite=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in DRAWERS:
        path = out_dir / f"{name}.png"
        if overwrite or not path.exists():
            render_emoji(name).save(path)
            written.append(path)
    return written


class EmojiLibrary:
    """Loads BGRA emoji images, falling back to the procedurally drawn defaults."""

    def __init__(self, emoji_dir=EMOJI_DIR):
        self.emoji_dir = Path(emoji_dir)
        self._cache = {}

    def get(self, name):
        if name not in self._cache:
            path = self.emoji_dir / f"{name}.png"
            if path.exists():
                rgba = np.asarray(Image.open(path).convert("RGBA"))
            elif name in DRAWERS:
                rgba = np.asarray(render_emoji(name))
            else:
                raise KeyError(f"No emoji image for '{name}' in {self.emoji_dir}")
            self._cache[name] = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        return self._cache[name]

    def names(self):
        return list(EMOJI_MAP)
