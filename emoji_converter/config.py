"""Shared constants: expression classes, emoji mapping, and default paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
EMOJI_DIR = ASSETS_DIR / "emojis"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best_model.pt"

# FER2013 label order (column "emotion" in fer2013.csv).
FER2013_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Label shown when the classifier is not confident enough (proposal Sec. III-D).
UNCERTAIN = "uncertain"

# Expression -> emoji (Table I of the proposal). The Unicode character is used for
# console output; the on-screen overlay uses the PNG assets in assets/emojis/ so the
# look is identical on every platform.
EMOJI_MAP = {
    "angry": "\U0001F620",     # angry face
    "disgust": "\U0001F922",   # nauseated / disgusted face
    "fear": "\U0001F628",      # fearful face
    "happy": "\U0001F600",     # grinning face
    "sad": "\U0001F641",       # slightly frowning face
    "surprise": "\U0001F62E",  # face with open mouth
    "neutral": "\U0001F610",   # neutral face
    UNCERTAIN: "\U0001F914",   # thinking face
}

# BGR colours used for boxes and probability bars in the OpenCV UI.
CLASS_COLORS = {
    "angry": (40, 40, 220),
    "disgust": (60, 160, 60),
    "fear": (170, 80, 150),
    "happy": (0, 200, 255),
    "sad": (200, 120, 40),
    "surprise": (0, 140, 255),
    "neutral": (180, 180, 180),
    UNCERTAIN: (120, 120, 120),
}
