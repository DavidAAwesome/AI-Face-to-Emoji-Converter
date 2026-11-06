"""Map emotion-model output to a compact set of avatar controls."""

EMOTION_PRESETS = {
    "neutral": {"smile": 0.0, "frown": 0.0, "mouth_open": 0.0, "brow_raise": 0.0, "brow_down": 0.0},
    "happy": {"smile": 1.0, "mouth_open": 0.15},
    "sad": {"frown": 0.8, "brow_raise": 0.25},
    "angry": {"frown": 0.55, "brow_down": 1.0},
    "surprise": {"mouth_open": 1.0, "brow_raise": 1.0},
    "surprised": {"mouth_open": 1.0, "brow_raise": 1.0},
    "fear": {"mouth_open": 0.65, "brow_raise": 0.7},
    "disgust": {"frown": 0.6, "brow_down": 0.35},
}


def get_avatar_expression(emotion: str) -> dict[str, float]:
    """Return normalized, renderer-independent animation controls."""
    values = dict(EMOTION_PRESETS["neutral"])
    values.update(EMOTION_PRESETS.get(str(emotion).strip().lower(), {}))
    return values
