"""Create a personalized 3D cartoon-avatar configuration from one image.

This version uses the face_reconstruction package when available. That gives
better proportions than the earlier default-only implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from face_reconstruction import create_avatar_config

ImageInput = Union[str, Path, np.ndarray]


def create_avatar(image: ImageInput, save_path: str | Path = "output/avatar.json") -> str:
    """Infer avatar parameters from one image, save JSON, and return its path."""
    return create_avatar_config(image, save_path)
