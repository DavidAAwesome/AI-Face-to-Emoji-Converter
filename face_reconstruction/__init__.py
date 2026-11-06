"""Lightweight facial landmark and avatar-parameter estimation helpers.

This package is designed for the user's existing desktop Python project.
It can work from a single photo or a first-time webcam frame.

Best quality: install `mediapipe` for dense facial landmarks.
Fallback: OpenCV face detection with approximate default proportions.
"""

from .face_landmarks import FaceAnalysis, analyze_face
from .reconstruct_face import build_avatar_config, create_avatar_config

__all__ = [
    "FaceAnalysis",
    "analyze_face",
    "build_avatar_config",
    "create_avatar_config",
]
