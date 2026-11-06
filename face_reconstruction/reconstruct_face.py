"""Convert a facial analysis result into avatar parameters.

This is not a full 3D morphable-face model. Instead, it extracts enough
measurements from a single image to personalize the procedural avatar used by
avatar_viewer.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .face_landmarks import FaceAnalysis, ImageInput, analyze_face



def _rgb_median(pixels: np.ndarray, default: list[int]) -> list[int]:
    if pixels.size == 0:
        return default
    rgb = np.median(pixels.reshape(-1, 3), axis=0)[::-1]  # OpenCV BGR -> RGB
    return [int(np.clip(v, 0, 255)) for v in rgb]



def _clip_crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=image.dtype)
    return image[y1:y2, x1:x2]



def _sample_colors(analysis: FaceAnalysis) -> tuple[list[int], list[int], list[int]]:
    frame = analysis.frame_bgr
    x, y, w, h = analysis.bbox_xywh
    face = frame[y:y + h, x:x + w]
    if face.size == 0:
        return [195, 143, 115], [52, 37, 31], [58, 77, 92]

    # Cheeks for skin tone.
    def region(rel_x1: float, rel_y1: float, rel_x2: float, rel_y2: float) -> np.ndarray:
        return face[int(h * rel_y1):int(h * rel_y2), int(w * rel_x1):int(w * rel_x2)]

    cheeks = np.concatenate([
        region(.12, .50, .30, .67).reshape(-1, 3),
        region(.70, .50, .88, .67).reshape(-1, 3),
    ], axis=0)
    skin = _rgb_median(cheeks, [195, 143, 115])

    # Hair from just above the forehead / upper box.
    forehead = analysis.landmarks_px.get("forehead", (x + w / 2, y + 0.10 * h))
    hair_patch = _clip_crop(frame,
                            int(forehead[0] - 0.28 * w),
                            int(forehead[1] - 0.18 * h),
                            int(forehead[0] + 0.28 * w),
                            int(forehead[1] + 0.03 * h))
    hair = _rgb_median(hair_patch, [52, 37, 31])
    if float(np.mean(hair)) > 200:
        hair = [65, 48, 40]

    # Iris estimate from the eye area when landmarks are available; otherwise default.
    eye_center = analysis.landmarks_px.get("left_eye")
    if eye_center is not None:
        eye_patch = _clip_crop(frame,
                               int(eye_center[0] - 0.06 * w),
                               int(eye_center[1] - 0.04 * h),
                               int(eye_center[0] + 0.06 * w),
                               int(eye_center[1] + 0.04 * h))
        iris = _rgb_median(eye_patch, [58, 77, 92])
        # Avoid a skin-colored iris if the patch is poor.
        if abs(float(np.mean(iris)) - float(np.mean(skin))) < 10:
            iris = [58, 77, 92]
    else:
        iris = [58, 77, 92]
    return skin, hair, iris



def build_avatar_config(image: ImageInput) -> dict:
    """Return a JSON-serializable avatar configuration.

    Output scales are chosen to match the existing Panda3D avatar_viewer.py.
    """
    analysis = analyze_face(image)
    skin, hair, iris = _sample_colors(analysis)
    props = dict(analysis.proportions)
    x, y, w, h = analysis.bbox_xywh

    config = {
        "version": 2,
        "generator": "face_reconstruction.reconstruct_face",
        "detector": analysis.detector,
        "skin_rgb": skin,
        "hair_rgb": hair,
        "eye_rgb": iris,
        "face_width": props.get("face_width", 1.0),
        "face_height": props.get("face_height", 1.0),
        "eye_spacing": props.get("eye_spacing", 1.0),
        "nose_size": props.get("nose_size", 1.0),
        "mouth_width": props.get("mouth_width", 1.0),
        # Extra fields are ignored by the current viewer but help future upgrades.
        "eye_to_mouth": props.get("eye_to_mouth", 1.0),
        "brow_height": props.get("brow_height", 1.0),
        "source_face_box_xywh": [int(x), int(y), int(w), int(h)],
        "note": (
            "Personalized from one image using facial measurements. "
            "This improves resemblance over defaults, but it is still a stylized avatar, "
            "not a full photorealistic 3D reconstruction."
        ),
    }
    return config



def create_avatar_config(image: ImageInput, save_path: str | Path = "output/avatar.json") -> str:
    """Create avatar config JSON on disk and return the file path."""
    config = build_avatar_config(image)
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(path)
