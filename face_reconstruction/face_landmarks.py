"""Facial landmark extraction for one image.

If MediaPipe is installed, this returns measurements derived from facial
landmarks. Otherwise it falls back to OpenCV face detection and returns
approximate default proportions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np

try:  # optional dependency
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mp = None

ImageInput = Union[str, Path, np.ndarray]

# MediaPipe FaceMesh landmark indices used below.
LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]
MOUTH = [61, 291, 13, 14]
NOSE = [1, 2, 4, 5, 98, 327]
JAW = [234, 454, 152, 10]
LEFT_BROW = [70, 63, 105]
RIGHT_BROW = [336, 296, 334]


@dataclass(slots=True)
class FaceAnalysis:
    frame_bgr: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    landmarks_px: dict[str, tuple[float, float]] = field(default_factory=dict)
    proportions: dict[str, float] = field(default_factory=dict)
    detector: str = "opencv"
    raw: dict[str, Any] = field(default_factory=dict)



def _load_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        frame = cv2.imread(str(image))
    else:
        frame = np.asarray(image)
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("Provide a readable BGR image (such as cv2.imread output)")
    return frame[:, :, :3].copy()



def _largest_face_bbox(frame: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("OpenCV face-detector data could not be loaded")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))
    if len(faces) == 0:
        raise ValueError("No frontal face found. Try a clear, front-facing photo.")
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(w), int(h)



def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))



def _mean_pt(points: list[tuple[float, float]]) -> tuple[float, float]:
    arr = np.asarray(points, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())



def _crop_landmark_bounds(pts: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1 = np.min(pts[:, :2], axis=0)
    x2, y2 = np.max(pts[:, :2], axis=0)
    pad_x = 0.12 * max(1.0, x2 - x1)
    pad_y = 0.18 * max(1.0, y2 - y1)
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(w, int(x2 + pad_x))
    bottom = min(h, int(y2 + pad_y))
    return left, top, max(1, right - left), max(1, bottom - top)



def _mediapipe_analysis(frame: np.ndarray) -> FaceAnalysis | None:
    if mp is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    except Exception:
        return None

    with face_mesh as mesh:
        results = mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    h, w = frame.shape[:2]
    lm = results.multi_face_landmarks[0].landmark
    pts = np.array([(p.x * w, p.y * h, p.z * w) for p in lm], dtype=np.float32)

    left_eye_pts = [(pts[i][0], pts[i][1]) for i in LEFT_EYE]
    right_eye_pts = [(pts[i][0], pts[i][1]) for i in RIGHT_EYE]
    mouth_pts = [(pts[i][0], pts[i][1]) for i in MOUTH]
    nose_pts = [(pts[i][0], pts[i][1]) for i in NOSE]
    jaw_pts = [(pts[i][0], pts[i][1]) for i in JAW]

    left_eye = _mean_pt(left_eye_pts)
    right_eye = _mean_pt(right_eye_pts)
    mouth_center = _mean_pt(mouth_pts)
    nose_center = _mean_pt(nose_pts)
    left_brow = _mean_pt([(pts[i][0], pts[i][1]) for i in LEFT_BROW])
    right_brow = _mean_pt([(pts[i][0], pts[i][1]) for i in RIGHT_BROW])
    chin = (float(pts[152][0]), float(pts[152][1]))
    forehead = (float(pts[10][0]), float(pts[10][1]))
    jaw_left = (float(pts[234][0]), float(pts[234][1]))
    jaw_right = (float(pts[454][0]), float(pts[454][1]))
    mouth_left = (float(pts[61][0]), float(pts[61][1]))
    mouth_right = (float(pts[291][0]), float(pts[291][1]))
    nose_left = (float(pts[98][0]), float(pts[98][1]))
    nose_right = (float(pts[327][0]), float(pts[327][1]))
    nose_tip = (float(pts[4][0]), float(pts[4][1]))

    bbox = _crop_landmark_bounds(pts, w, h)
    face_width_px = max(_distance(jaw_left, jaw_right), 1.0)
    face_height_px = max(_distance(forehead, chin), 1.0)
    eye_spacing_px = _distance(left_eye, right_eye)
    mouth_width_px = _distance(mouth_left, mouth_right)
    nose_width_px = _distance(nose_left, nose_right)
    eye_to_mouth_px = _distance(_mean_pt([left_eye, right_eye]), mouth_center)
    brow_height_px = ((_distance(left_eye, left_brow) + _distance(right_eye, right_brow)) / 2.0)

    proportions = {
        # Scaled around 1.0 so the viewer can directly use them.
        "face_width": float(np.clip((face_width_px / face_height_px) / 0.82, 0.75, 1.30)),
        "face_height": float(np.clip((face_height_px / face_width_px) / 1.18, 0.80, 1.25)),
        "eye_spacing": float(np.clip((eye_spacing_px / face_width_px) / 0.42, 0.70, 1.35)),
        "nose_size": float(np.clip((nose_width_px / face_width_px) / 0.16, 0.70, 1.40)),
        "mouth_width": float(np.clip((mouth_width_px / face_width_px) / 0.36, 0.65, 1.30)),
        "eye_to_mouth": float(np.clip((eye_to_mouth_px / face_height_px) / 0.40, 0.75, 1.30)),
        "brow_height": float(np.clip((brow_height_px / face_height_px) / 0.10, 0.70, 1.40)),
    }
    landmarks = {
        "left_eye": left_eye,
        "right_eye": right_eye,
        "left_brow": left_brow,
        "right_brow": right_brow,
        "nose_tip": nose_tip,
        "mouth_center": mouth_center,
        "mouth_left": mouth_left,
        "mouth_right": mouth_right,
        "jaw_left": jaw_left,
        "jaw_right": jaw_right,
        "chin": chin,
        "forehead": forehead,
    }
    return FaceAnalysis(
        frame_bgr=frame,
        bbox_xywh=bbox,
        landmarks_px=landmarks,
        proportions=proportions,
        detector="mediapipe",
        raw={"landmark_count": len(lm)},
    )



def _fallback_analysis(frame: np.ndarray) -> FaceAnalysis:
    bbox = _largest_face_bbox(frame)
    x, y, w, h = bbox
    # Approximate landmark positions inside the bounding box.
    left_eye = (x + 0.33 * w, y + 0.40 * h)
    right_eye = (x + 0.67 * w, y + 0.40 * h)
    mouth_left = (x + 0.37 * w, y + 0.74 * h)
    mouth_right = (x + 0.63 * w, y + 0.74 * h)
    mouth_center = (x + 0.50 * w, y + 0.74 * h)
    nose_tip = (x + 0.50 * w, y + 0.58 * h)
    landmarks = {
        "left_eye": left_eye,
        "right_eye": right_eye,
        "left_brow": (x + 0.33 * w, y + 0.31 * h),
        "right_brow": (x + 0.67 * w, y + 0.31 * h),
        "nose_tip": nose_tip,
        "mouth_center": mouth_center,
        "mouth_left": mouth_left,
        "mouth_right": mouth_right,
        "jaw_left": (x + 0.10 * w, y + 0.78 * h),
        "jaw_right": (x + 0.90 * w, y + 0.78 * h),
        "chin": (x + 0.50 * w, y + 0.98 * h),
        "forehead": (x + 0.50 * w, y + 0.05 * h),
    }
    # Rough neutral defaults.
    proportions = {
        "face_width": 1.0,
        "face_height": 1.0,
        "eye_spacing": 1.0,
        "nose_size": 1.0,
        "mouth_width": 1.0,
        "eye_to_mouth": 1.0,
        "brow_height": 1.0,
    }
    return FaceAnalysis(frame_bgr=frame, bbox_xywh=bbox, landmarks_px=landmarks,
                        proportions=proportions, detector="opencv")



def analyze_face(image: ImageInput) -> FaceAnalysis:
    """Return facial measurements for one image.

    The function prefers MediaPipe when available because it yields much better
    feature measurements than OpenCV alone.
    """
    frame = _load_image(image)
    analysis = _mediapipe_analysis(frame)
    if analysis is not None:
        return analysis
    return _fallback_analysis(frame)
