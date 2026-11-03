"""End-to-end pipeline: frame -> faces -> expressions -> emojis (proposal Fig. 1)."""

import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .config import EMOJI_MAP, UNCERTAIN
from .data import preprocess_faces
from .face_detector import FaceDetector, crop_face
from .model import load_checkpoint


@dataclass
class FaceResult:
    box: tuple                 # (x, y, w, h) in frame coordinates
    probs: np.ndarray          # smoothed class probabilities
    label: str                 # predicted expression, or UNCERTAIN below the threshold
    confidence: float
    track_id: int = -1

    @property
    def emoji(self):
        return EMOJI_MAP.get(self.label, EMOJI_MAP[UNCERTAIN])


@dataclass
class Timings:
    detect_ms: float = 0.0
    classify_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class _Track:
    box: tuple
    probs: np.ndarray
    missed: int = 0


class _Tracker:
    """Matches faces across frames by IoU so each face keeps its own smoothed emoji.

    Exponential smoothing of the probabilities stops the emoji flickering between
    classes on consecutive frames.
    """

    def __init__(self, alpha=0.6, iou_threshold=0.3, max_missed=5):
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = {}
        self._next_id = 0

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        return inter / (aw * ah + bw * bh - inter + 1e-9)

    def update(self, boxes, probs):
        assigned, smoothed = [], []
        free = set(self.tracks)
        for box, p in zip(boxes, probs):
            best = max(free, key=lambda t: self._iou(self.tracks[t].box, box), default=None)
            if best is not None and self._iou(self.tracks[best].box, box) >= self.iou_threshold:
                tr = self.tracks[best]
                tr.probs = self.alpha * p + (1 - self.alpha) * tr.probs
                tr.box, tr.missed = box, 0
                free.discard(best)
                tid = best
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = _Track(box, p.copy())
            assigned.append(tid)
            smoothed.append(self.tracks[tid].probs)
        for tid in free:
            self.tracks[tid].missed += 1
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]
        return assigned, smoothed


class ExpressionPipeline:
    def __init__(self, checkpoint, detector="haar", device=None, threshold=0.4,
                 smoothing=0.6, margin=0.1, max_faces=5, tta=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, meta = load_checkpoint(checkpoint, self.device)
        self.classes = meta["classes"]
        self.input_size = meta["input_size"]
        self.detector = FaceDetector(detector)
        self.threshold = threshold
        self.margin = margin
        self.max_faces = max_faces
        self.tta = tta
        self.tracker = _Tracker(alpha=smoothing) if smoothing < 1 else None

    @torch.no_grad()
    def classify_crops(self, gray_crops):
        """Class probabilities (N, C) for grayscale face crops."""
        batch = preprocess_faces(gray_crops, self.input_size).to(self.device)
        probs = torch.softmax(self.model(batch), dim=1)
        if self.tta:  # average with the mirrored face
            probs = (probs + torch.softmax(self.model(torch.flip(batch, dims=[3])), dim=1)) / 2
        return probs.cpu().numpy()

    def label_for(self, probs):
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return (self.classes[idx] if conf >= self.threshold else UNCERTAIN), conf

    def process(self, frame_bgr, track=True):
        """Run the full pipeline on one BGR frame. Returns ([FaceResult], Timings)."""
        t0 = time.perf_counter()
        boxes = self.detector.detect(frame_bgr)[: self.max_faces]
        t1 = time.perf_counter()

        results = []
        if boxes:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            probs = self.classify_crops([crop_face(gray, b, self.margin) for b in boxes])
            ids = [-1] * len(boxes)
            if track and self.tracker is not None:
                ids, probs = self.tracker.update(boxes, list(probs))
            for box, p, tid in zip(boxes, probs, ids):
                label, conf = self.label_for(p)
                results.append(FaceResult(box, np.asarray(p), label, conf, tid))
        elif track and self.tracker is not None:
            self.tracker.update([], [])
        t2 = time.perf_counter()

        return results, Timings((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3)
