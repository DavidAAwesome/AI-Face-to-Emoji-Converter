"""Face detection (pipeline stage 2).

``haar`` uses the Haar cascade bundled with OpenCV and needs no downloads. ``yunet`` uses
OpenCV's CNN face detector, which is more robust to pose and lighting; it needs the ONNX
file from the OpenCV model zoo (see README).
"""

from pathlib import Path

import cv2
import numpy as np

from .config import ASSETS_DIR

YUNET_PATH = ASSETS_DIR / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def download_yunet(path=YUNET_PATH):
    import urllib.request

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(YUNET_URL, path)
    return path


class FaceDetector:
    def __init__(self, method="haar", min_size=48, score_threshold=0.7, yunet_path=YUNET_PATH):
        self.method = method
        self.min_size = min_size
        if method == "haar":
            if not hasattr(cv2, "CascadeClassifier"):
                raise RuntimeError(
                    f"OpenCV {cv2.__version__} has no Haar cascades. Install opencv-python<5, or run "
                    "`python -m emoji_converter.face_detector --download` and use --detector yunet."
                )
            cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(str(cascade))
            if self._haar.empty():
                raise RuntimeError(f"Could not load Haar cascade from {cascade}")
        elif method == "yunet":
            if not Path(yunet_path).is_file():
                raise FileNotFoundError(
                    f"YuNet model not found at {yunet_path}. Run `python -m emoji_converter.face_detector --download` or use --detector haar."
                )
            self._yunet = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), score_threshold)
        else:
            raise ValueError(f"Unknown detector '{method}' (use 'haar' or 'yunet')")

    def detect(self, frame_bgr):
        """Return a list of (x, y, w, h) boxes, largest first."""
        if self.method == "haar":
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            boxes = self._haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, minSize=(self.min_size, self.min_size)
            )
            boxes = [tuple(int(v) for v in b) for b in boxes]
        else:
            h, w = frame_bgr.shape[:2]
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(frame_bgr)
            boxes = []
            for f in faces if faces is not None else []:
                x, y, bw, bh = (int(v) for v in f[:4])
                if min(bw, bh) >= self.min_size:
                    boxes.append((x, y, bw, bh))
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def crop_face(gray, box, margin=0.1):
    """Square crop around `box`, expanded by `margin`, clipped to the image."""
    x, y, w, h = box
    side = int(max(w, h) * (1 + 2 * margin))
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(cx - side // 2, 0), max(cy - side // 2, 0)
    x1, y1 = min(x0 + side, gray.shape[1]), min(y0 + side, gray.shape[0])
    return np.ascontiguousarray(gray[y0:y1, x0:x1])


if __name__ == "__main__":
    import sys

    if "--download" in sys.argv:
        print(f"Saved YuNet model to {download_yunet()}")
    else:
        print("Usage: python -m emoji_converter.face_detector --download")
