"""Smoke tests that need no dataset or trained model.

    python -m pytest tests      (or)      python tests/test_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emoji_converter.config import EMOJI_MAP, FER2013_CLASSES, UNCERTAIN  # noqa: E402
from emoji_converter.data import _remap, parse_merge, preprocess_faces  # noqa: E402
from emoji_converter.emoji_assets import EmojiLibrary  # noqa: E402
from emoji_converter.face_detector import FaceDetector, crop_face  # noqa: E402
from emoji_converter.model import ARCHITECTURES, build_model, load_checkpoint, save_checkpoint  # noqa: E402
from emoji_converter.pipeline import ExpressionPipeline, FaceResult, _Tracker  # noqa: E402
from emoji_converter.visualize import overlay_bgra, render  # noqa: E402


def test_every_class_has_an_emoji_and_image():
    lib = EmojiLibrary()
    for name in FER2013_CLASSES + [UNCERTAIN]:
        assert name in EMOJI_MAP
        img = lib.get(name)
        assert img.ndim == 3 and img.shape[2] == 4


def test_models_output_shape():
    for arch, (_, size) in ARCHITECTURES.items():
        model = build_model(arch, 7, pretrained=False).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 1, size, size))
        assert out.shape == (2, 7), arch


def test_merge_classes():
    labels, classes = _remap(["angry", "disgust", "happy"], parse_merge(["disgust=angry"]))
    assert classes == ["angry", "happy"]
    assert labels.tolist() == [0, 0, 1]


def test_preprocess_and_crop():
    gray = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    crop = crop_face(gray, (300, 200, 60, 60))  # partly outside the frame
    assert crop.size > 0
    batch = preprocess_faces([crop, gray], 48)
    assert batch.shape == (2, 1, 48, 48)
    assert -1.0 <= batch.min() and batch.max() <= 1.0


def test_overlay_clips_at_frame_edges():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    emoji = np.full((64, 64, 4), 255, dtype=np.uint8)
    overlay_bgra(frame, emoji, -20, 70, 50)
    assert frame[99, 0].tolist() == [255, 255, 255]
    overlay_bgra(frame, emoji, 500, 500, 50)  # fully outside: no error


def test_tracker_keeps_identity_and_smooths():
    tr = _Tracker(alpha=0.5)
    ids1, _ = tr.update([(10, 10, 50, 50)], [np.array([1.0, 0.0])])
    ids2, probs = tr.update([(12, 11, 50, 50)], [np.array([0.0, 1.0])])
    assert ids1 == ids2
    assert np.allclose(probs[0], [0.5, 0.5])


def test_end_to_end_with_untrained_model(tmp_path=None):
    tmp = Path(tmp_path or Path(__file__).parent / "_tmp")
    tmp.mkdir(exist_ok=True)
    ckpt = tmp / "untrained.pt"
    save_checkpoint(ckpt, build_model("small_cnn", 7), "small_cnn", FER2013_CLASSES, 48)
    model, meta = load_checkpoint(ckpt)
    assert meta["classes"] == FER2013_CLASSES

    pipe = ExpressionPipeline(ckpt, threshold=0.0)
    probs = pipe.classify_crops([np.zeros((60, 60), np.uint8)])
    assert probs.shape == (1, 7) and np.isclose(probs.sum(), 1.0, atol=1e-5)

    frame = np.full((240, 320, 3), 127, dtype=np.uint8)
    results, timings = pipe.process(frame)
    assert results == [] and timings.total_ms >= 0
    fake = FaceResult((100, 60, 80, 80), probs[0], *pipe.label_for(probs[0]))
    render(frame, [fake], EmojiLibrary(), pipe.classes)
    assert not np.all(frame == 127)


def test_haar_detector_loads():
    assert FaceDetector("haar").detect(np.zeros((120, 160, 3), np.uint8)) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
