"""Real-time performance measurements (proposal Sec. IV-D).

Reports average face-detection time, classification time, end-to-end latency and FPS
over a webcam / video stream, plus raw model inference time for one or more checkpoints
(for the accuracy-vs-speed comparison).

Examples:
    python benchmark.py --frames 300                          # webcam
    python benchmark.py --video clip.mp4
    python benchmark.py --model-only checkpoints/*.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from emoji_converter.config import DEFAULT_CHECKPOINT, RESULTS_DIR
from emoji_converter.model import load_checkpoint
from emoji_converter.pipeline import ExpressionPipeline


def summarize(values):
    v = np.asarray(values)
    return {"mean_ms": float(v.mean()), "median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95))}


@torch.no_grad()
def model_inference(checkpoint, runs=200, batch=1):
    model, meta = load_checkpoint(checkpoint)
    x = torch.randn(batch, 1, meta["input_size"], meta["input_size"])
    for _ in range(10):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1e3)
    params = sum(p.numel() for p in model.parameters())
    return {"checkpoint": str(checkpoint), "arch": meta["arch"], "params": params,
            "val_acc": meta.get("val_acc"), **summarize(times)}


def stream(args):
    pipe = ExpressionPipeline(args.checkpoint, detector=args.detector)
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
    else:
        cap = cv2.VideoCapture(args.webcam, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        raise SystemExit("Could not open video source")

    detect, classify, total, faces, resolution = [], [], [], 0, None
    t_start = time.perf_counter()
    while len(total) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        resolution = resolution or f"{frame.shape[1]}x{frame.shape[0]}"
        results, t = pipe.process(frame)
        detect.append(t.detect_ms)
        total.append(t.total_ms)
        if results:
            classify.append(t.classify_ms)
            faces += 1
    wall = time.perf_counter() - t_start
    cap.release()
    if not total:
        raise SystemExit("No frames were read")

    report = {
        "frames": len(total), "frames_with_face": faces, "detector": args.detector,
        "resolution": resolution,
        "face_detection": summarize(detect),
        "expression_classification": summarize(classify) if classify else None,
        "pipeline_latency": summarize(total),
        "pipeline_fps": 1000.0 / float(np.mean(total)),
        "observed_fps_including_capture": len(total) / wall,
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--webcam", type=int, default=0)
    p.add_argument("--video", type=Path)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--detector", choices=["haar", "yunet"], default="haar")
    p.add_argument("--model-only", type=Path, nargs="+", help="only time model inference for these checkpoints")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "benchmark.json")
    args = p.parse_args()

    if args.model_only:
        report = [model_inference(c) for c in args.model_only]
        for r in report:
            print(f"{r['arch']:<10s} {r['params'] / 1e6:6.2f}M params | val acc {r['val_acc'] or 0:.4f} | "
                  f"{r['mean_ms']:.2f} ms/face (p95 {r['p95_ms']:.2f})  [{r['checkpoint']}]")
    else:
        report = stream(args)
        print(json.dumps(report, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
