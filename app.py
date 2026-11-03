"""Real-time facial-expression-to-emoji demo.

Examples:
    python app.py                          # webcam 0
    python app.py --webcam 1 --mode side   # second camera, emoji beside the face
    python app.py --video clip.mp4 --save results/clip_emoji.mp4
    python app.py --image photo.jpg group.png --save results/demo

Keys (live window):
    q / Esc   quit
    m         cycle overlay mode: mask (emoji over face) -> side -> off
    + / -     raise / lower the confidence threshold
    p         toggle the side panel (emoji, confidence, probabilities)
    s         save a snapshot to results/snapshots/
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

from emoji_converter.config import DEFAULT_CHECKPOINT, EMOJI_DIR, RESULTS_DIR
from emoji_converter.emoji_assets import EmojiLibrary
from emoji_converter.pipeline import ExpressionPipeline
from emoji_converter.visualize import OVERLAY_MODES, draw_hud, render

WINDOW = "Face to Emoji"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--webcam", type=int, default=0, help="camera index (default 0)")
    src.add_argument("--video", type=Path, help="video file to process")
    src.add_argument("--image", type=Path, nargs="+", help="one or more still images")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--detector", choices=["haar", "yunet"], default="haar")
    p.add_argument("--threshold", type=float, default=0.4, help="below this confidence show the 'uncertain' emoji")
    p.add_argument("--smoothing", type=float, default=0.6,
                   help="weight of the newest frame in the probability moving average (1 = no smoothing)")
    p.add_argument("--mode", choices=OVERLAY_MODES, default="mask")
    p.add_argument("--emoji-dir", type=Path, default=EMOJI_DIR, help="folder of <expression>.png emoji images")
    p.add_argument("--tta", action="store_true", help="also classify the mirrored face (slower, more accurate)")
    p.add_argument("--width", type=int, default=960, help="requested webcam width")
    p.add_argument("--save", type=Path, help="output video file (video mode) or folder (image mode)")
    p.add_argument("--no-display", action="store_true", help="process without opening a window")
    return p.parse_args()


def run_images(pipe, emojis, args):
    out_dir = args.save or RESULTS_DIR / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.image:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Could not read {path}", file=sys.stderr)
            continue
        results, t = pipe.process(frame, track=False)
        print(f"{path.name}: {len(results)} face(s), {t.total_ms:.1f} ms")
        for i, r in enumerate(results):
            top = sorted(zip(pipe.classes, r.probs), key=lambda kv: -kv[1])[:3]
            print(f"  face {i}: {r.label:<9s} {r.emoji}  conf {r.confidence:.2f}  "
                  f"(top-3: {', '.join(f'{c} {p:.2f}' for c, p in top)})")
        frame = render(frame, results, emojis, pipe.classes, args.mode)
        out = out_dir / f"{path.stem}_emoji.png"
        cv2.imwrite(str(out), frame)
        print(f"  -> {out}")
        if not args.no_display:
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


def run_stream(pipe, emojis, args):
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
    else:
        cap = cv2.VideoCapture(args.webcam, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(args.webcam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {'video ' + str(args.video) if args.video else f'webcam {args.webcam}'}")

    writer = None
    mode, show_panel, fps = args.mode, True, 0.0
    snap_dir = RESULTS_DIR / "snapshots"
    last = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.video:
            frame = cv2.flip(frame, 1)  # mirror view feels natural for a webcam

        results, timings = pipe.process(frame)
        frame = render(frame, results, emojis, pipe.classes, mode, show_panel)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
        last = now
        draw_hud(frame, fps, timings, pipe.threshold, mode)

        if args.save:
            if writer is None:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
                writer_size = (frame.shape[1], frame.shape[0])
                writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, writer_size)
            # Toggling the side panel changes the width; keep the output size fixed.
            writer.write(frame if frame.shape[1::-1] == writer_size else cv2.resize(frame, writer_size))
        if args.no_display:
            continue

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("m"):
            mode = OVERLAY_MODES[(OVERLAY_MODES.index(mode) + 1) % len(OVERLAY_MODES)]
        elif key in (ord("+"), ord("=")):
            pipe.threshold = min(pipe.threshold + 0.05, 0.95)
        elif key in (ord("-"), ord("_")):
            pipe.threshold = max(pipe.threshold - 0.05, 0.0)
        elif key == ord("p"):
            show_panel = not show_panel
        elif key == ord("s"):
            snap_dir.mkdir(parents=True, exist_ok=True)
            path = snap_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(path), frame)
            print(f"Saved {path}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved {args.save}")
    cv2.destroyAllWindows()


def main():
    # Windows consoles default to cp1252, which cannot print emoji characters.
    sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint {args.checkpoint} not found. Train one first:\n"
                         f"  python train.py --data data/fer2013.csv")
    pipe = ExpressionPipeline(args.checkpoint, detector=args.detector, threshold=args.threshold,
                              smoothing=args.smoothing, tta=args.tta)
    emojis = EmojiLibrary(args.emoji_dir)
    print(f"Loaded {args.checkpoint} | classes {pipe.classes} | device {pipe.device}")
    if args.image:
        run_images(pipe, emojis, args)
    else:
        run_stream(pipe, emojis, args)


if __name__ == "__main__":
    main()
