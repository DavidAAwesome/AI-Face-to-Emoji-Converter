"""Evaluate a trained classifier on the held-out test set (proposal Sec. IV-C and IV-E).

Writes to results/<checkpoint name>/:
    metrics.json              accuracy, per-class precision / recall / F1, macro & weighted averages
    confusion_matrix.png      raw counts and row-normalised
    threshold_sweep.png/json  coverage vs. accuracy for each confidence threshold
    robustness.png/json       accuracy under lighting, blur, noise, rotation, occlusion, distance
    training_curves.png       loss / accuracy per epoch (if the .history.json exists)

Examples:
    python evaluate.py --data data/fer2013.csv
    python evaluate.py --data data/fer2013 --checkpoint checkpoints/resnet18.pt --tta
    python evaluate.py --history checkpoints/best_model.history.json
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402

from emoji_converter.config import DEFAULT_CHECKPOINT, RESULTS_DIR  # noqa: E402
from emoji_converter.data import load_splits, preprocess_faces  # noqa: E402
from emoji_converter.model import load_checkpoint  # noqa: E402

THRESHOLDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---- perturbations for robustness experiments (all operate on uint8 grayscale) ----

def _brightness(delta):
    return lambda im: np.clip(im.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def _contrast(factor):
    return lambda im: np.clip((im.astype(np.float32) - 128) * factor + 128, 0, 255).astype(np.uint8)


def _blur(k):
    return lambda im: cv2.GaussianBlur(im, (k, k), 0)


def _noise(sigma, rng=np.random.default_rng(0)):
    return lambda im: np.clip(im + rng.normal(0, sigma, im.shape), 0, 255).astype(np.uint8)


def _rotate(deg):
    def f(im):
        h, w = im.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(im, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    return f


def _occlude(region):
    def f(im):
        im = im.copy()
        h, w = im.shape
        if region == "eyes":
            im[int(h * 0.2):int(h * 0.5), :] = 0      # e.g. sunglasses
        elif region == "mouth":
            im[int(h * 0.6):, :] = 0                   # e.g. hand / mask
        return im
    return f


def _distance(scale):
    """Simulate a far-away face: downsample, then upsample back to the original size."""
    def f(im):
        h, w = im.shape
        small = cv2.resize(im, (max(int(w * scale), 4), max(int(h * scale), 4)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return f


PERTURBATIONS = {
    "clean": lambda im: im,
    "dark (-80)": _brightness(-80),
    "bright (+80)": _brightness(80),
    "low contrast": _contrast(0.4),
    "blur k=5": _blur(5),
    "noise s=20": _noise(20),
    "rotate 15deg": _rotate(15),
    "rotate 30deg": _rotate(30),
    "occlude eyes": _occlude("eyes"),
    "occlude mouth": _occlude("mouth"),
    "far (1/3 res)": _distance(1 / 3),
    "mirrored": lambda im: np.ascontiguousarray(im[:, ::-1]),
}


@torch.no_grad()
def predict(model, images, input_size, device, tta=False, batch_size=256):
    out = []
    for i in range(0, len(images), batch_size):
        x = preprocess_faces(list(images[i:i + batch_size]), input_size).to(device)
        p = torch.softmax(model(x), 1)
        if tta:
            p = (p + torch.softmax(model(torch.flip(x, dims=[3])), 1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()


def plot_confusion(cm, classes, path):
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title, fmt in ((axes[0], cm, "Counts", "d"), (axes[1], cm_norm, "Row-normalised (recall)", ".2f")):
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        thresh = mat.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def threshold_sweep(probs, labels, path_stem):
    conf, pred = probs.max(1), probs.argmax(1)
    rows = []
    for t in THRESHOLDS:
        keep = conf >= t
        rows.append({
            "threshold": t,
            "coverage": float(keep.mean()),
            "accuracy_on_confident": float((pred[keep] == labels[keep]).mean()) if keep.any() else None,
        })
    Path(f"{path_stem}.json").write_text(json.dumps(rows, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(THRESHOLDS, [r["coverage"] for r in rows], "o-", label="coverage (faces given an emoji)")
    ax.plot(THRESHOLDS, [r["accuracy_on_confident"] or np.nan for r in rows], "s-", label="accuracy on those faces")
    ax.set_xlabel("confidence threshold")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Confidence threshold trade-off")
    fig.tight_layout()
    fig.savefig(f"{path_stem}.png", dpi=130)
    plt.close(fig)
    return rows


def robustness(model, images, labels, input_size, device, tta, path_stem):
    rows = {}
    for name, fn in PERTURBATIONS.items():
        perturbed = np.stack([fn(im) for im in images])
        acc = accuracy_score(labels, predict(model, perturbed, input_size, device, tta).argmax(1))
        rows[name] = float(acc)
        print(f"  {name:<15s} {acc:.4f}")
    Path(f"{path_stem}.json").write_text(json.dumps(rows, indent=2))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = list(rows)
    ax.bar(names, [rows[n] for n in names], color=["#4c72b0"] + ["#dd8452"] * (len(names) - 1))
    ax.axhline(rows["clean"], color="#4c72b0", ls="--", lw=1)
    ax.set_ylabel("test accuracy")
    ax.set_title("Robustness to image conditions")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(f"{path_stem}.png", dpi=130)
    plt.close(fig)
    return rows


def plot_history(history_path, out_path):
    hist = json.loads(Path(history_path).read_text())
    ep = [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(ep, [h["train_loss"] for h in hist], label="train")
    a1.plot(ep, [h["val_loss"] for h in hist], label="val")
    a1.set_title("Loss")
    a2.plot(ep, [h["train_acc"] for h in hist], label="train")
    a2.plot(ep, [h["val_acc"] for h in hist], label="val")
    a2.set_title("Accuracy")
    for a in (a1, a2):
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Training curves -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="fer2013.csv or folder with train/ and test/")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--tta", action="store_true", help="average predictions with the mirrored image")
    p.add_argument("--no-robustness", action="store_true", help="skip the robustness experiments")
    p.add_argument("--history", type=Path, help="only plot a training .history.json")
    p.add_argument("--out", type=Path, help="output folder (default results/<checkpoint name>)")
    args = p.parse_args()

    out_dir = args.out or RESULTS_DIR / args.checkpoint.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.history:
        plot_history(args.history, out_dir / "training_curves.png")
        return
    if not args.data:
        p.error("--data is required unless --history is given")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(args.checkpoint, device)
    classes, size = meta["classes"], meta["input_size"]
    merge = meta.get("args", {}).get("merge", [])
    splits, data_classes = load_splits(args.data, merge=merge)
    if data_classes != classes:
        raise SystemExit(f"Dataset classes {data_classes} do not match checkpoint classes {classes}")
    images, labels = splits["test"]
    print(f"Evaluating {args.checkpoint} ({meta['arch']}) on {len(labels)} test images")

    probs = predict(model, images, size, device, args.tta)
    pred = probs.argmax(1)
    report = classification_report(labels, pred, target_names=classes, digits=4, output_dict=True, zero_division=0)
    print(classification_report(labels, pred, target_names=classes, digits=4, zero_division=0))

    cm = confusion_matrix(labels, pred, labels=range(len(classes)))
    plot_confusion(cm, classes, out_dir / "confusion_matrix.png")
    off_diag = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes))
                for j in range(len(classes)) if i != j]
    top_confusions = [{"true": t, "predicted": pr, "count": int(c)} for c, t, pr in sorted(off_diag, reverse=True)[:5]]
    print("Most frequent confusions:", ", ".join(f"{d['true']}->{d['predicted']} ({d['count']})" for d in top_confusions))

    sweep = threshold_sweep(probs, labels, out_dir / "threshold_sweep")
    print("Threshold sweep (threshold: coverage / accuracy):")
    for r in sweep:
        acc = r["accuracy_on_confident"]
        print(f"  {r['threshold']:.1f}: {r['coverage']:.3f} / {acc:.4f}" if acc is not None else f"  {r['threshold']:.1f}: 0")

    metrics = {
        "checkpoint": str(args.checkpoint), "arch": meta["arch"], "tta": args.tta,
        "accuracy": float(accuracy_score(labels, pred)),
        "report": report, "confusion_matrix": cm.tolist(), "classes": classes,
        "top_confusions": top_confusions,
    }
    if not args.no_robustness:
        print("Robustness (test accuracy):")
        metrics["robustness"] = robustness(model, images, labels, size, device, args.tta, out_dir / "robustness")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    hist = args.checkpoint.with_suffix(".history.json")
    if hist.exists():
        plot_history(hist, out_dir / "training_curves.png")
    print(f"Results -> {out_dir}")


if __name__ == "__main__":
    main()
