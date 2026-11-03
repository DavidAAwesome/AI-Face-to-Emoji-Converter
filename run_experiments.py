"""Run the training experiments from proposal Sec. IV-B and build a comparison table.

Each experiment trains a checkpoint, evaluates it on the test set and times inference.
The summary is written to results/experiments.md.

    python run_experiments.py --data data/fer2013.csv
    python run_experiments.py --data data/fer2013 --only baseline no_augment --epochs 20
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from emoji_converter.config import CHECKPOINT_DIR, RESULTS_DIR

EXPERIMENTS = {
    "baseline": ["--arch", "cnn"],                                   # 1) baseline CNN
    "small_cnn": ["--arch", "small_cnn"],                            # 2) different architecture
    "resnet18_transfer": ["--arch", "resnet18", "--lr", "3e-4"],     # 3) transfer learning
    "resnet18_scratch": ["--arch", "resnet18", "--no-pretrained"],   # 3) ... vs. from scratch
    "no_augment": ["--arch", "cnn", "--no-augment"],                 # 4) effect of augmentation
    "balanced_sampler": ["--arch", "cnn", "--balance", "sampler"],   # 5) preprocessing / imbalance
    "merge_disgust": ["--arch", "cnn", "--merge", "disgust=angry"],  # 5) adjusted class set
}


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), help="subset of experiments to run")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--skip-trained", action="store_true", help="reuse checkpoints that already exist")
    args = p.parse_args()

    rows = []
    for name in args.only or EXPERIMENTS:
        ckpt = CHECKPOINT_DIR / f"{name}.pt"
        if not (args.skip_trained and ckpt.exists()):
            run([sys.executable, "train.py", "--data", args.data, "--epochs", str(args.epochs),
                 "--out", str(ckpt), *EXPERIMENTS[name]])
        out_dir = RESULTS_DIR / name
        run([sys.executable, "evaluate.py", "--data", args.data, "--checkpoint", str(ckpt), "--out", str(out_dir)])
        run([sys.executable, "benchmark.py", "--model-only", str(ckpt), "--out", str(out_dir / "speed.json")])

        m = json.loads((out_dir / "metrics.json").read_text())
        speed = json.loads((out_dir / "speed.json").read_text())[0]
        rows.append({
            "experiment": name,
            "accuracy": m["accuracy"],
            "macro_f1": m["report"]["macro avg"]["f1-score"],
            "weighted_f1": m["report"]["weighted avg"]["f1-score"],
            "robust_mean": sum(m["robustness"].values()) / len(m["robustness"]),
            "ms_per_face": speed["mean_ms"],
            "params_m": speed["params"] / 1e6,
        })

    lines = ["| Experiment | Accuracy | Macro F1 | Weighted F1 | Mean robustness acc | ms / face (CPU) | Params (M) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['experiment']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | "
                     f"{r['robust_mean']:.4f} | {r['ms_per_face']:.2f} | {r['params_m']:.2f} |")
    table = "\n".join(lines)
    (RESULTS_DIR / "experiments.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
