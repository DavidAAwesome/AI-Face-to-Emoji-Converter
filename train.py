"""Train an expression classifier on FER2013.

Examples:
    python train.py --data data/fer2013.csv
    python train.py --data data/fer2013 --arch resnet18 --epochs 30
    python train.py --data data/fer2013 --no-augment --out checkpoints/cnn_noaug.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from emoji_converter.config import DEFAULT_CHECKPOINT
from emoji_converter.data import (FaceDataset, balanced_sampler, class_weights, eval_transform,
                                  load_splits, train_transform)
from emoji_converter.model import build_model, input_size_for, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="fer2013.csv or a folder with train/ and test/")
    p.add_argument("--arch", default="cnn", choices=["cnn", "small_cnn", "resnet18"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--no-augment", action="store_true", help="disable data augmentation")
    p.add_argument("--balance", choices=["none", "weights", "sampler"], default="weights",
                   help="how to handle class imbalance")
    p.add_argument("--merge", nargs="*", default=[], help="merge classes, e.g. disgust=angry")
    p.add_argument("--no-pretrained", action="store_true", help="train resnet18 from scratch")
    p.add_argument("--patience", type=int, default=8, help="early-stopping patience (epochs)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="use only N training images (quick tests)")
    p.add_argument("--out", type=Path, default=DEFAULT_CHECKPOINT)
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits, classes = load_splits(args.data, merge=args.merge, seed=args.seed)
    tr_x, tr_y = splits["train"]
    if args.limit:
        tr_x, tr_y = tr_x[: args.limit], tr_y[: args.limit]
    va_x, va_y = splits["val"]
    print(f"Classes: {classes}")
    print(f"Train {len(tr_y)} | val {len(va_y)} | test {len(splits['test'][1])} | device {device}")
    print("Train class counts:", dict(zip(classes, np.bincount(tr_y, minlength=len(classes)).tolist())))

    size = input_size_for(args.arch)
    train_ds = FaceDataset(tr_x, tr_y, train_transform(size, augment=not args.no_augment))
    val_ds = FaceDataset(va_x, va_y, eval_transform(size))
    sampler = balanced_sampler(tr_y, len(classes)) if args.balance == "sampler" else None
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=sampler is None, sampler=sampler,
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch_size * 2, num_workers=args.workers)

    model = build_model(args.arch, len(classes), pretrained=not args.no_pretrained).to(device)
    weight = class_weights(tr_y, len(classes)).to(device) if args.balance == "weights" else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_dl))

    history, best_acc, stale = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer, scheduler)
        va_loss, va_acc = run_epoch(model, val_dl, val_criterion, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        improved = va_acc > best_acc
        print(f"epoch {epoch:3d}/{args.epochs} | train {tr_loss:.3f} / {tr_acc:.3f} | "
              f"val {va_loss:.3f} / {va_acc:.3f} | {time.time() - t0:.0f}s{' *' if improved else ''}")
        if improved:
            best_acc, stale = va_acc, 0
            save_checkpoint(args.out, model, args.arch, classes, size,
                            extra={"val_acc": va_acc, "epoch": epoch, "args": vars(args) | {"out": str(args.out)}})
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping: no val improvement for {args.patience} epochs")
                break

    hist_path = args.out.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"Best val accuracy {best_acc:.4f} -> {args.out}")
    print(f"History -> {hist_path}  (plot with: python evaluate.py --history {hist_path})")


if __name__ == "__main__":
    main()
