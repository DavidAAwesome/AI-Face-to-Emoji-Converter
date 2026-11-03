"""FER2013 loading, train/val/test splitting, and image transforms.

Two on-disk layouts are supported:

* CSV  - the original ``fer2013.csv`` (columns: emotion, pixels, Usage). ``Training`` is
  used for training, ``PublicTest`` for validation and ``PrivateTest`` for testing.
* Folders - the Kaggle image layout ``<root>/train/<class>/*.png`` and
  ``<root>/test/<class>/*.png``. A stratified slice of ``train`` becomes the validation set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import FER2013_CLASSES

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
NORM_MEAN, NORM_STD = (0.5,), (0.5,)


class FaceDataset(Dataset):
    """In-memory dataset of grayscale face crops (uint8 HxW arrays)."""

    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        return self.transform(img), self.labels[idx]


def train_transform(input_size, augment=True):
    """Augmentation from proposal Sec. III-B: flips, small rotations, crops, brightness."""
    steps = [transforms.Resize((input_size, input_size))]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)]
    if augment:
        steps.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)))
    return transforms.Compose(steps)


def eval_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def preprocess_faces(gray_crops, input_size):
    """Turn a list of grayscale uint8 face crops into a normalised (N,1,S,S) batch."""
    tf = eval_transform(input_size)
    return torch.stack([tf(Image.fromarray(c)) for c in gray_crops])


def parse_merge(merge_args):
    """['disgust=angry'] -> {'disgust': 'angry'}"""
    merges = {}
    for item in merge_args or []:
        src, dst = item.split("=")
        merges[src.strip().lower()] = dst.strip().lower()
    return merges


def _remap(names, merges):
    """Apply class merges and return (label indices, final class list in FER order)."""
    names = [merges.get(n, n) for n in names]
    classes = [c for c in FER2013_CLASSES if c in set(names)]
    extra = sorted(set(names) - set(classes))
    classes += extra
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[n] for n in names]), classes


def _load_csv(csv_path):
    df = pd.read_csv(csv_path)
    pixels = np.stack([np.asarray(p.split(), dtype=np.uint8).reshape(48, 48) for p in df["pixels"]])
    names = [FER2013_CLASSES[e] for e in df["emotion"]]
    usage = df["Usage"].to_numpy() if "Usage" in df else np.array(["Training"] * len(df))
    return pixels, names, usage


def _load_folder(split_dir):
    images, names = [], []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                img = Image.open(f).convert("L").resize((48, 48))
                images.append(np.asarray(img, dtype=np.uint8))
                names.append(class_dir.name.lower())
    if not images:
        raise FileNotFoundError(f"No images found under {split_dir}")
    return np.stack(images), names


def load_splits(data_path, merge=None, val_fraction=0.1, seed=42):
    """Return {'train'|'val'|'test': (images, labels)} and the class list."""
    data_path = Path(data_path)
    merges = parse_merge(merge)

    if data_path.is_file() and data_path.suffix == ".csv":
        images, names, usage = _load_csv(data_path)
        labels, classes = _remap(names, merges)
        masks = {
            "train": usage == "Training",
            "val": usage == "PublicTest",
            "test": usage == "PrivateTest",
        }
        splits = {k: (images[m], labels[m]) for k, m in masks.items()}
        if len(splits["val"][1]) == 0:  # CSV without a Usage column
            splits = _split_train_val_test(images, labels, val_fraction, seed)
        return splits, classes

    train_dir = data_path / "train"
    test_dir = data_path / "test"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {data_path} to be fer2013.csv or a folder with train/ and test/ subfolders"
        )
    tr_images, tr_names = _load_folder(train_dir)
    te_images, te_names = _load_folder(test_dir) if test_dir.is_dir() else (None, [])
    labels, classes = _remap(tr_names + te_names, merges)
    tr_labels, te_labels = labels[: len(tr_names)], labels[len(tr_names):]

    if te_images is None:
        return _split_train_val_test(tr_images, tr_labels, val_fraction, seed), classes

    tr_idx, va_idx = train_test_split(
        np.arange(len(tr_labels)), test_size=val_fraction, stratify=tr_labels, random_state=seed
    )
    return {
        "train": (tr_images[tr_idx], tr_labels[tr_idx]),
        "val": (tr_images[va_idx], tr_labels[va_idx]),
        "test": (te_images, te_labels),
    }, classes


def _split_train_val_test(images, labels, val_fraction, seed):
    idx = np.arange(len(labels))
    rest, test = train_test_split(idx, test_size=0.15, stratify=labels, random_state=seed)
    train, val = train_test_split(
        rest, test_size=val_fraction, stratify=labels[rest], random_state=seed
    )
    return {k: (images[v], labels[v]) for k, v in (("train", train), ("val", val), ("test", test))}


def class_weights(labels, num_classes):
    """Inverse-frequency weights; FER2013 has ~16x fewer 'disgust' than 'happy' images."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def balanced_sampler(labels, num_classes):
    per_class = class_weights(labels, num_classes).numpy()
    return WeightedRandomSampler(per_class[labels], num_samples=len(labels), replacement=True)
