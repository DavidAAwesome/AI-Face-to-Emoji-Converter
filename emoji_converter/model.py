"""Expression classifiers: a baseline CNN trained from scratch and a transfer-learning ResNet."""

import torch
import torch.nn as nn
from torchvision import models


def _conv_block(in_ch, out_ch, dropout):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout),
    )


class EmotionCNN(nn.Module):
    """VGG-style CNN for 48x48 grayscale faces.

    Early blocks learn edges and textures; deeper blocks respond to eye, brow and
    mouth configurations (proposal Sec. III-C).
    """

    def __init__(self, num_classes=7, in_channels=1, width=64, dropout=0.25):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _conv_block(in_channels, w, dropout),      # 48 -> 24
            _conv_block(w, w * 2, dropout),            # 24 -> 12
            _conv_block(w * 2, w * 4, dropout),        # 12 -> 6
            _conv_block(w * 4, w * 8, dropout),        # 6 -> 3
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(w * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SmallCNN(nn.Module):
    """Lightweight baseline used for the accuracy-vs-speed comparison."""

    def __init__(self, num_classes=7, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ResNetTransfer(nn.Module):
    """ImageNet-pretrained ResNet-18 fine-tuned on faces.

    Grayscale input is repeated to three channels so the pretrained stem can be reused.
    """

    def __init__(self, num_classes=7, in_channels=1, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.backbone.fc.in_features, num_classes)
        )
        self.in_channels = in_channels

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


# name -> (constructor, input image size)
ARCHITECTURES = {
    "cnn": (EmotionCNN, 48),
    "small_cnn": (SmallCNN, 48),
    "resnet18": (ResNetTransfer, 96),
}


def build_model(arch, num_classes, pretrained=True):
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(ARCHITECTURES)}")
    ctor, _ = ARCHITECTURES[arch]
    if arch == "resnet18":
        return ctor(num_classes=num_classes, pretrained=pretrained)
    return ctor(num_classes=num_classes)


def input_size_for(arch):
    return ARCHITECTURES[arch][1]


def save_checkpoint(path, model, arch, classes, input_size, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": arch,
            "classes": list(classes),
            "input_size": input_size,
            "state_dict": model.state_dict(),
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    """Return (model, metadata) with the model in eval mode on `device`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
