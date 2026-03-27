#!/usr/bin/env python3
"""
scripts/setup_models.py — Download, fine-tune, and export models to ONNX.

Usage:
  python scripts/setup_models.py --action download    # download pretrained
  python scripts/setup_models.py --action finetune    # fine-tune on dataset
  python scripts/setup_models.py --action export      # export to ONNX
  python scripts/setup_models.py --action all         # do everything

Requirements:
  pip install torch torchvision timm onnx onnxruntime datasets transformers
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("setup_models")

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNet-B4 — AI Image Detector
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetB4Classifier(nn.Module):
    """
    EfficientNet-B4 with custom binary head for AI vs Real detection.
    Pretrained on ImageNet, fine-tuned on GenImage + ArtiFact datasets.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=0,      # remove head
            global_pool="avg",
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class XceptionClassifier(nn.Module):
    """
    XceptionNet for deepfake / face-manipulation detection.
    Fine-tuned on FaceForensics++ (c23 compression level).
    """

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),   # binary: deepfake probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(input_size: int, train: bool = True):
    """ImageNet-normalised transforms with augmentation for training."""
    import torchvision.transforms as T

    if train:
        return T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.1),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize(int(input_size * 1.1)),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    scaler=None,
) -> tuple[float, float]:
    """One training epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:  # mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1) if outputs.shape[1] > 1 else (outputs > 0).long().squeeze()
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> tuple[float, float, list, list]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)[:, 1] if outputs.shape[1] > 1 else torch.sigmoid(outputs).squeeze()
        preds = (probs > 0.5).long()
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, all_probs, all_labels


def finetune_efficientnet(
    data_dir: str,
    output_path: str = "models/efficientnet_b4_ai_detector.pth",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
):
    """
    Fine-tune EfficientNet-B4 on AI vs Real image dataset.

    Expected data_dir structure:
      data_dir/
        train/
          real/     (authentic photos)
          ai/       (AI-generated images)
        val/
          real/
          ai/

    Recommended datasets:
      - GenImage: https://github.com/GenImage-Dataset/GenImage
      - ArtiFact: https://github.com/awsaf49/artifact
      - CIFAKE: https://github.com/jordan-bird/CIFAKE
      - FaceForensics++: https://github.com/ondyari/FaceForensics
    """
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)

    # Datasets
    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=get_transforms(380, train=True),
    )
    val_ds = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=get_transforms(380, train=False),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = EfficientNetB4Classifier().to(device)

    # Loss: weighted cross-entropy to handle class imbalance
    class_counts = [len(os.listdir(os.path.join(data_dir, "train", c))) for c in ["real", "ai"]]
    total = sum(class_counts)
    weights = torch.tensor([total / (2 * c) for c in class_counts], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    # Optimiser: 2-stage LR (backbone lower, head higher)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr * 0.1},
        {"params": model.classifier.parameters(), "lr": lr},
    ], weight_decay=1e-4)

    # Cosine LR schedule with warm restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %02d/%02d | Train: loss=%.4f acc=%.1f%% | Val: loss=%.4f acc=%.1f%%",
            epoch, epochs, train_loss, train_acc * 100, val_loss, val_acc * 100
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            logger.info("  ✅ New best model saved (val_acc=%.1f%%)", val_acc * 100)

    logger.info("Training complete. Best val_acc: %.1f%%", best_val_acc * 100)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────────────────────────────────────

def export_to_onnx(
    model_class,
    weights_path: str,
    output_path: str,
    input_size: tuple,
    opset: int = 17,
):
    """Export PyTorch model → ONNX → quantise to INT8."""
    logger.info("Exporting %s → %s", weights_path, output_path)
    model = model_class()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, *input_size)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    logger.info("✅ ONNX exported: %s", output_path)

    # INT8 quantisation for faster CPU inference
    quantised_path = output_path.replace(".onnx", "_int8.onnx")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            output_path, quantised_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        logger.info("✅ INT8 quantised model: %s", quantised_path)
    except Exception as e:
        logger.warning("Quantisation failed: %s", e)

    return output_path


def download_pretrained():
    """Download and verify all required pretrained checkpoints."""
    logger.info("Downloading pretrained model weights...")

    # EfficientNet-B4 via timm (auto-download)
    try:
        import timm
        m = timm.create_model("efficientnet_b4", pretrained=True)
        logger.info("✅ EfficientNet-B4 (ImageNet pretrained) — ready for fine-tuning")
    except Exception as e:
        logger.error("EfficientNet download failed: %s", e)

    # XceptionNet via timm
    try:
        import timm
        m = timm.create_model("xception", pretrained=True)
        logger.info("✅ XceptionNet (ImageNet pretrained) — ready for fine-tuning")
    except Exception as e:
        logger.error("Xception download failed: %s", e)

    # CLIP via HuggingFace
    try:
        from transformers import CLIPModel, CLIPProcessor
        CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        logger.info("✅ CLIP ViT-L/14 downloaded")
    except Exception as e:
        logger.error("CLIP download failed: %s", e)

    # Sentence-Transformers for RAG embeddings
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        logger.info("✅ SentenceTransformer all-mpnet-base-v2 downloaded")
    except Exception as e:
        logger.error("SentenceTransformer download failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TruthScan model setup")
    parser.add_argument("--action", choices=["download", "finetune", "export", "all"], required=True)
    parser.add_argument("--data_dir", default="data/training", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    if args.action in ("download", "all"):
        download_pretrained()

    if args.action in ("finetune", "all"):
        if not os.path.exists(args.data_dir):
            logger.error("Data directory not found: %s", args.data_dir)
            logger.info("Please download one of: GenImage, CIFAKE, ArtiFact, FaceForensics++")
            return
        pth = finetune_efficientnet(
            args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    if args.action in ("export", "all"):
        # Export EfficientNet
        eff_pth = "models/efficientnet_b4_ai_detector.pth"
        if os.path.exists(eff_pth):
            export_to_onnx(EfficientNetB4Classifier, eff_pth, "models/efficientnet_b4_ai_detector.onnx", (380, 380))
        # Export Xception
        xcp_pth = "models/xception_deepfake.pth"
        if os.path.exists(xcp_pth):
            export_to_onnx(XceptionClassifier, xcp_pth, "models/xception_deepfake.onnx", (299, 299))

    logger.info("🎉 Setup complete!")


if __name__ == "__main__":
    main()
