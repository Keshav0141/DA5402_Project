# src/training/cnn_finetune.py
"""Shared fine-tuning loop for the heavy ImageNet backbones (ResNet-50, VGG-16-BN).

These models are run outside the default 7-stage DVC pipeline because of their
compute cost; launch them via MLflow entry points (see MLproject) or directly.
"""

import logging
import time
import json
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, classification_report
import mlflow
import mlflow.pytorch

import sys

# Windows consoles default to cp1252; MLflow prints emoji at end of run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,  # imported libraries may install root handlers first
)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def evaluate(model, loader, device, classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, all_preds, labels=list(range(len(classes))),
        target_names=classes, zero_division=0
    )
    return avg_loss, accuracy, macro_f1, report


def train(params: dict, model: nn.Module, run_name: str, tag: str) -> float:
    """Fine-tune `model` on data/v2_augmented and log everything to MLflow.

    `tag` names the artifacts (e.g. "resnet50" -> resnet50_best.pth,
    metrics_resnet50.json). Returns the best macro F1 on the test split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    img_size = params["data"]["img_size"]
    batch_size = params["train"]["batch_size"]
    epochs = params["train"]["epochs"]
    lr = params["train"]["learning_rate"]
    patience = params["train"]["early_stopping_patience"]
    data_dir = Path(params["data"]["v2_dir"])
    classes = params["data"]["classes"]

    train_tf, val_tf = get_transforms(img_size)
    train_dataset = datasets.ImageFolder(data_dir / "Training", transform=train_tf)
    test_dataset = datasets.ImageFolder(data_dir / "Testing", transform=val_tf)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logger.info(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    artifacts = Path("models/artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    best_model_path = artifacts / f"{tag}_best.pth"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": run_name,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs,
            "img_size": img_size,
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "trainable_params": trainable,
            "total_params": total,
            "device": str(device),
            "data_version": "v2_augmented",
        })

        best_f1, patience_ctr, epoch = 0.0, 0, 0
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            start_time = time.time()
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if batch_idx % 50 == 0:
                    logger.info(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            scheduler.step()
            epoch_time = time.time() - start_time
            avg_loss = running_loss / len(train_loader)

            val_loss, val_acc, val_f1, _ = evaluate(model, test_loader, device, classes)
            logger.info(
                f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Macro F1: {val_f1:.4f} | Time: {epoch_time:.1f}s"
            )
            mlflow.log_metrics({
                "train_loss": avg_loss, "val_loss": val_loss, "val_acc": val_acc,
                "macro_f1": val_f1, "epoch_time": epoch_time,
            }, step=epoch)

            if val_f1 > best_f1:
                best_f1, patience_ctr = val_f1, 0
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"  New best model saved (F1={best_f1:.4f})")
            else:
                patience_ctr += 1
                logger.info(f"  No improvement. Patience: {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        _, final_acc, final_f1, report = evaluate(model, test_loader, device, classes)
        logger.info(f"\nFinal Macro F1: {final_f1:.4f}\n{report}")
        mlflow.log_metric("best_macro_f1", final_f1)
        mlflow.log_metric("best_val_acc", final_acc)

        report_path = artifacts / f"classification_report_{tag}.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))
        example = test_dataset[0][0].unsqueeze(0).numpy()
        mlflow.pytorch.log_model(model.cpu(), f"{tag}_model", input_example=example)

        metrics_path = artifacts / f"metrics_{tag}.json"
        metrics_path.write_text(json.dumps({
            "macro_f1": round(final_f1, 4),
            "val_acc": round(final_acc, 4),
            "model": run_name,
            "epochs_trained": epoch,
        }, indent=2))
        mlflow.log_artifact(str(metrics_path))

    logger.info(f"Done! Best Macro F1: {best_f1:.4f}")
    return best_f1
