# src/training/train_baseline_cnn.py
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import yaml
import logging
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, classification_report
import numpy as np
import mlflow
import mlflow.pytorch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class BaseCNN(nn.Module):
    """Simple 3-block CNN baseline."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
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
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report   = classification_report(
        all_labels, all_preds, target_names=classes
    )
    return avg_loss, accuracy, macro_f1, report


def train(params: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    img_size   = params["data"]["img_size"]
    batch_size = params["train"]["batch_size"]
    epochs     = params["train"]["epochs"]
    lr         = params["train"]["learning_rate"]
    patience   = params["train"]["early_stopping_patience"]
    data_dir   = Path(params["data"]["v1_dir"])
    classes    = params["data"]["classes"]

    logger.info("Loading datasets...")
    train_tf, val_tf = get_transforms(img_size)

    train_dataset = datasets.ImageFolder(
        data_dir / "Training", transform=train_tf
    )
    test_dataset = datasets.ImageFolder(
        data_dir / "Testing", transform=val_tf
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False
    )

    logger.info(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    model     = BaseCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="BaseCNN"):
        mlflow.log_param("model",         "BaseCNN")
        mlflow.log_param("optimizer",     "Adam")
        mlflow.log_param("scheduler",     "StepLR")
        mlflow.log_param("batch_size",    batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs",        epochs)
        mlflow.log_param("img_size",      img_size)
        mlflow.log_param("train_samples", len(train_dataset))
        mlflow.log_param("test_samples",  len(test_dataset))
        mlflow.log_param("device",        str(device))
        mlflow.log_param("data_version",  "v1_resized")

        best_f1      = 0.0
        patience_ctr = 0
        best_model_path = Path("models/artifacts/baseline_cnn_best.pth")
        best_model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Starting training...")

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            start_time   = time.time()

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if batch_idx % 50 == 0:
                    logger.info(
                        f"  Epoch {epoch} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

            scheduler.step()
            epoch_time = time.time() - start_time
            avg_loss   = running_loss / len(train_loader)

            val_loss, val_acc, val_f1, _ = evaluate(
                model, test_loader, device, classes
            )

            logger.info(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Macro F1: {val_f1:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            mlflow.log_metric("train_loss", avg_loss,   step=epoch)
            mlflow.log_metric("val_loss",   val_loss,   step=epoch)
            mlflow.log_metric("val_acc",    val_acc,    step=epoch)
            mlflow.log_metric("macro_f1",   val_f1,     step=epoch)
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_ctr = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"  New best model saved (F1={best_f1:.4f})")
            else:
                patience_ctr += 1
                logger.info(
                    f"  No improvement. "
                    f"Patience: {patience_ctr}/{patience}"
                )

            if patience_ctr >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(torch.load(best_model_path))
        _, _, final_f1, report = evaluate(
            model, test_loader, device, classes
        )

        logger.info(f"\nFinal Macro F1: {final_f1:.4f}")
        logger.info(f"\n{report}")

        mlflow.log_metric("best_macro_f1", final_f1)

        report_path = Path("models/artifacts/classification_report_cnn.txt")
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))

        mlflow.pytorch.log_model(model, "baseline_cnn_model")

        metrics = {
            "macro_f1":       round(final_f1, 4),
            "model":          "BaseCNN",
            "epochs_trained": epoch
        }
        metrics_path = Path("models/artifacts/metrics_baseline_cnn.json")
        metrics_path.write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(metrics_path))

        logger.info(f"Done! Best Macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    try:
        params = load_params()
        train(params)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise