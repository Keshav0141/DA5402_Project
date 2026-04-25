# src/training/train_mlp.py

import os
import yaml
import logging
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

class ShallowMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ShallowMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.fc_layers(x)

def train(params: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 64 # Small size to prevent OOM with flattened MLPs
    batch_size = params["train"]["batch_size"]
    epochs = 10 # Shorter for MLP
    lr = params["train"]["learning_rate"]
    data_dir = Path(params["data"]["v2_dir"])
    classes = params["data"]["classes"]

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(data_dir / "Training", transform=tf)
    test_dataset = datasets.ImageFolder(data_dir / "Testing", transform=tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ShallowMLP(input_dim=3 * img_size * img_size, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="Shallow_MLP"):
        mlflow.log_param("model", "MLP")
        mlflow.log_param("img_size", img_size)
        mlflow.log_param("data_version", "v2_augmented")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("scheduler", "None")
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("train_samples", len(train_dataset))
        mlflow.log_param("test_samples", len(test_dataset))
        mlflow.log_param("device", str(device))

        best_f1 = 0.0
        best_model_path = Path("models/artifacts/mlp_best.pth")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # --- Training phase ---
            model.train()
            running_loss = 0.0
            num_batches = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                num_batches += 1
            
            train_loss = running_loss / num_batches

            # --- Validation phase ---
            model.eval()
            all_preds, all_labels_list = [], []
            val_running_loss = 0.0
            val_batches = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    v_loss = criterion(outputs, labels)
                    val_running_loss += v_loss.item()
                    val_batches += 1
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels_list.extend(labels.cpu().numpy())
            
            val_loss = val_running_loss / val_batches
            val_acc = correct / total
            macro_f1 = f1_score(all_labels_list, all_preds, average="macro")
            epoch_time = time.time() - epoch_start

            logger.info(f"Epoch {epoch}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | macro_f1: {macro_f1:.4f}")
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("macro_f1", macro_f1, step=epoch)
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), best_model_path)
            mlflow.log_metric("best_macro_f1", best_f1, step=epoch)

        mlflow.pytorch.log_model(model, "mlp_model")
        metrics = {"macro_f1": round(best_f1, 4), "model": "MLP"}
        metrics_path = Path("models/artifacts/metrics_mlp.json")
        metrics_path.write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(metrics_path))

if __name__ == "__main__":
    train(load_params())
