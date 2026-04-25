# src/pipeline/evaluate.py

import yaml
import json
import logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, classification_report

from src.training.train_mobilenet import build_model # Reuse MobileNetV2 architecture

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def evaluate(params: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = params["data"]["img_size"]
    data_dir = Path(params["data"]["v2_dir"])
    classes = params["data"]["classes"]

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Strictly evaluate on the 10% Test split as required by rubric
    test_dataset = datasets.ImageFolder(data_dir / "Testing", transform=tf)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model_path = Path("models/artifacts/mobilenetv2_best.pth")
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run training first.")
        return

    logger.info("Loading MobileNetV2 for DVC Evaluation...")
    model = build_model(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    logger.info(f"Final Test Macro F1: {macro_f1:.4f}")

    # Generate Confusion Matrix for DVC Plots
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save as CSV for DVC
    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    
    # Format required by dvc plots: actual,predicted
    df = pd.DataFrame({
        'actual': [classes[i] for i in all_labels],
        'predicted': [classes[i] for i in all_preds]
    })
    df.to_csv(eval_dir / "confusion_matrix.csv", index=False)
    
    logger.info(f"Saved DVC Plot data to {eval_dir / 'confusion_matrix.csv'}")

if __name__ == "__main__":
    evaluate(load_params())
