# src/training/train_svm.py

import os
import yaml
import logging
import time
import json
from pathlib import Path
import joblib

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def extract_features(loader):
    features = []
    labels_list = []
    for images, labels in loader:
        # Flatten images: (batch, C, H, W) -> (batch, C*H*W)
        feats = images.view(images.size(0), -1).numpy()
        features.append(feats)
        labels_list.append(labels.numpy())
    return np.vstack(features), np.concatenate(labels_list)

def train(params: dict):
    # SVM is heavy, use a smaller image size for feature extraction
    img_size = 64 
    data_dir = Path(params["data"]["v2_dir"])
    classes = params["data"]["classes"]

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # grayscale-ish norm
    ])

    logger.info("Loading datasets for SVM...")
    train_dataset = datasets.ImageFolder(data_dir / "Training", transform=tf)
    test_dataset = datasets.ImageFolder(data_dir / "Testing", transform=tf)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    logger.info("Extracting flattened features...")
    X_train, y_train = extract_features(train_loader)
    X_test, y_test = extract_features(test_loader)

    logger.info(f"X_train shape before PCA: {X_train.shape}")

    from sklearn.decomposition import PCA
    logger.info("Applying PCA to reduce dimensionality to 100 components...")
    pca = PCA(n_components=100, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    logger.info(f"X_train shape after PCA: {X_train.shape}")

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="Traditional_SVM"):
        mlflow.log_param("model", "SVM (RBF)")
        mlflow.log_param("img_size", img_size)
        mlflow.log_param("data_version", "v2_augmented")
        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("pca_components", 100)
        mlflow.log_param("train_samples", len(train_dataset))
        mlflow.log_param("test_samples", len(test_dataset))
        mlflow.log_param("optimizer", "SMO (LibSVM)")
        mlflow.log_param("epochs", "N/A (Single Shot)")
        mlflow.log_param("batch_size", "N/A")

        logger.info("Training SVM (this may take a few minutes)...")
        start_time = time.time()
        
        clf = SVC(kernel='rbf', probability=True, random_state=42)
        clf.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.1f}s")

        logger.info("Evaluating SVM...")
        preds = clf.predict(X_test)
        
        accuracy = np.mean(preds == y_test)
        macro_f1 = f1_score(y_test, preds, average="macro")
        report = classification_report(y_test, preds, target_names=classes)

        logger.info(f"SVM Macro F1: {macro_f1:.4f}")
        
        mlflow.log_metric("val_acc", accuracy)
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("best_macro_f1", macro_f1)
        mlflow.log_metric("train_time", train_time)

        # Log classification report as artifact
        report_path = Path("models/artifacts/classification_report_svm.txt")
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))

        # Save model
        model_path = Path("models/artifacts/svm_model.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_path)
        
        mlflow.log_artifact(str(model_path))

        metrics = {"macro_f1": round(macro_f1, 4), "model": "SVM"}
        metrics_path = Path("models/artifacts/metrics_svm.json")
        metrics_path.write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(metrics_path))

        logger.info("SVM Pipeline Finished.")

if __name__ == "__main__":
    train(load_params())
