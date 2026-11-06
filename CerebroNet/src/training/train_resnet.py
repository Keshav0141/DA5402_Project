# src/training/train_resnet.py
# ResNet-50 (ImageNet) fine-tuned for 4-class brain MRI classification.
# Heavy baseline: run via `mlflow run . -e train_resnet` (not part of dvc.yaml).

import sys
from pathlib import Path

import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_finetune import load_params, train, logger  # noqa: E402


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    # Freeze the backbone; fine-tune only the last residual stage + new head.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


if __name__ == "__main__":
    try:
        params = load_params()
        model = build_model(len(params["data"]["classes"]))
        train(params, model, run_name="ResNet-50", tag="resnet50")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
