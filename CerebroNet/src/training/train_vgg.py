# src/training/train_vgg.py
# VGG-16 with BatchNorm (ImageNet) fine-tuned for 4-class brain MRI classification.
# Heavy baseline: run via `mlflow run . -e train_vgg` (not part of dvc.yaml).

import sys
from pathlib import Path

import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_finetune import load_params, train, logger  # noqa: E402


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.VGG16_BN_Weights.DEFAULT if pretrained else None
    model = models.vgg16_bn(weights=weights)
    # Freeze conv features except the last conv block (features[34:]).
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[34:].parameters():
        param.requires_grad = True
    # Replace the 3x4096 FC head (~120M params) with a light head.
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


if __name__ == "__main__":
    try:
        params = load_params()
        model = build_model(len(params["data"]["classes"]))
        train(params, model, run_name="VGG-16-BN", tag="vgg16")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
