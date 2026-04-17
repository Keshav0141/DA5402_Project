# src/api/main.py
import os
import logging
import time
import io
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ExifTags
import yaml

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi.responses import Response

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Security config
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "cerebronet_requests_total",
    "Total prediction requests",
    ["endpoint", "status"]
)
INFERENCE_LATENCY = Histogram(
    "cerebronet_inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)
PREDICTION_COUNTER = Counter(
    "cerebronet_predictions_total",
    "Total predictions per class",
    ["predicted_class"]
)
CONFIDENCE_HISTOGRAM = Histogram(
    "cerebronet_confidence_score",
    "Prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
MODEL_LOADED = Gauge(
    "cerebronet_model_loaded",
    "Whether model is loaded (1) or not (0)"
)
ACTIVE_REQUESTS = Gauge(
    "cerebronet_active_requests",
    "Number of active requests"
)
REJECTED_UPLOADS = Counter(
    "cerebronet_rejected_uploads_total",
    "Total rejected uploads due to security",
    ["reason"]
)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_mobilenet(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model


def strip_exif(image: Image.Image) -> Image.Image:
    """
    Strip ALL EXIF metadata from image for privacy.
    Patient data (GPS, device info) is permanently removed.
    """
    clean = Image.new(image.mode, image.size)
    clean.putdata(list(image.getdata()))
    logger.info("EXIF metadata stripped for privacy protection")
    return clean


def validate_file_security(
    filename: str,
    content_type: str,
    file_size: int
) -> None:
    """
    Validate uploaded file for security.
    Raises HTTPException if validation fails.
    """
    # Check file extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        REJECTED_UPLOADS.labels(reason="invalid_extension").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: JPG, JPEG, PNG"
        )

    # Check content type
    if content_type not in ALLOWED_CONTENT_TYPES:
        REJECTED_UPLOADS.labels(reason="invalid_content_type").inc()
        raise HTTPException(
            status_code=400,
            detail="Invalid content type. Must be an image."
        )

    # Check file size
    if file_size > MAX_FILE_SIZE_BYTES:
        REJECTED_UPLOADS.labels(reason="file_too_large").inc()
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_FILE_SIZE_MB}MB"
        )


# ── Global state ───────────────────────────────────────────────
model = None
device = None
classes = None
transform = None
params = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device, classes, transform, params

    logger.info("Loading model on startup...")
    params = load_params()
    classes = params["data"]["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path("models/artifacts/mobilenetv2_best.pth")

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        MODEL_LOADED.set(0)
        return

    model = build_mobilenet(num_classes=len(classes))
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    MODEL_LOADED.set(1)
    logger.info(f"Model loaded on {device}")
    yield


# ── App setup ──────────────────────────────────────────────────
app = FastAPI(
    title="CerebroNet API",
    description="Brain Tumor MRI Classification API — Privacy First",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health endpoints
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.get("/ready")
async def ready():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet"
        )
    return {"status": "ready"}


# Predict endpoint
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        # Read file
        contents = await file.read()

        # ── Security validation
        validate_file_security(
            filename=file.filename or "upload.jpg",
            content_type=file.content_type or "image/jpeg",
            file_size=len(contents)
        )

        # ── Open and validate as real image ───────────────────
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            REJECTED_UPLOADS.labels(reason="corrupt_image").inc()
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupt image file"
            )

        # ── Strip EXIF metadata (PRIVACY) ──────────────────────
        image = strip_exif(image)

        # ── Inference ──────────────────────────────────────────
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence = probs.max().item()
            pred_idx = probs.argmax().item()
            pred_class = classes[pred_idx]

        # Image is NOT saved — processed in memory only
        del contents, image, tensor

        latency = time.time() - start_time

        INFERENCE_LATENCY.observe(latency)
        PREDICTION_COUNTER.labels(predicted_class=pred_class).inc()
        CONFIDENCE_HISTOGRAM.observe(confidence)
        REQUEST_COUNT.labels(endpoint="predict", status="success").inc()

        all_probs = {
            classes[i]: round(probs[0][i].item(), 4)
            for i in range(len(classes))
        }

        logger.info(
            f"Prediction: {pred_class} | "
            f"Confidence: {confidence:.4f} | "
            f"Latency: {latency*1000:.1f}ms | "
            f"EXIF stripped: Yes"
        )

        return {
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "all_probs": all_probs,
            "latency_ms": round(latency * 1000, 2),
            "model": "MobileNetV2",
            "device": str(device),
            "privacy": {
                "image_stored": False,
                "exif_stripped": True,
                "data_retained": False
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Info endpoint
@app.get("/info")
async def info():
    return {
        "model": "MobileNetV2",
        "classes": classes,
        "input_size": "224x224",
        "framework": "PyTorch",
        "macro_f1": 0.9483,
        "device": str(device),
        "security": {
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "allowed_formats": list(ALLOWED_EXTENSIONS),
            "exif_stripping": True,
            "image_storage": False
        }
    }