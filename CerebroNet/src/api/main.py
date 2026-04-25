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
import base64
import matplotlib.pyplot as plt
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
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
    ["endpoint", "status", "mode", "client_ip"]
)

PAYLOAD_SIZE_BYTES = Summary(
    "cerebronet_payload_size_bytes",
    "Summary of incoming payload sizes in bytes"
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


# ── Grad-CAM Implementation ───────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        # Requires grad for backward pass
        input_tensor.requires_grad = True
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)
        
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap) + 1e-8
        
        return heatmap.cpu().numpy()

def get_cam_image_base64(heatmap, original_img):
    # Resize heatmap to match image using PIL
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(original_img.size, Image.Resampling.BILINEAR)
    
    # Apply JET colormap
    cm = plt.get_cmap('jet')
    heatmap_colored = cm(np.array(heatmap_img)/255.0)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    heatmap_colored_img = Image.fromarray(heatmap_colored).convert("RGBA")
    original_rgba = original_img.convert("RGBA")
    
    # Blend images
    blended = Image.blend(original_rgba, heatmap_colored_img, alpha=0.5)
    
    buffered = io.BytesIO()
    blended.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


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
    client_ip = request.client.host if request.client else "unknown"
    if model is None:
        REQUEST_COUNT.labels(endpoint="predict", status="error", mode="single", client_ip=client_ip).inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        # Read file
        contents = await file.read()
        PAYLOAD_SIZE_BYTES.observe(len(contents))

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
        REQUEST_COUNT.labels(endpoint="predict", status="success", mode="single", client_ip=client_ip).inc()

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
                "image_stored": False,
                "exif_stripped": True,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="predict", status="error", mode="single", client_ip=client_ip).inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


# Predict with Grad-CAM Endpoint
@app.post("/predict_cam")
async def predict_cam(request: Request, file: UploadFile = File(...)):
    client_ip = request.client.host if request.client else "unknown"
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = strip_exif(image)

        tensor = transform(image).unsqueeze(0).to(device)

        # Grad-CAM requires gradient calculation, so no torch.no_grad() here
        cam = GradCAM(model, model.features[-1])
        heatmap = cam.generate(tensor)
        
        # We also need the prediction info
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence = probs.max().item()
            pred_idx = probs.argmax().item()
            pred_class = classes[pred_idx]

        cam_base64 = get_cam_image_base64(heatmap, image)

        all_probs = {
            classes[i]: round(probs[0][i].item(), 4)
            for i in range(len(classes))
        }

        latency = time.time() - start_time
        
        # Log Prometheus Metrics
        client_ip = request.client.host if request and request.client else "unknown"
        INFERENCE_LATENCY.observe(latency)
        PREDICTION_COUNTER.labels(predicted_class=pred_class).inc()
        CONFIDENCE_HISTOGRAM.observe(confidence)
        REQUEST_COUNT.labels(endpoint="predict_cam", status="success", mode="single", client_ip=client_ip).inc()

        return {
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "all_probs": all_probs,
            "cam_base64": cam_base64,
            "latency_ms": round(latency * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Grad-CAM error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating Grad-CAM")


import zipfile

@app.post("/predict_bulk")
async def predict_bulk(request: Request, file: UploadFile = File(...)):
    """Accepts a ZIP file of images and returns batch predictions."""
    client_ip = request.client.host if request.client else "unknown"
    if model is None:
        REQUEST_COUNT.labels(endpoint="predict_bulk", status="error", mode="bulk", client_ip=client_ip).inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported for bulk mode.")

    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    results = []

    try:
        contents = await file.read()
        PAYLOAD_SIZE_BYTES.observe(len(contents))
        
        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            for item in archive.infolist():
                if item.is_dir() or not any(item.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                    continue
                    
                with archive.open(item) as img_file:
                    img_bytes = img_file.read()
                    
                    try:
                        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        image = strip_exif(image)
                        tensor = transform(image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            outputs = model(tensor)
                            probs = torch.softmax(outputs, dim=1)
                            confidence = probs.max().item()
                            pred_idx = probs.argmax().item()
                            pred_class = classes[pred_idx]
                            
                        results.append({
                            "filename": item.filename,
                            "prediction": pred_class,
                            "confidence": round(confidence, 4)
                        })
                        PREDICTION_COUNTER.labels(predicted_class=pred_class).inc()
                    except Exception as e:
                        logger.error(f"Failed to process {item.filename}: {e}")
                        results.append({
                            "filename": item.filename,
                            "error": "Failed to process image"
                        })

        latency = time.time() - start_time
        INFERENCE_LATENCY.observe(latency)
        REQUEST_COUNT.labels(endpoint="predict_bulk", status="success", mode="bulk", client_ip=client_ip).inc()
        
        return {
            "batch_size": len(results),
            "results": results,
            "latency_ms": round(latency * 1000, 2),
            "model": "MobileNetV2"
        }

    except zipfile.BadZipFile:
        REQUEST_COUNT.labels(endpoint="predict_bulk", status="error", mode="bulk", client_ip=client_ip).inc()
        raise HTTPException(status_code=400, detail="Invalid ZIP archive")
    except Exception as e:
        logger.error(f"Bulk prediction error: {str(e)}")
        REQUEST_COUNT.labels(endpoint="predict_bulk", status="error", mode="bulk", client_ip=client_ip).inc()
        raise HTTPException(status_code=500, detail="Internal server error")
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