# High-Level Design (HLD)

## Overview
CerebroNet is an end-to-end Machine Learning pipeline that classifies brain MRI scans into 4 categories: Glioma, Meningioma, Pituitary, and No Tumor. The system strictly follows MLOps practices, enforcing automation, reproducibility, and environment parity.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Docker Compose Network                          │
│                                                                         │
│  ┌──────────────┐    REST API     ┌──────────────────┐                  │
│  │   Streamlit   │───────────────▶│   FastAPI Backend │                  │
│  │   Frontend    │   /predict     │   (Inference)     │                  │
│  │   :8501       │   /predict_cam │   :8000           │                  │
│  │               │   /predict_bulk│                    │                  │
│  └──────────────┘                └────────┬───────────┘                  │
│                                           │                              │
│                             ┌─────────────┼─────────────┐               │
│                             │             │             │                │
│                             ▼             ▼             ▼                │
│                     ┌────────────┐ ┌──────────┐ ┌──────────────┐        │
│                     │ Prometheus │ │  MLflow   │ │Node Exporter │        │
│                     │   :9090    │ │  :5000    │ │   :9100      │        │
│                     └──────┬─────┘ └──────────┘ └──────────────┘        │
│                            │                                             │
│                            ▼                                             │
│                     ┌────────────┐                                       │
│                     │  Grafana   │                                       │
│                     │   :3001    │                                       │
│                     └────────────┘                                       │
│                                                                          │
│  ┌──────────────┐                                                        │
│  │   Airflow     │  Orchestrates data ingestion DAGs                     │
│  │   :8080       │  and ETL pipelines                                    │
│  └──────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Design Choices & Rationale

### 1. Model Selection: MobileNetV2 over EfficientNet-B0
MobileNetV2 was chosen as the production model due to its optimal accuracy-latency tradeoff:
- **MobileNetV2**: F1 = 0.948, Training speed = 80.5s/epoch, Inference < 200ms
- **EfficientNet-B0**: F1 = 0.960, Training speed = 650s/epoch, Inference ~500ms
- **Decision**: The 1.2% F1 difference does not justify the 8x slower training and 2.5x slower inference. For a real-time clinical screening tool, low latency is critical.

### 2. Containerization Strategy
Docker and Docker Compose orchestrate 7 isolated microservices:
- **Frontend** (Streamlit) — User-facing web application
- **Backend** (FastAPI + PyTorch) — Model inference engine
- **MLflow** — Experiment tracking and model registry
- **Prometheus** — Metrics scraping at 15s intervals
- **Grafana** — 12-panel observability dashboard (auto-provisioned)
- **Node Exporter** — Host CPU/memory metrics
- **Airflow** — ETL DAG orchestration

This guarantees environment parity across development and production as mandated by the rubric.

The **Frontend** exposes 8 UI pages: Predict (Single + Batch), Model Tracker, Pipeline Visualization, Apps Hub, Privacy Policy, FAQ, Contact Us, and About. The **Backend** supports single-image prediction (`/predict`, `/predict_cam`) and batch ZIP processing (`/predict_bulk`).

### 3. Loose Coupling (Frontend ↔ Backend)
The frontend and backend are **strictly decoupled**:
- Frontend communicates exclusively via REST API calls to `http://backend:8000`
- No shared filesystem, no shared memory, no direct imports
- The `API_URL` is configurable via environment variable in `docker-compose.yml`
- Frontend can be replaced with any HTTP client without modifying the backend

### 4. Data Privacy Architecture
The system is designed to be fully in-memory:
- Images uploaded via the frontend are sent to the FastAPI backend
- Decoded in RAM, stripped of EXIF metadata, predicted, and immediately discarded
- No image is ever written to disk — `del contents, image, tensor` is called post-inference
- EXIF stripping removes GPS coordinates, timestamps, and device identifiers

### 5. Data Versioning with DVC
DVC is utilized alongside Git to manage large datasets:
- `dvc.yaml` defines 7 pipeline stages (prepare → transform → 5 trainers → evaluate)
- `params.yaml` centralizes hyperparameters for reproducibility
- Data artifacts are version-controlled without bloating the Git repository
- Full pipeline reproducible via `dvc repro`

### 6. Observability Stack
Prometheus instrumentation is natively embedded into the FastAPI application with 8 custom metrics:
- `cerebronet_requests_total` — Total requests by endpoint, status, mode
- `cerebronet_inference_latency_seconds` — Histogram with P50/P95/P99 buckets
- `cerebronet_predictions_total` — Per-class prediction counter
- `cerebronet_confidence_score` — Confidence distribution histogram
- `cerebronet_model_loaded` — Binary gauge for model status
- `cerebronet_active_requests` — Concurrent request gauge
- `cerebronet_payload_size_bytes` — Upload size summary
- `cerebronet_rejected_uploads_total` — Security rejection counter

### 7. Data Flow
```
Raw Data (5,712 images)
    │
    ▼ [prepare.py] Resize to 224x224
    │
    ▼ [transform.py] Augment + Train/Test Split → 11,200 images
    │
    ├──▶ [train_baseline_cnn.py]  → BaseCNN (F1: 0.810)
    ├──▶ [train_mobilenet.py]     → MobileNetV2 (F1: 0.948) ← DEPLOYED
    ├──▶ [train_efficientnet.py]  → EfficientNet-B0 (F1: 0.960)
    ├──▶ [train_svm.py]          → SVM (F1: 0.859)
    └──▶ [train_mlp.py]          → MLP (F1: 0.840)
    │
    ▼ [evaluate.py] Confusion matrix + best model selection
    │
    ▼ [FastAPI] Serve MobileNetV2 via /predict endpoint
```

## Technology Stack
| Layer | Technology | Justification |
|---|---|---|
| Frontend | Streamlit | Rapid prototyping with Python-native components |
| Backend | FastAPI + PyTorch | Async API server with GPU-ready inference |
| Experiment Tracking | MLflow | Industry standard for ML experiment management |
| Data Versioning | DVC + Git | Large dataset management without cloud storage |
| Orchestration | Apache Airflow | DAG-based ETL pipeline scheduling |
| Monitoring | Prometheus + Grafana | Real-time metrics collection and visualization |
| Containerization | Docker Compose | Multi-service orchestration with network isolation |
