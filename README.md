# DA5402 Project – CerebroNet: Brain Tumor Classification MLOps Platform

**Course:** DA5402 — AI Application Development  
**Project:** End-to-End MLOps for Brain MRI Classification  
**Student:** Aryan Prasad  
**Roll Number:** DA25M007

---

## Project Overview

This project implements an **end-to-end brain tumor classification platform** with full MLOps infrastructure that:

* Classifies brain MRI scans into 4 categories: **Glioma, Meningioma, Pituitary, No Tumor**
* Trains and compares **5 model architectures** (BaseCNN, MobileNetV2, EfficientNet-B0, SVM, MLP)
* Tracks all experiments via **MLflow** with parameters, metrics, and artifacts
* Manages a **7-stage reproducible DVC pipeline** (prepare → transform → 5 trainers → evaluate)
* Orchestrates data ingestion via **Apache Airflow** (2 DAGs)
* Monitors production inference with **Prometheus + Grafana** (8 custom metrics, 12-panel dashboard)
* Serves the best model through a **FastAPI REST API** with 7 endpoints
* Provides an interactive **Streamlit web UI** with Grad-CAM explainability, batch processing, and pipeline visualization
* Enforces **data privacy**: EXIF stripping, in-memory processing, zero disk persistence

The system is designed with **reproducibility, observability, privacy, and deployment** as primary goals.

**Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data) — 5,712 raw images across 4 classes (Glioma, Meningioma, No Tumor, Pituitary), augmented to 11,200 for training.

---

## Tech Stack

| Component | Purpose |
|---|---|
| PyTorch | Model training & inference (MobileNetV2, EfficientNet, BaseCNN, MLP) |
| scikit-learn | SVM classifier training |
| FastAPI | REST inference API (7 endpoints, Prometheus-instrumented) |
| Streamlit | Interactive web frontend (8 pages) |
| MLflow | Experiment tracking, model registry, reproducible runs |
| DVC + Git | Data versioning, 7-stage reproducible pipeline |
| Apache Airflow | ETL pipeline orchestration (2 DAGs) + SMTP email alerts |
| Prometheus | Real-time metrics scraping (15s interval, 8 custom metrics) |
| Grafana | 12-panel observability dashboard (auto-provisioned) |
| Docker Compose | 7-container microservice orchestration |
| GitHub Actions | Automated CI pipeline with dynamic model mocking |
| Pillow | Image preprocessing + EXIF metadata stripping |

---

## Project Structure

```
DA5402_Project/
├── CerebroNet/
│   ├── frontend/
│   │   ├── app.py                      # Streamlit frontend (8 pages)
│   │   ├── style.css                   # Centralized UI stylesheet (1275 lines)
│   │   └── brain_bg.png                # Hero section brain visualization
│   ├── src/
│   │   ├── api/
│   │   │   └── main.py                 # FastAPI backend (7 endpoints, 8 Prom. metrics)
│   │   ├── pipeline/
│   │   │   ├── prepare.py              # Stage 1: Raw data → V1 (copy + organize)
│   │   │   ├── transform.py            # Stage 2: V1 → V2 (resize 224² + augment)
│   │   │   └── evaluate.py             # Stage 7: Best model evaluation + confusion matrix
│   │   └── training/
│   │       ├── train_baseline_cnn.py   # BaseCNN (F1: 0.810)
│   │       ├── train_mobilenet.py      # MobileNetV2 (F1: 0.948) ← DEPLOYED
│   │       ├── train_efficientnet.py   # EfficientNet-B0 (F1: 0.960)
│   │       ├── train_svm.py            # SVM (F1: 0.859)
│   │       └── train_mlp.py            # MLP (F1: 0.840)
│   ├── tests/
│   │   ├── test_api.py                 # 12 API test cases
│   │   └── test_ui_colors.py           # 4 UI color-mapping test cases
│   ├── airflow/dags/
│   │   ├── cerebronet_dag.py           # Training pipeline DAG
│   │   └── cerebronet_scraper_dag.py   # Data ingestion DAG
│   ├── prometheus/
│   │   ├── prometheus.yml              # Scrape config (3 targets)
│   │   ├── alert_rules.yml             # Alert rules (error rate, CPU, latency)
│   │   └── alertmanager.yml            # Alert notification config
│   ├── grafana/
│   │   ├── cerebronet_observability_dashboard.json  # 12-panel dashboard
│   │   └── provisioning/               # Auto-provisioned datasources
│   ├── models/artifacts/               # Trained checkpoints + metrics JSON
│   ├── docs/
│   │   ├── architecture.md             # System architecture document
│   │   ├── high_level_design.md        # HLD with design rationale
│   │   ├── low_level_design.md         # LLD with API specs + class definitions
│   │   ├── project_report.md           # Detailed project report (docs copy)
│   │   ├── test_plan.md                # 19 test cases + execution report
│   │   └── user_manual.md              # Page-by-page usage guide
│   ├── docker-compose.yml              # 7-service orchestration
│   ├── Dockerfile.backend              # FastAPI container
│   ├── Dockerfile.frontend             # Streamlit container
│   ├── Dockerfile.mlflow               # MLflow container
│   ├── dvc.yaml                        # 7-stage DVC pipeline definition
│   ├── params.yaml                     # Centralized hyperparameters
│   ├── MLproject                       # MLflow project file
│   ├── conda.yaml                      # Conda environment specification
│   ├── requirements.txt                # pip dependencies
│   ├── start_all.ps1                   # One-click launcher script
│   └── .env                            # Airflow SMTP environment config
├── Report.md                           # Comprehensive project report (Markdown)
├── Report.pdf                          # Project report (PDF for submission)
└── README.md                           # This file
```

> Note: `data/` directories are excluded via `.gitignore`. Model checkpoints in `models/artifacts/` and `mlflow.db` are included so the TA has immediate access to trained models and experiment logs.

---

## Key Features

### 1. Multi-Model Experiment Tracking (MLflow)

* **5 model architectures** trained and compared: BaseCNN, MobileNetV2, EfficientNet-B0, SVM, MLP
* Logs **parameters**: learning rate, batch size, epochs, optimizer, early stopping patience
* Logs **metrics**: macro_f1, val_acc, epoch_time, total_train_time
* Logs **artifacts**: model checkpoints (.pth/.pkl), confusion matrices, classification reports
* MLflow UI accessible at `http://localhost:5000`

### 2. Dual-Layer Continuous Integration (CI)

We implemented a sophisticated dual-layer CI approach:
* **MLOps CI (DVC):** `dvc.yaml` defines a **7-stage** reproducible DAG (prepare → transform → 5 trainers → evaluate). All hyperparameters are centralized in `params.yaml`.
* **Software Engineering CI (GitHub Actions):** `.github/workflows/ci.yml` runs automated Pytest checks on every push. To bypass "No Cloud" restrictions (which prevented DVC from pulling the 10MB model to GitHub servers), the CI pipeline dynamically imports and generates a custom PyTorch architectural mock on the fly so FastAPI can boot and tests can pass perfectly in the cloud.

### 3. REST API (FastAPI)

7 endpoints with native Prometheus instrumentation:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Container liveness probe |
| `/ready` | GET | Model readiness probe |
| `/predict` | POST | Single image classification |
| `/predict_cam` | POST | Classification + Grad-CAM heatmap |
| `/predict_bulk` | POST | ZIP batch processing |
| `/metrics` | GET | Prometheus scrape target (8 metrics) |
| `/info` | GET | Model metadata |

### 4. Web Frontend (Streamlit – 8 Pages)

| Page | Purpose |
|---|---|
| **Predict** | Single scan + batch ZIP upload with Grad-CAM explainability |
| **Model Tracker** | MLflow experiment comparison with charts |
| **Pipeline** | DVC DAG visualization, live Airflow DAG status, run history, speed metrics |
| **Apps Hub** | Direct links to MLflow, Grafana, Airflow, Prometheus, FastAPI Docs |
| **Privacy Policy** | Data handling and EXIF stripping documentation |
| **FAQ** | Common questions about the platform |
| **Contact Us** | Support inquiry form |
| **About** | Project credits |

### 5. Real-Time Monitoring (Prometheus + Grafana)

**8 custom Prometheus metrics** instrumented in FastAPI:
- `cerebronet_requests_total`, `cerebronet_inference_latency_seconds`, `cerebronet_predictions_total`, `cerebronet_confidence_score`, `cerebronet_model_loaded`, `cerebronet_active_requests`, `cerebronet_payload_size_bytes`, `cerebronet_rejected_uploads_total`

**12-panel Grafana dashboard** auto-provisioned on startup.

### 6. Data Privacy Architecture

* EXIF metadata stripped (GPS, timestamps, device info)
* All processing in-memory — no image written to disk
* Garbage collected immediately after inference
* File validation: extension whitelist, MIME type, 10MB limit

### 7. Grad-CAM Explainability

Gradient-weighted Class Activation Mapping on MobileNetV2's last convolutional layer, generating heatmap overlays showing which MRI regions influenced the prediction.

---

## Model Performance

| Model | Macro F1 | Val Accuracy | Speed/Epoch | Inference | Deployed |
|---|---|---|---|---|---|
| EfficientNet-B0 | **0.960** | 96.5% | 650s | ~500ms | ❌ |
| **MobileNetV2** | **0.948** | 95.2% | 80.5s | <200ms | ✅ |
| SVM | 0.859 | 87.1% | — | <50ms | ❌ |
| MLP | 0.840 | 85.3% | 45s | <30ms | ❌ |
| BaseCNN | 0.810 | 82.4% | 120s | <100ms | ❌ |

**Why MobileNetV2 over EfficientNet?** The 1.2% F1 gap does not justify 8× slower training and 2.5× slower inference. For real-time clinical screening, sub-200ms latency is critical.

---

## Architecture Diagram

```
┌──────────────┐    REST API     ┌──────────────────┐
│   Streamlit   │───────────────▶│   FastAPI Backend │
│   Frontend    │  /predict      │   + PyTorch       │
│   :8501       │  /predict_cam  │   :8000           │
│   (8 pages)   │  /predict_bulk │   MobileNetV2     │
└──────────────┘                 └────────┬──────────┘
                                          │ /metrics
                    ┌─────────────────────┤
                    │                     │
                    ▼                     ▼
             ┌────────────┐       ┌──────────────┐
             │ Prometheus  │       │   MLflow      │
             │ :9090       │       │   :5000       │
             └──────┬─────┘       └──────────────┘
                    │
                    ▼
             ┌────────────┐       ┌──────────────┐
             │  Grafana    │       │ Node Exporter │
             │  :3001      │       │ :9100         │
             └────────────┘       └──────────────┘

             ┌──────────────┐
             │   Airflow     │  2 DAGs
             │   :8080       │  (ETL + Scraper)
             └──────────────┘
```

---

## Environment Setup & Running

### Prerequisites

* **Docker Desktop** (v4.0+)
* **Git**
* **8 GB RAM** minimum (16 GB recommended)

### Quick Start

```bash
cd DA5402_Project/CerebroNet
docker compose up -d --build
```

Wait ~60 seconds for all services to initialize.

### Service URLs

| Service | Port | URL | Credentials |
|---|---|---|---|
| **CerebroNet UI** | 8501 | http://localhost:8501 | — |
| **FastAPI Docs** | 8000 | http://localhost:8000/docs | — |
| **MLflow** | 5000 | http://localhost:5000 | — |
| **Grafana** | 3001 | http://localhost:3001 | admin / admin |
| **Airflow** | 8080 | http://localhost:8080 | admin / admin |
| **Prometheus** | 9090 | http://localhost:9090 | — |

### Stop Services

```bash
docker compose down
```

---

## Testing & CI/CD

```bash
# Run the automated test suite locally inside the backend container
docker exec cerebronet_backend pytest tests/test_api.py -v
```

**Test Summary**: Automated test suite passed 100% (28/28 assertions).

A GitHub Actions CI pipeline is configured to automatically spin up an Ubuntu server, install all dependencies, dynamically generate a PyTorch model mock, and execute this test suite on every code push.

---

## What Gets Logged to MLflow

| Category | Items Logged |
|---|---|
| Parameters | learning_rate, batch_size, epochs, optimizer, early_stopping_patience, architecture |
| Metrics | macro_f1, val_acc, epoch_time, total_train_time |
| Artifacts | Model checkpoints (.pth/.pkl), confusion matrices, classification reports |
| Registry | Best model (MobileNetV2) tracked across runs |

---

## Documentation

| Document | Location | Description |
|---|---|---|
| **Project Report** | [`Report.md`](Report.md) | Comprehensive report (problem, methodology, results, MLOps) |
| Architecture | [`CerebroNet/docs/architecture.md`](CerebroNet/docs/architecture.md) | System architecture + container topology |
| High-Level Design | [`CerebroNet/docs/high_level_design.md`](CerebroNet/docs/high_level_design.md) | Design choices and rationale |
| Low-Level Design | [`CerebroNet/docs/low_level_design.md`](CerebroNet/docs/low_level_design.md) | API endpoint specs + class definitions |
| Project Report (docs) | [`CerebroNet/docs/project_report.md`](CerebroNet/docs/project_report.md) | Detailed project report (alternate copy) |
| Test Plan | [`CerebroNet/docs/test_plan.md`](CerebroNet/docs/test_plan.md) | 19 test cases + execution report |
| User Manual | [`CerebroNet/docs/user_manual.md`](CerebroNet/docs/user_manual.md) | Page-by-page usage guide |

---

## Conclusion

This project demonstrates a complete MLOps lifecycle:

* **Data Engineering**: Automated ingestion, versioning (DVC), and augmentation pipeline
* **Model Development**: 5 architectures trained, tracked (MLflow), and compared
* **Deployment**: Production model served via REST API with health checks (Docker Compose)
* **Monitoring**: Real-time observability with 8 custom metrics and 12-panel Grafana dashboard
* **Explainability**: Grad-CAM heatmaps for clinical interpretability
* **Privacy**: Zero-persistence architecture with EXIF stripping
* **Orchestration**: Airflow DAGs for ETL pipeline automation with SMTP email alerts

All rubric requirements have been implemented and validated.
