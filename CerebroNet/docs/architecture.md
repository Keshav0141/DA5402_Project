# Architecture Document

## System Architecture

The CerebroNet platform is designed with a decoupled, microservice-oriented architecture running on local infrastructure without cloud dependencies. The system is orchestrated via Docker Compose and leverages industry-standard MLOps tooling.

```
                               ┌─────────────────────────┐
                               │   User / Clinician      │
                               │   (Web Browser)         │
                               └────────────┬────────────┘
                                            │ HTTP :8501
                                            ▼
┌───────────────────────────────────────────────────────────────────────┐
│                     Docker Compose Network                            │
│                                                                       │
│  ┌────────────────────┐          ┌────────────────────────┐           │
│  │  Streamlit Frontend │  REST   │  FastAPI Backend        │           │
│  │  (Container 1)      │────────▶│  (Container 2)          │           │
│  │  Port: 8501         │ /predict│  Port: 8000             │           │
│  │                     │ /cam    │  PyTorch + MobileNetV2  │           │
│  │  - Predict Page     │ /bulk   │  - Grad-CAM Engine      │           │
│  │  - Model Tracker    │ /health │  - Prometheus Exporter  │           │
│  │  - Pipeline Viz     │ /info   │  - EXIF Stripper        │           │
│  │  - Apps Hub         │ /metrics│  - Security Validator   │           │
│  └────────────────────┘         └──────────┬─────────────┘           │
│                                             │ /metrics                 │
│              ┌──────────────────────────────┤                         │
│              │                              │                          │
│              ▼                              ▼                          │
│  ┌────────────────────┐       ┌────────────────────┐                  │
│  │  Prometheus         │       │  MLflow Tracking    │                  │
│  │  (Container 3)      │       │  (Container 6)      │                  │
│  │  Port: 9090         │       │  Port: 5000          │                  │
│  │  Scrape: 15s        │       │  SQLite Backend      │                  │
│  │  Targets:           │       │  5 Experiment Runs   │                  │
│  │   - backend:8000    │       └────────────────────┘                  │
│  │   - node_exp:9100   │                                               │
│  │   - localhost:9090   │                                               │
│  └──────────┬─────────┘                                               │
│             │                                                          │
│             ▼                                                          │
│  ┌────────────────────┐       ┌────────────────────┐                  │
│  │  Grafana            │       │  Node Exporter       │                  │
│  │  (Container 4)      │       │  (Container 5)       │                  │
│  │  Port: 3001         │       │  Port: 9100           │                  │
│  │  12-panel dashboard │       │  CPU/Memory metrics   │                  │
│  │  Auto-provisioned   │       └────────────────────┘                  │
│  └────────────────────┘                                               │
│                                                                        │
│  ┌────────────────────┐                                               │
│  │  Apache Airflow     │                                               │
│  │  (Container 7)      │                                               │
│  │  Port: 8080          │                                               │
│  │  2 DAGs:             │                                               │
│  │   - cerebronet_dag   │                                               │
│  │   - scraper_dag      │                                               │
│  └────────────────────┘                                               │
└───────────────────────────────────────────────────────────────────────┘

External (Host):
  ┌──────────────────┐
  │  DVC + Git        │  Data & model versioning
  │  dvc.yaml         │  7-stage reproducible pipeline
  │  params.yaml      │  Centralized hyperparameters
  └──────────────────┘
```

## Component Details

### 1. Frontend (Streamlit, Container 1)
A Streamlit application rendering an interactive, rich UI with 8 navigation pages:
- **Predict**: Single scan upload with Grad-CAM explainability + Batch ZIP upload
- **Model Tracker**: Live MLflow experiment comparison with charts
- **Pipeline**: DVC DAG visualization, Airflow status, run history console
- **Apps Hub**: Links to MLflow, Grafana, Airflow, Prometheus, FastAPI Docs
- **Privacy Policy, FAQ, Contact Us, About**: Supporting pages

### 2. Backend (FastAPI + PyTorch, Container 2)
Provides the REST inference engine with 7 endpoints:
- `/health`, `/ready` — Container orchestration probes
- `/predict` — Single image classification
- `/predict_cam` — Classification with Grad-CAM heatmap overlay
- `/predict_bulk` — ZIP batch processing
- `/metrics` — Prometheus-compatible scrape target (8 custom metrics)
- `/info` — Model metadata endpoint

### 3. MLOps Tracking (MLflow, Container 6)
Tracks experiments across 5 model architectures:
- Parameters: learning rate, batch size, epochs, optimizer
- Metrics: macro_f1, val_acc, epoch_time, train_time
- Artifacts: model checkpoints, confusion matrices

### 4. Data Engineering (Apache Airflow, Container 7)
Automates ETL with 2 DAGs:
- `cerebronet_dag`: Data preparation and model training orchestration
- `cerebronet_scraper_dag`: Data ingestion from external sources
- **Email Alerting**: Automated success/failure notifications via SMTP (`.env` configuration)

### 5. Data Versioning (DVC + Git, Host)
Manages the reproducible ML pipeline:
- 7 stages: prepare → transform → train_baseline → train_mobilenet → train_efficientnet → train_svm → train_mlp → evaluate
- Parameterized via `params.yaml`
- Large data artifacts tracked without Git bloat

### 6. Observability (Prometheus + Grafana, Containers 3-5)
- Prometheus scrapes backend metrics every 15 seconds
- Node Exporter provides host-level CPU/memory data
- Grafana auto-provisions a 12-panel dashboard on startup:
  - Model status, inference latency percentiles, throughput, prediction distribution, confidence histogram, host resource utilization

## Network Topology
All containers communicate over the `cerebronet_network` Docker bridge network. The frontend resolves the backend via Docker DNS (`http://backend:8000`). No external cloud services are used.

## Security Measures
- File extension whitelist: `.jpg`, `.jpeg`, `.png`
- MIME type validation
- 10MB upload size limit
- EXIF metadata stripping (GPS, timestamps, device info)
- In-memory processing only — no disk persistence of user data
- CORS middleware configured for cross-origin requests
