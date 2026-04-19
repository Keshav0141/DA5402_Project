# High-Level Design (HLD)

## Overview
CerebroNet is an end-to-end Machine Learning pipeline that classifies brain MRI scans into 4 categories: Glioma, Meningioma, Pituitary, and No Tumor. The system strictly follows MLOps practices, enforcing automation, reproducibility, and environment parity.

## Design Choices & Rationale
1. **Model Selection**: MobileNetV2 was chosen as the production model due to its high accuracy (94.8% Macro F1) and extremely low inference latency. EfficientNet-B0 was evaluated but rejected due to high epoch training times (~650s vs 80s).
2. **Containerization**: Docker and Docker Compose are used to package the Frontend, Backend, MLflow, Prometheus, and Grafana into isolated microservices. This guarantees environment parity across development and production.
3. **Data Privacy**: The system is designed to be fully in-memory. Inference images uploaded via the frontend are sent to the FastAPI backend, decoded in RAM, stripped of EXIF metadata, predicted, and immediately discarded via garbage collection.
4. **Data Versioning**: DVC is utilized alongside Git to manage large datasets.
5. **Observability**: Prometheus instrumentation is natively embedded into the FastAPI application.
