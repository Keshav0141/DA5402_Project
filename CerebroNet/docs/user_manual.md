# CerebroNet User Manual

## Welcome to CerebroNet
CerebroNet is an AI-powered diagnostic web platform designed to analyze brain MRI scans and classify them into four categories:
1. **Glioma** — A malignant tumor originating from glial cells
2. **Meningioma** — A usually benign tumor arising from the meninges
3. **Pituitary Tumor** — A growth in the pituitary gland
4. **No Tumor** — Healthy brain scan with no detectable abnormality

The platform is powered by a MobileNetV2 deep learning model achieving 94.8% Macro F1 Score, with model explainability via Grad-CAM heatmaps.

---

## Getting Started

### Launching the Application
1. Ensure Docker Desktop is running on your machine.
2. Open a terminal in the CerebroNet project folder and run:
   ```bash
   docker compose up -d
   ```
3. Wait approximately 60 seconds for all services to initialize.
4. Open your web browser and navigate to: **http://localhost:8501**

### Navigating the Platform
The left sidebar provides access to all 8 sections of the platform:
- **Predict** — Upload and classify brain MRI scans
- **Model Tracker** — Compare model performance metrics
- **Pipeline** — Visualize the end-to-end ML pipeline
- **Apps Hub** — Access backend MLOps tool UIs
- **Privacy Policy** — Data handling and security information
- **FAQ** — Frequently asked questions
- **Contact Us** — Submit support inquiries
- **About** — Project information

---

## Page-by-Page Guide

### 1. Predict (Main Dashboard)
This is the primary page for classifying brain MRI scans.

#### Single Scan Mode
1. Select **"Single Scan"** from the mode selector (selected by default).
2. Drag and drop your MRI scan (JPEG or PNG, max 10MB) into the upload area.
3. The platform will automatically:
   - Strip all EXIF metadata (GPS, timestamps, device info) for privacy
   - Process the image securely in-memory
   - Display the prediction result with confidence score and latency
4. **Prediction Result Card**: Shows the predicted class with a color-coded indicator:
   - 🔴 **Red** — Glioma
   - 🟡 **Yellow** — Meningioma
   - 🟢 **Green** — No Tumor
   - 🔵 **Blue** — Pituitary
5. **Grad-CAM Explainability**: Below the result, a side-by-side comparison shows your original scan next to an AI Heatmap highlighting exactly which regions the model focused on.
6. **Probability Chart**: A horizontal bar chart displays the confidence distribution across all 4 classes.
7. **Brain Visual Indicator**: The animated digital brain on the hero section shifts its glow color to match the prediction class.

#### Batch Mode (ZIP Upload)
1. Select **"Batch Mode (ZIP)"** from the mode selector.
2. Upload a ZIP archive containing multiple MRI scan images.
3. The platform will process each image and display a results table with:
   - Filename, Prediction, Confidence, and Latency per image.

### 2. Model Tracker
This page provides live telemetry on the deployed AI model and compares it against all trained architectures.

- **Production Model Metrics**: Shows the deployed model (MobileNetV2), its F1 score, and training speed.
- **Run Comparison Matrix**: A sortable table of all 5 trained models (EfficientNet-B0, MobileNetV2, BaseCNN, SVM, MLP) with Macro F1, Validation Accuracy, and Speed per Epoch.
- **Graphic Analysis**: A horizontal bar chart visualizing Macro F1 scores across all architectures.

Data is fetched live from MLflow when available, with a static fallback if the MLflow container is unavailable.

### 3. Pipeline
This page visualizes the complete ML pipeline from raw data to deployed model.

- **DVC Pipeline DAG**: An ASCII diagram showing the 7-stage pipeline: `data/raw → prepare → transform → 5 training stages → evaluate`.
- **Pipeline Speed & Throughput**: Metrics cards showing preparation time (~12s), augmentation time (~45s), best training speed (80.5s/epoch), and inference latency (<200ms).
- **Stage Duration Table**: Detailed table with command, duration, and output for each pipeline stage.
- **Airflow Orchestration Status**: Live table of Airflow DAG runs with state, start time, and duration.
- **Run History & Error Console**: MLflow experiment run log showing status (✅ FINISHED / ❌ FAILED), duration, and Macro F1 for each training run.

### 4. Apps Hub
This page provides direct links to the backend MLOps tool interfaces:
- **MLflow** (localhost:5000) — Model registry and experiment tracking
- **Airflow** (localhost:8080) — Data engineering DAG orchestration (Credentials: admin/admin)
- **Grafana** (localhost:3001) — Real-time metrics dashboards (Credentials: admin/admin)
- **Prometheus** (localhost:9090) — Metrics collection and alert rules
- **FastAPI Docs** (localhost:8000/docs) — Swagger UI for backend API
- **Raw Metrics** (localhost:8000/metrics) — Prometheus scrape target

### 5. Privacy Policy
Details the platform's privacy-first architecture:
- All processing is in-memory only — no images are saved to disk
- EXIF metadata (GPS, timestamps, device info) is stripped automatically
- Images are garbage collected immediately after inference

### 6. FAQ
Answers to common questions about supported formats, data persistence, the inference engine, and why MobileNetV2 was chosen over EfficientNet.

### 7. Contact Us
A form to submit support inquiries. All submissions are logged locally.

### 8. About
Project information and credits.

---

## Supported File Formats
| Format | Single Scan | Batch Mode |
|---|---|---|
| JPEG (.jpg, .jpeg) | ✅ | ✅ (inside ZIP) |
| PNG (.png) | ✅ | ✅ (inside ZIP) |
| ZIP (.zip) | ❌ | ✅ (container) |
| Other formats | ❌ Rejected | ❌ Skipped |

**Maximum file size**: 10MB per image.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Page shows "Connection Error" | Ensure `docker compose up -d` has been run and wait ~60s for backend startup |
| Sidebar not visible | Click the `>` arrow button in the top-left corner of the page |
| Prediction shows "Model not loaded" | The backend is still initializing. Wait 30 seconds and retry |
| Batch mode returns empty results | Ensure the ZIP contains valid .jpg/.jpeg/.png files (not nested in folders) |
