# CerebroNet User Manual

## Welcome to CerebroNet
CerebroNet is an AI-powered diagnostic web platform designed to analyze brain MRI scans and classify them into four categories:
1. Glioma
2. Meningioma
3. Pituitary Tumor
4. No Tumor

## How to Use the Platform

### 1. The Main Dashboard ("Predict" Tab)
When you navigate to the main page, you will see the Upload terminal.
1. Drag and drop your MRI scan (JPEG or PNG format) into the glowing purple upload box.
2. The image will be processed securely in-memory.
3. The platform will instantly render the prediction results, displaying the most likely class, the prediction confidence, and the real-time inference latency.
4. **Visual Indicator**: The glowing digital brain will shift color based on the result:
   - **Red**: Glioma
   - **Yellow**: Meningioma
   - **Blue**: Pituitary
   - **Green**: No Tumor

### 2. Monitoring the Model ("Model Tracker" Tab)
Use the navigation sidebar to access the Model Tracker. This page provides live telemetry on the deployed AI model, comparing its performance (F1-score) and training speed against other architectures.

### 3. Service Management ("Apps Hub" Tab)
This page is for technical administrators. It provides secure links to the backend MLOps orchestration tools:
- **MLflow**: View historical model training experiments.
- **Airflow**: Monitor data ETL pipelines.
- **Grafana & Prometheus**: Monitor live server traffic, latency, and prediction statistics.

### Security Note
Your uploads are never saved to disk. All patient metadata (EXIF data) is automatically stripped upon upload to ensure strict privacy compliance.
