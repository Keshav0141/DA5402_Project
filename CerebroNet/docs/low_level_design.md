# Low-Level Design (LLD)

## Components

### `src/api/main.py`
The FastAPI application responsible for serving the MobileNetV2 model.
It implements comprehensive logging via Python `logging` and exception handling using FastAPI `HTTPException`.

### `src/pipeline/prepare.py`
Data preparation stage. Reads raw MRI images from `data/raw/`, resizes them to 224×224, and writes to `data/v1_resized/`.

### `src/pipeline/transform.py`
Data augmentation and train/test split. Applies random flips, rotations, and color jitter. Outputs to `data/v2_augmented/`.

### `src/pipeline/evaluate.py`
Model evaluation stage. Loads the best MobileNetV2 checkpoint, runs inference on the test set, and generates `eval/confusion_matrix.csv`.

---

## API Endpoint Definitions

### `GET /health`
Liveness probe for container orchestration (Docker healthcheck).
- **Input**: None
- **Processing**: Checks if model object is not None
- **Output (JSON)**:
  ```json
  {
      "status": "healthy",
      "model_loaded": true,
      "device": "cpu"
  }
  ```
- **Error**: None (always returns 200)

---

### `GET /ready`
Readiness probe indicating the model is loaded and accepting traffic.
- **Input**: None
- **Processing**: Returns 503 if model is None, 200 otherwise
- **Output (JSON)**:
  ```json
  { "status": "ready" }
  ```
- **Error**: HTTP 503 `"Model not loaded yet"` if model initialization failed

---

### `POST /predict`
Performs a single inference pass on the uploaded MRI scan.
- **Input**: `multipart/form-data` containing a `file` field (must be JPG, JPEG, or PNG, max 10MB)
- **Processing**:
  1. Validates file extension, content-type, and size
  2. Opens and validates the image (rejects corrupt files)
  3. Strips EXIF metadata (GPS, timestamps, device info)
  4. Preprocesses: Resize 224×224 → ToTensor → Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
  5. Forward pass through MobileNetV2
  6. Softmax on logits → confidence + predicted class
  7. Deletes all in-memory image data immediately
- **Output (JSON)**:
  ```json
  {
      "prediction": "glioma",
      "confidence": 0.9920,
      "all_probs": {
          "glioma": 0.9920,
          "meningioma": 0.0050,
          "notumor": 0.0010,
          "pituitary": 0.0020
      },
      "latency_ms": 145.2,
      "model": "MobileNetV2",
      "device": "cpu",
      "privacy": {
          "image_stored": false,
          "exif_stripped": true
      }
  }
  ```
- **Errors**:
  - HTTP 400: Invalid file type, invalid content type, file too large, corrupt image
  - HTTP 503: Model not loaded
  - HTTP 500: Internal server error during prediction

---

### `POST /predict_cam`
Performs inference with Grad-CAM explainability overlay.
- **Input**: `multipart/form-data` containing a `file` field (JPG/JPEG/PNG)
- **Processing**:
  1. Same validation and preprocessing as `/predict`
  2. Runs Grad-CAM on `model.features[-1]` (last convolutional layer)
  3. Generates heatmap → blends with original image (alpha=0.5)
  4. Encodes overlay as base64 JPEG string
- **Output (JSON)**:
  ```json
  {
      "prediction": "meningioma",
      "confidence": 0.8750,
      "all_probs": { ... },
      "cam_base64": "<base64-encoded JPEG string>",
      "latency_ms": 320.5
  }
  ```
- **Errors**:
  - HTTP 503: Model not loaded
  - HTTP 500: Error generating Grad-CAM

---

### `POST /predict_bulk`
Accepts a ZIP archive of images and returns batch predictions.
- **Input**: `multipart/form-data` containing a `file` field (must be a `.zip` archive)
- **Processing**:
  1. Validates the file is a valid ZIP archive
  2. Iterates over all image files inside the ZIP (skips directories, non-image files)
  3. For each image: open → strip EXIF → preprocess → inference
  4. Collects per-image results into a list
- **Output (JSON)**:
  ```json
  {
      "batch_size": 3,
      "results": [
          { "filename": "scan1.jpg", "prediction": "glioma", "confidence": 0.9912 },
          { "filename": "scan2.jpg", "prediction": "notumor", "confidence": 0.9801 },
          { "filename": "scan3.jpg", "error": "Failed to process image" }
      ],
      "latency_ms": 890.3,
      "model": "MobileNetV2"
  }
  ```
- **Errors**:
  - HTTP 400: Not a ZIP file, invalid ZIP archive
  - HTTP 503: Model not loaded
  - HTTP 500: Internal server error

---

### `GET /metrics`
Prometheus-compatible metrics endpoint. Scraped by Prometheus every 15 seconds.
- **Input**: None
- **Output**: Prometheus text format (`text/plain; version=0.0.4; charset=utf-8`)
- **Metrics Exposed**:
  | Metric | Type | Labels |
  |---|---|---|
  | `cerebronet_requests_total` | Counter | endpoint, status, mode, client_ip |
  | `cerebronet_inference_latency_seconds` | Histogram | buckets: 0.05-5.0s |
  | `cerebronet_predictions_total` | Counter | predicted_class |
  | `cerebronet_confidence_score` | Histogram | buckets: 0.1-1.0 |
  | `cerebronet_model_loaded` | Gauge | — |
  | `cerebronet_active_requests` | Gauge | — |
  | `cerebronet_payload_size_bytes` | Summary | — |
  | `cerebronet_rejected_uploads_total` | Counter | reason |

---

### `GET /info`
Returns metadata about the currently deployed model.
- **Input**: None
- **Output (JSON)**:
  ```json
  {
      "model": "MobileNetV2",
      "classes": ["glioma", "meningioma", "notumor", "pituitary"],
      "input_size": "224x224",
      "framework": "PyTorch",
      "macro_f1": 0.9483,
      "device": "cpu",
      "security": {
          "max_file_size_mb": 10,
          "allowed_formats": [".jpg", ".jpeg", ".png"],
          "exif_stripping": true,
          "image_storage": false
      }
  }
  ```

---

## Class Definitions

### `GradCAM`
Implements Gradient-weighted Class Activation Mapping for model explainability.
- **`__init__(model, target_layer)`**: Registers forward and backward hooks on the target convolutional layer
- **`generate(input_tensor, class_idx=None)`**: Performs a forward+backward pass, computes weighted activation maps, returns a 2D heatmap numpy array

### `get_cam_image_base64(heatmap, original_img)`
Utility function that resizes the Grad-CAM heatmap, applies a JET colormap, blends with the original image at 50% alpha, and returns a base64-encoded JPEG string.

### `strip_exif(image)`
Privacy utility that creates a clean copy of the PIL Image with all EXIF metadata removed.

### `validate_file_security(filename, content_type, file_size)`
Security validation gate. Checks file extension against whitelist, validates MIME type, enforces 10MB size limit.

### `build_mobilenet(num_classes)`
Factory function that constructs a MobileNetV2 architecture with a custom classifier head (Dropout → Linear 256 → ReLU → Dropout → Linear num_classes).
