# Low-Level Design (LLD)

## Components
### `src/api/main.py`
The FastAPI application responsible for serving the MobileNetV2 model. 
It implements comprehensive logging and exception handling using standard Python `logging` and FastAPI `HTTPException`.

### Endpoint Definitions

#### `POST /predict`
Performs an inference pass on the uploaded MRI scan.
- **Input**: `multipart/form-data` containing a `file` field (must be JPG, JPEG, or PNG, max 10MB).
- **Processing**:
  1. Validates file extension, content-type, and size.
  2. Strips EXIF metadata.
  3. Preprocesses the image (Resize to 224x224, CenterCrop, Normalize).
  4. Passes the tensor to the MobileNetV2 model.
  5. Applies Softmax to logits.
- **Output (JSON)**:
  ```json
  {
      "class": "glioma",
      "confidence": 0.992,
      "inference_time_ms": 145.2,
      "probabilities": {
          "glioma": 0.992,
          "meningioma": 0.005,
          "notumor": 0.001,
          "pituitary": 0.002
      }
  }
  ```

#### `GET /info`
Returns telemetry about the currently loaded model.
- **Input**: None.
- **Output (JSON)**:
  ```json
  {
      "model": "MobileNetV2",
      "input_size": "224x224",
      "classes": ["glioma", "meningioma", "notumor", "pituitary"],
      "macro_f1": 0.948
  }
  ```

#### `GET /metrics`
Prometheus exporter endpoint scraping metrics (Request Count, Inference Latency, Prediction Counter).
