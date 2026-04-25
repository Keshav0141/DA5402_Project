# Test Plan & Cases

## Overview
This document specifies the software test plan and execution report for CerebroNet, designed to ensure inference stability, UI robustness, API security, and pipeline correctness.

## Acceptance Criteria
- [x] API must correctly identify standard images and reject invalid extensions/payload sizes.
- [x] UI color coding must match the prediction output classes.
- [x] Model latency must remain under 500ms on a cold start CPU.
- [x] EXIF data must be stripped prior to inference.
- [x] All 7 API endpoints must return correct status codes.
- [x] Batch mode must correctly process ZIP archives with mixed valid/invalid files.

## Test Cases Enlistment

### API Tests (`tests/test_api.py`)
| Test ID | Component | Description | Expected Result | Status |
|---|---|---|---|---|
| TC-01 | Health | `GET /health` returns 200 | HTTP 200, status=healthy, model_loaded=true | PASSED |
| TC-02 | Health | `GET /ready` returns 200 | HTTP 200, status=ready | PASSED |
| TC-03 | Info | `GET /info` returns model metadata | model=MobileNetV2, 4 classes, F1 ≥ 0.90 | PASSED |
| TC-04 | Metrics | `GET /metrics` returns Prometheus format | Contains `cerebronet_requests_total`, `cerebronet_inference_latency` | PASSED |
| TC-05 | Predict | Upload valid JPEG → correct prediction | HTTP 200, prediction ∈ {glioma, meningioma, notumor, pituitary} | PASSED |
| TC-06 | Predict | Upload invalid .txt file | HTTP 400, "Invalid file type" | PASSED |
| TC-07 | Predict | Upload 15MB oversized file | HTTP 400, "File too large" | PASSED |
| TC-08 | Predict | Verify confidence in [0, 1] range | 0.0 ≤ confidence ≤ 1.0 | PASSED |
| TC-09 | Predict | Verify `all_probs` contains 4 classes | len(all_probs) == 4 | PASSED |
| TC-10 | Predict | Verify latency_ms > 0 | latency_ms is a positive float | PASSED |
| TC-11 | Predict_cam | Upload valid JPEG → Grad-CAM overlay | HTTP 200, cam_base64 is non-empty string | PASSED |
| TC-12 | Predict_bulk | Upload valid ZIP → batch results | HTTP 200, results array with per-file predictions | PASSED |

### Frontend Tests (`tests/test_ui_colors.py`)
| Test ID | Component | Description | Expected Result | Status |
|---|---|---|---|---|
| TC-13 | Frontend | Glioma prediction → Red hue (155deg) | CSS injects `hue-rotate(155deg)` | PASSED |
| TC-14 | Frontend | Meningioma prediction → Yellow hue (180deg) | CSS injects `hue-rotate(180deg)` | PASSED |
| TC-15 | Frontend | No Tumor prediction → Green hue (240deg) | CSS injects `hue-rotate(240deg)` | PASSED |
| TC-16 | Frontend | Pituitary prediction → Blue hue (0deg) | CSS injects `hue-rotate(0deg)` | PASSED |

### Integration Tests
| Test ID | Component | Description | Expected Result | Status |
|---|---|---|---|---|
| TC-17 | Privacy | Upload image with GPS EXIF data | EXIF is stripped before model prediction | PASSED |
| TC-18 | Telemetry | Hit `/predict` endpoint 5 times | Prometheus `cerebronet_requests_total` increments by 5 | PASSED |
| TC-19 | Pipeline | Run `dvc repro --dry` | All 7 stages listed in correct dependency order | PASSED |

## Test Report
- **Total Test Cases**: 19
- **Passed**: 19
- **Failed**: 0
- **Coverage**: Core inference loop and security barriers have 100% automated test coverage via `tests/test_api.py` (22 pytest assertions). UI color tests verified via `tests/test_ui_colors.py`.

## Test Execution
```bash
# Run all tests inside the backend container
docker exec cerebronet_backend pytest tests/ -v

# Run specific test suite
docker exec cerebronet_backend pytest tests/test_api.py -v
```
