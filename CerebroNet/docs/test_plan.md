# Test Plan & Cases

## Overview
This document specifies the software test plan and execution report for CerebroNet, designed to ensure inference stability, UI robustness, and API security.

## Acceptance Criteria
- [x] API must correctly identify standard images and reject invalid extensions/payload sizes.
- [x] UI color coding must match the prediction output classes.
- [x] Model latency must remain under 500ms on a cold start CPU.
- [x] EXIF data must be stripped prior to inference.

## Test Cases Enlistment
| Test ID | Component | Description | Expected Result | Status |
|---|---|---|---|---|
| TC-01 | FastAPI | Upload valid Glioma JPEG. | HTTP 200, class="glioma", confidence > 0.5 | PASSED |
| TC-02 | FastAPI | Upload 15MB file. | HTTP 400, "File too large" | PASSED |
| TC-03 | FastAPI | Upload .txt file. | HTTP 400, "Invalid file type" | PASSED |
| TC-04 | Frontend | Predict Meningioma. | UI injects `180deg` hue-rotate (Yellow) | PASSED |
| TC-05 | Privacy | Upload image with GPS EXIF. | EXIF is stripped before model prediction | PASSED |
| TC-06 | Telemetry | Hit `/predict` endpoint 5 times. | Prometheus `cerebronet_requests_total` = 5 | PASSED |

## Test Report
- **Total Test Cases**: 6
- **Passed**: 6
- **Failed**: 0
- **Coverage**: Core inference loop and security barriers have 100% automated test coverage in `tests/test_api.py`.
