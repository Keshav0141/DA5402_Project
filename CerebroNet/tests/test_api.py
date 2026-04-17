import pytest
import sys
import os
import io
from PIL import Image

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
))

from fastapi.testclient import TestClient
from src.api.main import app


def make_image():
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


# Use context manager to trigger startup event
@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_has_status(self, client):
        data = client.get("/health").json()
        assert "status" in data

    def test_health_status_is_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_has_model_loaded(self, client):
        data = client.get("/health").json()
        assert "model_loaded" in data

    def test_health_model_is_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] == True

    def test_ready_returns_200(self, client):
        assert client.get("/ready").status_code == 200

    def test_ready_has_status(self, client):
        data = client.get("/ready").json()
        assert "status" in data


class TestInfoEndpoint:

    def test_info_returns_200(self, client):
        assert client.get("/info").status_code == 200

    def test_info_has_model(self, client):
        data = client.get("/info").json()
        assert "model" in data

    def test_info_has_classes(self, client):
        data = client.get("/info").json()
        assert "classes" in data

    def test_info_has_4_classes(self, client):
        data = client.get("/info").json()
        assert len(data["classes"]) == 4

    def test_info_classes_correct(self, client):
        data = client.get("/info").json()
        expected = {"glioma", "meningioma", "notumor", "pituitary"}
        assert set(data["classes"]) == expected

    def test_info_has_f1(self, client):
        data = client.get("/info").json()
        assert "macro_f1" in data

    def test_info_f1_above_threshold(self, client):
        data = client.get("/info").json()
        assert data["macro_f1"] >= 0.90


class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_metrics_has_cerebronet(self, client):
        assert b"cerebronet" in client.get("/metrics").content

    def test_metrics_has_requests_total(self, client):
        assert b"cerebronet_requests_total" in client.get("/metrics").content

    def test_metrics_has_latency(self, client):
        assert b"cerebronet_inference_latency" in client.get("/metrics").content


class TestPredictEndpoint:

    def test_predict_no_file_returns_422(self, client):
        assert client.post("/predict").status_code == 422

    def test_predict_invalid_file_returns_400(self, client):
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_predict_valid_image_returns_200(self, client):
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", make_image(), "image/jpeg")}
        )
        assert response.status_code == 200

    def test_predict_has_prediction(self, client):
        data = client.post(
            "/predict",
            files={"file": ("test.jpg", make_image(), "image/jpeg")}
        ).json()
        assert "prediction" in data

    def test_predict_has_confidence(self, client):
        data = client.post(
            "/predict",
            files={"file": ("test.jpg", make_image(), "image/jpeg")}
        ).json()
        assert "confidence" in data

    def test_predict_confidence_range(self, client):
        data = client.post(
            "/predict",
            files={"file": ("test.jpg", make_image(), "image/jpeg")}
        ).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_class_is_valid(self, client):
        data = client.post(
            "/predict",
            files={"file": ("test.jpg", make_image(), "image/jpeg")}
        ).json()
        assert data["prediction"] in {
            "glioma", "meningioma", "notumor", "pituitary"
        }

    def test_predict_has_all_probs(self, client):
        data = client.post(
            "/predict",
            files={"file": ("test.jpg", make_image(), "image/jpeg")}
        ).json()
        assert "all_probs" in data
        assert len(data["all_probs"]) == 4

    def test_predict_has_latency(self, client):
        data = client.post(
            "/predict",
            files={"file": ("test.jpg", make_image(), "image/jpeg")}
        ).json()
        assert "latency_ms" in data
        assert data["latency_ms"] > 0