import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

SAMPLE_PATIENT = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150.0,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0.0,
    "thal": 1.0
}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests" in response.text or "process_cpu_seconds" in response.text or "python_gc" in response.text

def test_predict_valid_data():
    response = client.post("/predict", json=SAMPLE_PATIENT)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_level"] in ("Low", "Moderate", "High")
    assert "diagnosis" in data

def test_predict_invalid_data():
    # Age below valid range (< 18)
    invalid_patient = SAMPLE_PATIENT.copy()
    invalid_patient["age"] = 10

    response = client.post("/predict", json=invalid_patient)
    assert response.status_code == 422
    assert "detail" in response.json()
