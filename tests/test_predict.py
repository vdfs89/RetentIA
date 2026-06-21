import pickle

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

import src.services.model_service as model_service
from src.features.preprocessor import ChurnPreprocessor
from src.main import app
from src.models.mlp import ChurnMLP


@pytest.fixture(autouse=True)
def setup_mock_model(tmp_path, monkeypatch):
    # Temporary paths — never touches real models/
    preprocessor_path = tmp_path / "preprocessor.pkl"
    weights_path = tmp_path / "mlp_weights.pt"
    threshold_path = tmp_path / "threshold.pkl"
    log_path = tmp_path / "input_samples.jsonl"

    preprocessor = ChurnPreprocessor()
    df = pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        ]
    )
    preprocessor.fit(df)
    preprocessor.save(str(preprocessor_path))

    dim = preprocessor.transform(df).shape[1]
    model = ChurnMLP(dim)
    torch.save(model.state_dict(), str(weights_path))
    with open(threshold_path, "wb") as f:
        pickle.dump(0.5, f)

    # Monkeypatch paths in model_service so it never touches real models/
    monkeypatch.setattr(model_service, "preprocessor_path", str(preprocessor_path))
    monkeypatch.setattr(model_service, "weights_path", str(weights_path))
    monkeypatch.setattr(model_service, "threshold_path", str(threshold_path))
    monkeypatch.setattr(model_service, "LOG_FILE", str(log_path))

    # Reset cached globals
    monkeypatch.setattr(model_service, "_preprocessor", None)
    monkeypatch.setattr(model_service, "_model", None)
    monkeypatch.setattr(model_service, "_threshold", 0.5)


def test_predict_endpoint():
    client = TestClient(app)
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data


def test_predict_invalid_category_422():
    client = TestClient(app)
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Forever",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
