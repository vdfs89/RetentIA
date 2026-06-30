import json
import os
import pickle
import time

import pandas as pd
import torch

from src.features.preprocessor import ChurnPreprocessor
from src.models.mlp import ChurnMLP

# Caminhos configuráveis para permitir sandboxing de teste via monkeypatch
preprocessor_path = "models/preprocessor.pkl"
weights_path = "models/mlp_weights.pt"
threshold_path = "models/threshold.pkl"
LOG_FILE = "logs/input_samples.jsonl"

_preprocessor = None
_model = None
_threshold = 0.5


def load_model():
    global _preprocessor, _model, _threshold
    if _model is not None:
        return

    if not os.path.exists(preprocessor_path) or not os.path.exists(weights_path):
        raise FileNotFoundError("Arquivos do modelo não encontrados. Execute o treinamento primeiro.")

    _preprocessor = ChurnPreprocessor.load(preprocessor_path)

    dummy = pd.DataFrame(
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
    dim = _preprocessor.transform(dummy).shape[1]

    _model = ChurnMLP(dim)
    _model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    _model.eval()

    if os.path.exists(threshold_path):
        with open(threshold_path, "rb") as f:
            _threshold = pickle.load(f)


def log_input_sample(features: dict, prob: float, pred: bool):
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    log_entry = {
        "timestamp": time.time(),
        "features": features,
        "prediction": {
            "churn_probability": prob,
            "churn_prediction": pred,
            "threshold": _threshold,
        },
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def predict_one(features_dict: dict, log_to_file: bool = True) -> tuple:
    load_model()
    df = pd.DataFrame([features_dict])
    X = _preprocessor.transform(df)

    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        logit = _model(X_t)
        prob = torch.sigmoid(logit).item()

    pred = prob >= _threshold
    if log_to_file:
        log_input_sample(features_dict, prob, bool(pred))

    return prob, bool(pred), _threshold


def predict_batch(samples: list[dict], log_to_file: bool = True) -> list[dict]:
    """Roda predição para múltiplos clientes, reutilizando predict_one."""
    results = []
    for sample in samples:
        prob, pred, threshold = predict_one(sample, log_to_file=log_to_file)
        results.append(
            {
                "churn_probability": prob,
                "churn_prediction": pred,
                "threshold": threshold,
            }
        )
    return results
