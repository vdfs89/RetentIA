import json
import os

import torch

import src.services.model_service as model_service
from src.data.ingest import ingest_data


def run_batch():
    input_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_path = "data/processed/batch_output.jsonl"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model_service.load_model()

    df = ingest_data(input_path)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    # Vectorized transform (no per-row loop)
    X_processed = model_service._preprocessor.transform(X)

    print(f"Starting vectorized batch prediction on {len(X_processed)} rows...")

    with torch.no_grad():
        X_t = torch.FloatTensor(X_processed)
        logits = model_service._model(X_t)
        probs = torch.sigmoid(logits).numpy().squeeze()

    preds = probs >= model_service._threshold

    with open(output_path, "w") as f:
        for idx, (prob, pred) in enumerate(zip(probs, preds)):
            out_row = {
                "customerID": df.loc[idx, "customerID"],
                "churn_probability": float(prob),
                "churn_prediction": bool(pred),
                "threshold": float(model_service._threshold),
            }
            f.write(json.dumps(out_row) + "\n")

    print(f"Batch prediction finished. Output saved to {output_path}")


if __name__ == "__main__":
    run_batch()
