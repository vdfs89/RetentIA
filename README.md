# RetentIA

> Predict customer churn before it happens.

Churn prediction project with MLP (PyTorch), FastAPI, MLflow and MLOps practices.
FIAP Tech Challenge — Pos-Tech MLET.

## Quickstart

```bash
make install
make train        # downloads dataset + trains baselines + MLP + cost threshold
make test         # runs all tests (isolated, won't touch trained models)
make run          # API at http://localhost:8000 (/docs, /health, /predict, /metrics)
make run-batch    # batch inference -> data/processed/batch_output.jsonl
```

## Structure

src/
main.py                  # thin FastAPI app (routers + latency middleware)
routes/metrics.py        # /health + /metrics (Prometheus)
routes/predict.py        # /predict -> services
services/model_service.py # load_model, predict_one, drift log
api/schemas.py           # Pydantic v2 contract (19 features)
data/ingest.py           # ingestion + TotalCharges gotcha fix
features/columns.py      # single source of truth for features
features/preprocessor.py # ColumnTransformer (OHE + scaling)
models/mlp.py            # MLP architecture
validation/schemas.py    # pandera DataFrame validation
cost/threshold.py        # cost-sensitive threshold optimization
train.py                 # training pipeline (baselines + MLP + MLflow)
scripts/run_batch.py       # vectorized batch inference
tests/                     # test_health, test_metrics, test_predict, test_model, test_schema
docs/                      # MODEL_CARD, CRISP_DM, ML_CANVAS, DEPLOY, MONITORING


## Dataset
IBM Telco Customer Churn (~7,043 customers). Auto-downloaded on first `make train`.

## Results
`(FILL AFTER make train — see docs/MODEL_CARD.md)`
