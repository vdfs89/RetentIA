# Deploy Architecture — RetentIA

- **App:** FastAPI + Uvicorn (`src.main:app`).
- **Artifacts:** `models/preprocessor.pkl`, `models/mlp_weights.pt`, `models/threshold.pkl`.
- **Release strategy:** `render.yaml` runs `python -m src.train` at build time.
- **Cold start (free tier):** service sleeps after ~15min; first request ~50s.
- **Observability:** `X-Process-Time` header; Prometheus `/metrics`; drift log `logs/input_samples.jsonl`.
