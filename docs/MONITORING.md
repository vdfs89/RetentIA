# Monitoring Playbook — RetentIA

## Service Health
- `/health` → liveness check.
- `/metrics` → Prometheus counters and latency histogram.
- Latency header `X-Process-Time` per request.

## Model Quality
- **Data drift:** monitor distribution of `tenure`, `MonthlyCharges`, `Contract`, `PaymentMethod` vs training baseline.
- **Prediction drift:** score distribution and positive rate at threshold.
- **Performance:** when actual churn arrives, recalculate recall/precision/cost.

## Retraining Triggers
- Significant drift, recall drop, or change in cost assumptions (CLV/offer cost).

## Drift Sampling
- Each `/predict` appends to `logs/input_samples.jsonl` (input for Evidently/NannyML).
- Batch inference (`scripts/run_batch.py`) does NOT log to drift file (avoids pollution).
