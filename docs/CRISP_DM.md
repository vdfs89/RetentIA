# CRISP-DM — RetentIA

## Phase 1 — Business Understanding
Churn erodes recurring revenue. Goal: prioritize retention based on predicted risk,
optimizing cost (FN = lost customer ≈ CLV; FP = wasted offer).

## Phase 2 — Data Understanding
IBM Telco Customer Churn (~7k customers, 19 features, target `Churn`).
Imbalanced (churn ≈ 26.5% `CONFIRM`). TotalCharges gotcha (blanks at tenure==0).

## Phase 3 — Data Preparation
Ingest + TotalCharges fix (→0.0). OneHot on categoricals, StandardScaler on numericals.
Stratified train/val/test split. Preprocessor fit only on train (anti-leakage).

## Phase 4 — Modeling — `(FILL AFTER make train)`
Baselines (Dummy, LogReg) vs MLP (early stopping, pos_weight, DataLoader). See MODEL_CARD.

## Phase 5 — Evaluation — `(FILL AFTER make train)`
Comparison with 4 metrics + cost analysis FP/FN + threshold selection on validation set.

## Phase 6 — Deployment
FastAPI API (`/predict`, `/health`, `/metrics`) + public deploy (Render).
Monitoring: drift log + Prometheus metrics. See MONITORING.md.
