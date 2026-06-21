# CRISP-DM — RetentIA

## Phase 1 — Business Understanding

Churn erodes recurring revenue. A telecom loses not just the current month's billing but the entire remaining Customer Lifetime Value (CLV) when a customer cancels. The business goal is to **prioritize retention actions** (discount offers, proactive support calls) for customers most likely to churn, optimizing the tradeoff between the cost of losing a customer (FN) and the cost of a wasted retention offer (FP).

## Phase 2 — Data Understanding

- **Dataset:** IBM Telco Customer Churn (~7,043 customers, 19 features, binary target `Churn`).
- **Target balance:** Churn ≈ 26.5% — moderately imbalanced.
- **Key gotcha:** `TotalCharges` column contains 11 blank spaces (not NaN) for customers with `tenure==0`. This is **structural missingness** (new customers, never billed), not random — validated by checking that all blanks correspond to `tenure==0`.
- **Feature types:** 3 numerical (`tenure`, `MonthlyCharges`, `TotalCharges`), 16 categorical (contract, service, demographic).

## Phase 3 — Data Preparation

- `TotalCharges` blanks → `pd.to_numeric(errors="coerce").fillna(0.0)` with structural validation.
- OneHotEncoder (drop=first, handle_unknown=ignore) on 16 categoricals.
- StandardScaler on 3 numericals.
- Stratified split: Train 60% / Validation 20% / Test 20%.
- **Anti-leakage:** Preprocessor `.fit()` only on train; `.transform()` on val and test.

## Phase 4 — Modeling

Three models trained and compared:

1. **DummyClassifier** (most_frequent): Baseline — always predicts majority class.
2. **LogisticRegression** (class_weight=balanced, max_iter=1000): Strong linear baseline.
3. **MLP** (PyTorch): [32→16→1], BatchNorm + ReLU + Dropout(0.2), BCEWithLogitsLoss with pos_weight=2.73, Adam lr=0.005, batch_size=64, early stopping at epoch 19 (patience=10).

## Phase 5 — Evaluation

| Model         | Accuracy | Precision | Recall | F1     |
|---------------|----------|-----------|--------|--------|
| Dummy         | 0.7346   | 0.0000    | 0.0000 | 0.0000 |
| LogReg @0.5   | 0.7381   | 0.5043    | 0.7807 | 0.6128 |
| MLP @0.36     | 0.6558   | 0.4295    | 0.9037 | 0.5823 |

**Cost-sensitive threshold:** 0.36 (tuned on validation set). With C_FN=500 and C_FP=100, the optimal operating point favors recall (catching churners) over precision (avoiding false alarms). The MLP captures 90.4% of actual churners.

**Honest assessment:** The MLP does not outperform LogReg on F1 or accuracy. The comparison is not fully apples-to-apples (LogReg at 0.5 vs MLP at 0.36). The primary contribution is the cost-sensitive threshold, not the model architecture. See MODEL_CARD.md for detailed analysis.

## Phase 6 — Deployment

- **API:** FastAPI with `/predict`, `/health`, `/metrics` (Prometheus).
- **Service layer:** `model_service.py` loads preprocessor + MLP + threshold at startup.
- **Drift logging:** Each `/predict` call appends input to `logs/input_samples.jsonl`.
- **Batch:** `scripts/run_batch.py` — vectorized inference without drift log pollution.
- **CI/CD:** GitHub Actions (lint + test on PR, build Docker on push to main).
- **Public deploy:** Render free tier (cold start ~50s documented).
