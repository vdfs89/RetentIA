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

Four models trained and compared:

1. **DummyClassifier** (most_frequent): Baseline — always predicts majority class.
2. **LogisticRegression** (class_weight=balanced, max_iter=1000): Strong linear baseline.
3. **XGBClassifier** (n_estimators=200, max_depth=4, lr=0.05, scale_pos_weight=2.73): Gradient boosting baseline.
4. **MLP** (PyTorch): [32→16→1], BatchNorm + ReLU + Dropout(0.2), BCEWithLogitsLoss with pos_weight=2.73, Adam lr=0.005, batch_size=64, early stopping (patience=10, max 100 epochs).

## Phase 5 — Evaluation

| Modelo         | Accuracy | Precision | Recall | F1     | ROC-AUC | PR-AUC |
|----------------|----------|-----------|--------|--------|---------|--------|
| Dummy          | 0.7346   | 0.0000    | 0.0000 | 0.0000 | 0.5000  | 0.2654 |
| LogReg @0.5    | 0.7381   | 0.5043    | 0.7807 | 0.6128 | 0.8429  | 0.6340 |
| XGBoost @0.5   | 0.7480   | 0.5167    | 0.7861 | 0.6236 | 0.8420  | 0.6534 |
| MLP @0.37      | 0.6828   | 0.4510    | 0.8984 | 0.6005 | 0.8453  | 0.6372 |

ROC-AUC plateau at ~0.842–0.845 across all non-dummy models. XGBoost leads on F1/PR-AUC at
default threshold. MLP @0.37 leads on recall via cost-sensitive threshold (C_FN=500, C_FP=100).
Honest finding: architecture does not drive performance on this dataset — features do.

## Phase 6 — Deployment

- **API:** FastAPI with `/predict`, `/health`, `/metrics` (Prometheus).
- **Service layer:** `model_service.py` loads preprocessor + MLP + threshold at startup.
- **Drift logging:** Each `/predict` call appends input to `logs/input_samples.jsonl`.
- **Batch:** `scripts/run_batch.py` — vectorized inference without drift log pollution.
- **CI/CD:** GitHub Actions (lint + test on PR, build Docker on push to main).
- **Public deploy:** DigitalOcean Droplet (Docker + Nginx reverse proxy) at [retentia.vitorsilva.engineer](http://retentia.vitorsilva.engineer). See [DEPLOY.md](DEPLOY.md).
- **Frontend:** UI estática em PT-BR servida pelo Nginx na raiz — gauge animado, sidebar de navegação, fatores de risco por heurística do EDA.
