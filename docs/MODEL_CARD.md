# MODEL CARD — RetentIA Churn Classifier

> All metrics below are `(FILL AFTER make train)`. Never hardcode — report YOUR run numbers.

## 1. Model Details
- **Task:** Binary churn classification
- **Architecture:** MLP (PyTorch) — [32, 16], BatchNorm + ReLU + Dropout, logit output
- **Baselines:** DummyClassifier (most_frequent), LogisticRegression (class_weight=balanced)
- **Version:** 0.1.0
- **Dataset:** IBM Telco Customer Churn (~7,043 customers, 19 features)

## 2. Intended Use
- **In-scope:** Prioritize customers for proactive retention campaigns.
- **Out-of-scope:** Automated decisions without human review; markets outside original dataset.

## 3. Data & Features
- 3 numerical (scaled): `tenure`, `MonthlyCharges`, `TotalCharges`
- 16 categorical (OneHot, drop=first): `gender`, `SeniorCitizen`, `Contract`, `PaymentMethod`, ...
- **TotalCharges gotcha:** 11 rows with blank = customers with `tenure==0` (structural, not random) → imputed 0.0
- **Target balance:** churn ≈ 26.5% `(CONFIRM in EDA)` — imbalanced

## 4. Performance (Test Set) — `(FILL AFTER make train)`
| Model          | Accuracy | Precision | Recall | F1   |
|----------------|----------|-----------|--------|------|
| Dummy          |    —     |     —     |   —    |  —   |
| LogReg         |    —     |     —     |   —    |  —   |
| MLP @0.5       |    —     |     —     |   —    |  —   |
| MLP @cost_thr  |    —     |     —     |   —    |  —   |

## 5. Cost Analysis FP vs FN
- **FN** (missed churn): customer lost ≈ CLV — expensive.
- **FP** (false alarm): wasted retention offer — cheap.
- Threshold **derived** from cost ratio, not fixed at 0.5.
- **Cost assumptions:** `C_FN = (ESTIMATE)`, `C_FP = (ESTIMATE)` — document source.
- **Chosen threshold:** `(FILL)` | **Total cost on test:** `(FILL)`

## 6. Limitations & Bias
- Class imbalance handled with `pos_weight` (MLP) and `class_weight` (LogReg).
- MLP rarely beats strong baselines on small tabular — report honestly.

## 7. Governance
- Fixed seed (42), stratified splits, preprocessor fit only on train (anti-leakage).
- MLflow tracking (params, metrics, artifacts).
