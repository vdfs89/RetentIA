# MODEL CARD — RetentIA Churn Classifier

## 1. Model Details
- **Task:** Binary churn classification
- **Architecture:** MLP (PyTorch) — [32, 16], BatchNorm + ReLU + Dropout(0.2), logit output
- **Baselines:** DummyClassifier (most_frequent), LogisticRegression (class_weight=balanced, max_iter=1000)
- **Version:** 0.1.0
- **Dataset:** IBM Telco Customer Churn (~7,043 customers, 19 features)

## 2. Intended Use
- **In-scope:** Prioritize customers for proactive retention campaigns.
- **Out-of-scope:** Automated decisions without human review; markets outside original dataset.

## 3. Data & Features
- 3 numerical (StandardScaler): `tenure`, `MonthlyCharges`, `TotalCharges`
- 16 categorical (OneHotEncoder, drop=first): `gender`, `SeniorCitizen`, `Contract`, `PaymentMethod`, `InternetService`, and 11 others
- **TotalCharges gotcha:** 11 rows with blank = customers with `tenure==0` (structural missingness, not random) → imputed 0.0
- **Target balance:** churn ≈ 26.5% — imbalanced. Handled with `pos_weight=2.73` (MLP) and `class_weight=balanced` (LogReg).
- **Split:** Train 60% / Validation 20% / Test 20% (stratified, seed=42)

## 4. Performance (Test Set)

| Model         | Accuracy | Precision | Recall | F1     |
|---------------|----------|-----------|--------|--------|
| Dummy         | 0.7346   | 0.0000    | 0.0000 | 0.0000 |
| LogReg @0.5   | 0.7381   | 0.5043    | 0.7807 | 0.6128 |
| MLP @0.36     | 0.6558   | 0.4295    | 0.9037 | 0.5823 |

**MLP training details:** early stopping at epoch 19 (patience=10), Adam lr=0.005, batch_size=64, BCEWithLogitsLoss with pos_weight=2.73.

## 5. Cost Analysis FP vs FN

| Type | Scenario | Cost |
|------|----------|------|
| **FN** (missed churn) | Customer predicted "stays" but cancels → no retention action → customer lost | ≈ CLV (~$500) |
| **FP** (false alarm) | Customer predicted "churns" but stays → wasted retention offer | ≈ campaign cost (~$100) |

- **Cost assumptions:** C_FN = 500 (lost customer ≈ CLV), C_FP = 100 (wasted retention offer)
- **Optimal threshold:** 0.36 — derived by minimizing `FN × C_FN + FP × C_FP` over 101 thresholds on the validation set, not arbitrarily fixed at 0.5.
- As C_FN >> C_FP, the optimal threshold falls below 0.5, favoring recall over precision.

## 6. Honest Assessment & Limitations

The MLP does **not** outperform LogReg on F1 (0.5823 vs 0.6128) or accuracy (0.6558 vs 0.7381). The comparison is not fully apples-to-apples — LogReg is evaluated at 0.5 while MLP at the cost-optimized 0.36.

The primary contribution is the **cost-sensitive threshold derivation**, not the architecture. At threshold=0.36, the MLP captures **90.4% of actual churners** (recall=0.9037), which is the correct business objective when missing a churner costs 5× more than a false alarm.

Known limitations:
- Class imbalance handled statistically but not with augmentation or resampling.
- Small dataset (~7K rows) — results may not generalize to larger telco datasets.
- No feature selection — all 19 features used regardless of individual predictive power.

## 7. Governance
- Fixed seed (42), stratified splits, preprocessor `.fit()` only on train set (anti-leakage).
- MLflow experiment tracking: params, metrics, and artifacts logged per run (`sqlite:///mlflow.db`).
- Drift monitoring: each `/predict` call appends input to `logs/input_samples.jsonl`.
