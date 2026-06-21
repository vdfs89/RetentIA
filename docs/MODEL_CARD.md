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

| Modelo         | Accuracy | Precision | Recall | F1     | ROC-AUC | PR-AUC |
|----------------|----------|-----------|--------|--------|---------|--------|
| Dummy          | 0.7346   | 0.0000    | 0.0000 | 0.0000 | 0.5000  | 0.2654 |
| LogReg @0.5    | 0.7381   | 0.5043    | 0.7807 | 0.6128 | 0.8429  | 0.6340 |
| XGBoost @0.5   | 0.7480   | 0.5167    | 0.7861 | 0.6236 | 0.8420  | 0.6534 |
| MLP @0.37      | 0.6828   | 0.4510    | 0.8984 | 0.6005 | 0.8453  | 0.6372 |

### Reading the results honestly

ROC-AUC is nearly identical across all three non-dummy models (~0.842–0.845). The underlying
discriminative power comes from the features, not the model architecture. Differences emerge
in threshold choice and calibration.

**XGBoost** achieves the best F1 (0.624) and PR-AUC (0.653) at the default 0.5 threshold —
making it the strongest baseline for balanced precision/recall tradeoffs.

**MLP @0.37** achieves the highest recall (89.8%) by operating at a cost-optimized threshold
(C_FN=500, C_FP=100). This is the correct operating point when losing a churner costs 5×
more than a wasted retention offer. The lower accuracy (0.683) is the expected cost of
aggressive churn detection.

**Important caveat:** the comparison between models at 0.5 vs MLP at 0.37 is not fully
apples-to-apples. Applying the same cost-optimized threshold to XGBoost would also push its
recall higher. The key contribution is the **cost-sensitive threshold framework**, not
architectural superiority.

## 5. Cost Analysis FP vs FN

| Type | Scenario | Cost |
|------|----------|------|
| **FN** (false negative) | Customer predicted "stays" but churns → no retention action → customer lost | ≈ CLV (high) |
| **FP** (false positive) | Customer predicted "churns" but stays → wasted retention offer | ≈ campaign cost (low) |

Since C_FN >> C_FP, the optimal threshold falls **below 0.50**, favoring recall over precision.

- **Cost assumptions:** `C_FN = 500`, `C_FP = 100` (placeholders — estimate from real CLV and campaign cost)
- **Threshold derivation:** minimizes total cost on the **validation set** (not test) → `t* = 0.37`
- **Bayes sanity check:** `t* = C_FP / (C_FP + C_FN) = 100 / 600 ≈ 0.167` — empirical threshold higher due to probability calibration, but direction (< 0.5) is consistent

## 6. Limitations & Known Issues

- **MLP does not outperform XGBoost** on F1 or PR-AUC at comparable thresholds — documented honestly, not spun. On small tabular datasets (~7k rows), gradient boosting is competitive.
- **Class imbalance:** handled with `pos_weight` (MLP), `class_weight=balanced` (LogReg), and `scale_pos_weight` (XGBoost).
- **ROC-AUC ceiling:** all three models plateau at ~0.84–0.85, consistent with literature on this dataset. Feature engineering or external data would be required to push further.
- **No per-segment analysis:** recall likely varies across `Contract` types and `InternetService`. Known gap.
- **Threshold sensitivity:** optimal threshold depends on the cost ratio. If business costs change, recalculate.
- **No probability calibration:** raw probabilities are not calibrated. The threshold compensates but true churn likelihoods should be interpreted with caution.

## 7. Governance
- Fixed seed (42), stratified splits, preprocessor `.fit()` only on train set (anti-leakage).
- MLflow experiment tracking: params, metrics, and artifacts logged per run (`sqlite:///mlflow.db`).
- Drift monitoring: each `/predict` call appends input to `logs/input_samples.jsonl`.
