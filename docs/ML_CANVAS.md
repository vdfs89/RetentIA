# ML Canvas — RetentIA

## 1. Value Proposition
Reduce recurring revenue loss by anticipating which customers tend to cancel, enabling proactive retention before churn occurs.

## 2. Stakeholders
- **Retention/CRM team:** Consumes churn scores to prioritize outreach.
- **Finance:** Validates CLV assumptions and campaign ROI.
- **Data/ML team:** Maintains and retrains the model.

## 3. Prediction Task
Binary classification: `P(churn)` per customer, from 19 profile, contract, and billing features. Output: probability + binary decision at cost-optimized threshold.

## 4. Data Sources
IBM Telco Customer Churn dataset (~7,043 customers) as proxy for real telco CRM data. Auto-downloaded from GitHub mirror on first run.

## 5. Features
- **Contract:** `Contract` (month-to-month, one year, two year), `PaymentMethod`, `PaperlessBilling`
- **Services:** `InternetService`, `PhoneService`, add-ons (OnlineSecurity, TechSupport, Streaming, etc.)
- **Usage/tenure:** `tenure` (months as customer)
- **Billing:** `MonthlyCharges`, `TotalCharges`
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`

## 6. Business Metric
Retained revenue vs. campaign cost (ROI of retention actions). API latency SLO: p95 < 500ms.

## 7. Technical Metrics (from test set)
- **MLP @0.37 threshold:** Accuracy 0.6828, Precision 0.4510, Recall 0.8984, F1 0.6005
- **LogReg @0.5 baseline:** Accuracy 0.7381, Precision 0.5043, Recall 0.7807, F1 0.6128
- **Cost assumptions:** C_FN=500 (lost customer ≈ CLV), C_FP=100 (wasted retention offer)

## 8. Decision
Score above threshold (0.37) → customer enters retention queue for human review (offer, call, discount). Threshold derived from cost minimization on validation set, not fixed at 0.5.

## 9. Serving
- **Online:** FastAPI `/predict` (sync, 1 customer per request).
- **Batch:** `scripts/run_batch.py` (vectorized, full dataset).

## 10. Retraining
Periodic retrain when observed churn data becomes available. Triggers: significant feature drift, recall degradation, or changed cost assumptions.

## 11. Monitoring
- **Service:** `/health`, `/metrics` (Prometheus), `X-Process-Time` header.
- **Drift:** `logs/input_samples.jsonl` for input distribution monitoring (Evidently/NannyML).
- **Performance:** Recalculate metrics when ground truth (actual churn) is available.
