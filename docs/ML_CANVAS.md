# ML Canvas — RetentIA

## 1. Value Proposition
Reduce recurring revenue loss by anticipating which customers tend to cancel.

## 2. Stakeholders
Retention/CRM team (consumes scores), Finance (CLV/ROI), Data/ML (maintains model).

## 3. Prediction Task
Binary classification: `P(churn)` per customer from 19 profile/contract/billing features.

## 4. Data Sources
IBM Telco Customer Churn dataset as proxy for telco CRM data.

## 5. Features
Contract (`Contract`, `PaymentMethod`), service (add-ons), usage (`tenure`),
billing (`MonthlyCharges`, `TotalCharges`).

## 6. Business Metric
Retained revenue vs campaign cost (ROI). API SLO: p95 latency < 500ms.

## 7. Technical Metric — `(FILL AFTER make train)`
Accuracy, Precision, Recall, F1 + total FP/FN cost at chosen threshold.

## 8. Decision
Score above threshold → customer enters retention queue.
Threshold from cost minimization, not fixed 0.5.

## 9. Serving
FastAPI `/predict` (sync, 1 customer). Batch via `scripts/run_batch.py`.

## 10. Retraining
Periodic retrain with observed churn; monitor feature and score drift.

## 11. Monitoring
Latency, error rate, feature drift, prediction drift. See MONITORING.md.
