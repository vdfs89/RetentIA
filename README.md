# RetentIA

[![CI](https://github.com/vdfs89/RetentIA/actions/workflows/ci.yml/badge.svg)](https://github.com/vdfs89/RetentIA/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org)

> Predict customer churn before it happens.

Projeto de predição de churn com MLP (PyTorch), FastAPI, MLflow e práticas de MLOps.
**FIAP Pós-Tech MLET — Tech Challenge.**

🔗 **API pública:** [retentia.vitorsilva.engineer](http://retentia.vitorsilva.engineer/docs)

---

## Resultados

| Modelo        | Accuracy | Precision | Recall   | F1     |
|---------------|----------|-----------|----------|--------|
| Dummy         | 0.7346   | 0.0000    | 0.0000   | 0.0000 |
| LogReg @0.5   | 0.7381   | 0.5043    | 0.7807   | 0.6128 |
| **MLP @0.36** | 0.6558   | 0.4295    | **0.9037** | 0.5823 |

**Threshold otimizado por custo: 0.36** — tunado no conjunto de validação (C_FN=500, C_FP=100).

O MLP captura **90% dos churners reais**, trocando accuracy por recall. Quando perder um cliente custa 5× mais que uma oferta de retenção desperdiçada, esse é o tradeoff correto. O MLP não supera a LogReg em F1 — e isso é documentado honestamente. A contribuição é o **threshold derivado por custo**, não a arquitetura. Ver [MODEL_CARD](docs/MODEL_CARD.md).

---

## Quickstart

```bash
# Instalar dependências
make install                  # ou: uv pip install --system -r requirements.txt

# Treinar (baixa o dataset automaticamente no primeiro run)
make train                    # baselines + MLP + threshold por custo → models/

# Testes (isolados — não tocam os modelos treinados)
make test                     # 8 testes: health, metrics, predict, model, schema

# Subir a API local
make run                      # http://localhost:8000/docs

# Inferência em batch
make run-batch                # → data/processed/batch_output.jsonl

# Lint
make lint                     # ruff check + format check
```

---

## Arquitetura

Segue a convenção do projeto de referência [`swe4ds-credit-api`](https://github.com/Cataldir/Materiais-MLET) (Pós-Tech MLET, Fase 01 — Eng. de Software para Cientistas de Dados).

```
src/
  main.py                     # FastAPI thin (routers + middleware de latência)
  routes/
    metrics.py                # /health + /metrics (Prometheus)
    predict.py                # /predict → services
  services/
    model_service.py          # load_model, predict_one, drift log
  api/
    schemas.py                # Pydantic v2 (19 features, Literal types)
  data/
    ingest.py                 # Ingestão + fix do gotcha TotalCharges
  features/
    columns.py                # Fonte única de verdade (features + domínios)
    preprocessor.py           # ColumnTransformer (OHE + StandardScaler)
  models/
    mlp.py                    # MLP [32→16→1], BatchNorm, Dropout
  validation/
    schemas.py                # Pandera (validação de DataFrame)
  cost/
    threshold.py              # Otimização de threshold sensível a custo
  train.py                    # Pipeline completo (baselines + MLP + MLflow)
scripts/
  run_batch.py                # Inferência vetorizada em batch
tests/                        # 5 arquivos de teste (fixtures isoladas com tmp_path)
docs/                         # MODEL_CARD, CRISP_DM, ML_CANVAS, DEPLOY, MONITORING
notebooks/                    # EDA exploratória
```

---

## Dataset

**IBM Telco Customer Churn** (~7.043 clientes, 19 features).

Auto-download na primeira execução do `make train` (mirror: [treselle-systems/customer_churn_analysis](https://github.com/treselle-systems/customer_churn_analysis)).

### Gotcha: TotalCharges

11 linhas contêm espaço em branco em vez de número — são **exatamente** os clientes com `tenure == 0` (recém-assinados, nunca cobrados). Missing **estrutural**, não aleatório. Imputado como `0.0` com validação de que os blanks coincidem com `tenure == 0`.

---

## Análise Exploratória (EDA)

Notebook completo em [notebooks/eda.ipynb](notebooks/eda.ipynb). Principais achados:

| | |
|---|---|
| ![Churn Distribution](docs/churn_distribution.png) | ![Numerical Distributions](docs/numerical_distributions.png) |
| ![Categorical Churn Rates](docs/categorical_churn_rates.png) | ![Correlation Matrix](docs/correlation_matrix.png) |
| ![Tenure vs Churn](docs/tenure_vs_churn.png) | ![Charges Scatter](docs/charges_scatter.png) |

![All Categorical Churn Rates](docs/all_categorical_churn_rates.png)

---

## API

| Endpoint   | Método | Descrição                             |
|-----------|--------|---------------------------------------|
| `/health` | GET    | Liveness check                        |
| `/metrics`| GET    | Prometheus (counters + latência)      |
| `/predict`| POST   | Predição de churn (1 cliente, JSON)   |
| `/docs`   | GET    | Swagger UI interativo                 |

### Exemplo de request

```bash
curl -X POST http://retentia.vitorsilva.engineer/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35, "TotalCharges": 845.5
  }'
```

### Exemplo de response

```json
{
  "churn_probability": 0.8234,
  "churn_prediction": true,
  "threshold": 0.36
}
```

---

## Análise de Custo FP vs FN

| Tipo | Cenário | Custo |
|------|---------|-------|
| **FN** (falso negativo) | Cliente previsto "fica" mas cancela → sem ação de retenção → cliente perdido | ≈ CLV (alto) |
| **FP** (falso positivo) | Cliente previsto "cancela" mas fica → oferta de retenção desperdiçada | ≈ custo da campanha (baixo) |

Como C_FN >> C_FP, o threshold ótimo cai **abaixo de 0.50** (favorece recall).
Threshold: **0.36** — derivado por minimização de custo no conjunto de validação, não arbitrado.

---

## Deploy

- **Infraestrutura:** DigitalOcean Droplet (8GB) + Docker + Nginx reverse proxy
- **Frontend:** [retentia.vitorsilva.engineer](http://retentia.vitorsilva.engineer) — UI estática servida pelo Nginx
- **API / Swagger:** [retentia.vitorsilva.engineer/docs](http://retentia.vitorsilva.engineer/docs)
- **CI/CD:** GitHub Actions (lint + testes em PR; lint + format + Docker build em push)
- **Observabilidade:** Header `X-Process-Time`, Prometheus `/metrics`, drift log `logs/input_samples.jsonl`

### Arquitetura de roteamento (Nginx)

```
retentia.vitorsilva.engineer/           → HTML estático (retentia-ui/index-static.html)
retentia.vitorsilva.engineer/predict    → FastAPI (proxy → Docker :8000)
retentia.vitorsilva.engineer/health     → FastAPI
retentia.vitorsilva.engineer/metrics    → FastAPI (Prometheus)
retentia.vitorsilva.engineer/docs       → FastAPI Swagger UI
```

---

## Documentação

| Documento | Descrição |
|-----------|-----------|
| [MODEL_CARD](docs/MODEL_CARD.md) | Performance, análise de custo, limitações |
| [CRISP_DM](docs/CRISP_DM.md) | Fases da metodologia CRISP-DM |
| [ML_CANVAS](docs/ML_CANVAS.md) | Framing de negócio |
| [DEPLOY](docs/DEPLOY.md) | Arquitetura de deploy |
| [MONITORING](docs/MONITORING.md) | Monitoramento de drift e serviço |

---

## Stack

| Componente | Tecnologia |
|-----------|-----------|
| Modelo | PyTorch (MLP) |
| API | FastAPI + Uvicorn |
| Tracking | MLflow |
| Validação (API) | Pydantic v2 |
| Validação (DataFrame) | Pandera |
| Métricas | Prometheus-client |
| Preprocessing | scikit-learn (ColumnTransformer) |
| Lint | Ruff |
| Testes | Pytest (fixtures isoladas) |
| CI/CD | GitHub Actions |
| Deploy | Docker + Nginx + DigitalOcean |

---

## Licença

MIT
