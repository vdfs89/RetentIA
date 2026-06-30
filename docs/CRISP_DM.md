# CRISP-DM — RetentIA

## Fase 1 — Entendimento do Negócio

O churn corrói a receita recorrente. Uma empresa de telecomunicações não perde apenas o faturamento do mês corrente, mas todo o Valor do Tempo de Vida do Cliente (CLV) restante quando um cliente cancela. O objetivo do negócio é **priorizar ações de retenção** (ofertas de desconto, chamadas de suporte proativo) para os clientes com maior probabilidade de churn, otimizando o equilíbrio entre o custo de perder um cliente (FN) e o custo de uma oferta de retenção desperdiçada (FP).

## Fase 2 — Entendimento dos Dados

- **Conjunto de dados:** IBM Telco Customer Churn (~7.043 clientes, 19 features, alvo binário `Churn`).
- **Balanço do alvo:** Churn ≈ 26,5% — moderadamente desbalanceado.
- **Ponto de atenção:** A coluna `TotalCharges` contém 11 espaços em branco (não NaN) para clientes com `tenure==0`. Isso é uma **ausência estrutural** (clientes novos, nunca faturados), não aleatória — validado verificando que todos os brancos correspondem a `tenure==0`.
- **Tipos de feature:** 3 numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`), 16 categóricas (contrato, serviço, demográfico).

## Fase 3 — Preparação dos Dados

- Brancos em `TotalCharges` → `pd.to_numeric(errors="coerce").fillna(0.0)` com validação estrutural.
- OneHotEncoder (drop=first, handle_unknown=ignore) em 16 features categóricas.
- StandardScaler em 3 features numéricas.
- Divisão estratificada: Treino 60% / Validação 20% / Teste 20%.
- **Anti-vazamento:** Preprocessor `.fit()` apenas no treino; `.transform()` na validação e teste.

## Fase 4 — Modelagem

Quatro modelos treinados e comparados:

1.  **DummyClassifier** (most_frequent): Linha de base — sempre prevê a classe majoritária.
2.  **LogisticRegression** (class_weight=balanced, max_iter=1000): Forte linha de base linear.
3.  **XGBClassifier** (n_estimators=200, max_depth=4, lr=0.05, scale_pos_weight=2.73): Linha de base com gradient boosting.
4.  **MLP** (PyTorch): [32→16→1], BatchNorm + ReLU + Dropout(0.2), BCEWithLogitsLoss com pos_weight=2.73, Adam lr=0.005, batch_size=64, parada antecipada (paciência=10, máx 100 épocas).

## Fase 5 — Avaliação

| Modelo         | Acurácia | Precisão  | Recall | F1     | ROC-AUC | PR-AUC |
|----------------|----------|-----------|--------|--------|---------|--------|
| Dummy          | 0.7346   | 0.0000    | 0.0000 | 0.0000 | 0.5000  | 0.2654 |
| LogReg @0.5    | 0.7381   | 0.5043    | 0.7807 | 0.6128 | 0.8429  | 0.6340 |
| XGBoost @0.5   | 0.7480   | 0.5167    | 0.7861 | 0.6236 | 0.8420  | 0.6534 |
| MLP @0.37      | 0.6828   | 0.4510    | 0.8984 | 0.6005 | 0.8453  | 0.6372 |

O ROC-AUC atinge um platô em ~0.842–0.845 em todos os modelos não-dummy. O XGBoost lidera em F1/PR-AUC no limiar padrão. O MLP @0.37 lidera em recall através de um limiar sensível ao custo (C_FN=500, C_FP=100).
Conclusão honesta: a arquitetura não impulsiona o desempenho neste conjunto de dados — as features sim.

## Fase 6 — Implantação

- **API:** FastAPI com `/predict`, `/health`, `/metrics` (Prometheus).
- **Camada de serviço:** `model_service.py` carrega o pré-processador + MLP + limiar na inicialização.
- **Log de drift:** Cada chamada a `/predict` anexa a entrada em `logs/input_samples.jsonl`.
- **Batch:** `scripts/run_batch.py` — inferência vetorizada sem poluir o log de drift.
- **CI/CD:** GitHub Actions (lint + teste em PR, build do Docker em push para main).
- **Deploy público:** DigitalOcean Droplet (Docker + Nginx reverse proxy) em [retentia.vitorsilva.engineer](http://retentia.vitorsilva.engineer). Veja [DEPLOY.md](DEPLOY.md).
- **Frontend:** UI estática em PT-BR servida pelo Nginx na raiz — gauge animado, sidebar de navegação, fatores de risco por heurística do EDA.
