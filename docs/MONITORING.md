# Monitoring Playbook — RetentIA

## Service Health
- `/health` → liveness check.
- `/metrics` → Prometheus counters and latency histogram.
- Latency header `X-Process-Time` per request.

## Model Quality
- **Data drift:** monitor distribution of `tenure`, `MonthlyCharges`, `Contract`, `PaymentMethod` vs training baseline.
- **Prediction drift:** score distribution and positive rate at threshold.
- **Performance:** when actual churn arrives, recalculate recall/precision/cost.

## Retraining Triggers
- Significant drift, recall drop, or change in cost assumptions (CLV/offer cost).

## Drift Sampling
- Each `/predict` appends to `logs/input_samples.jsonl` (input for Evidently/NannyML).
- Batch inference (`scripts/run_batch.py`) does NOT log to drift file (avoids pollution).

## Evolução: Observabilidade Avançada

O monitoramento atual usa Prometheus (endpoint `/metrics`) para métricas de requisição e latência, mais logging estruturado de inputs (`logs/input_samples.jsonl`) para análise de drift. Em um ambiente de maior escala, a evolução natural seria instrumentar a aplicação via OpenTelemetry e visualizar tudo no stack LGTM:

- **Loki** — agregação e consulta de logs
- **Grafana** — dashboards unificados e alertas visuais
- **Tempo** — tracing distribuído de requisições
- **Prometheus** — métricas (já implementado neste projeto)

O OpenTelemetry Collector atuaria como camada de coleta padronizada, recebendo a telemetria da aplicação e roteando cada tipo para o backend correspondente (métricas → Prometheus, logs → Loki, traces → Tempo), com o Grafana como hub central de visualização.
