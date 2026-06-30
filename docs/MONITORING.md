# Manual de Monitoramento — RetentIA

## Saúde do Serviço
- `/health` → verificação de atividade (liveness check).
- `/metrics` → contadores e histograma de latência para o Prometheus.
- Header de latência `X-Process-Time` por requisição.

## Qualidade do Modelo
- **Drift de dados:** monitorar a distribuição de `tenure`, `MonthlyCharges`, `Contract`, `PaymentMethod` vs. a linha de base do treinamento.
- **Drift de predição:** distribuição das pontuações e taxa de positivos no limiar.
- **Desempenho:** quando os dados reais de churn chegarem, recalcular recall/precisão/custo.

## Gatilhos para Retreinamento
- Drift significativo, queda no recall ou mudança nas premissas de custo (CLV/custo da oferta).

## Amostragem para Drift
- Cada chamada a `/predict` anexa dados a `logs/input_samples.jsonl` (entrada para Evidently/NannyML).
- A inferência em lote (`scripts/run_batch.py`) NÃO registra no arquivo de drift (evita poluição).

## Evolução: Observabilidade Avançada

O monitoramento atual usa Prometheus (endpoint `/metrics`) para métricas de requisição e latência, mais logging estruturado de inputs (`logs/input_samples.jsonl`) para análise de drift. Em um ambiente de maior escala, a evolução natural seria instrumentar a aplicação via OpenTelemetry e visualizar tudo no stack LGTM:

- **Loki** — agregação e consulta de logs
- **Grafana** — dashboards unificados e alertas visuais
- **Tempo** — tracing distribuído de requisições
- **Prometheus** — métricas (já implementado neste projeto)

O OpenTelemetry Collector atuaria como camada de coleta padronizada, recebendo a telemetria da aplicação e roteando cada tipo para o backend correspondente (métricas → Prometheus, logs → Loki, traces → Tempo), com o Grafana como hub central de visualização.
