# ML Canvas — RetentIA

## 1. Proposta de Valor
Reduzir a perda de receita recorrente antecipando quais clientes tendem a cancelar, permitindo a retenção proativa antes que o churn ocorra.

## 2. Partes Interessadas (Stakeholders)
- **Equipe de Retenção/CRM:** Consome as pontuações de churn para priorizar o contato.
- **Financeiro:** Valida as premissas de CLV e o ROI da campanha.
- **Equipe de Dados/ML:** Mantém e retreina o modelo.

## 3. Tarefa de Predição
Classificação binária: `P(churn)` por cliente, a partir de 19 features de perfil, contrato e faturamento. Saída: probabilidade + decisão binária em um limiar otimizado por custo.

## 4. Fontes de Dados
Conjunto de dados IBM Telco Customer Churn (~7.043 clientes) como um proxy para dados reais de CRM de telecomunicações. Baixado automaticamente de um espelho do GitHub na primeira execução.

## 5. Features
- **Contrato:** `Contract` (mês a mês, um ano, dois anos), `PaymentMethod`, `PaperlessBilling`
- **Serviços:** `InternetService`, `PhoneService`, adicionais (OnlineSecurity, TechSupport, Streaming, etc.)
- **Uso/Tempo de Contrato:** `tenure` (meses como cliente)
- **Faturamento:** `MonthlyCharges`, `TotalCharges`
- **Demografia:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`

## 6. Métrica de Negócio
Receita retida vs. custo da campanha (ROI das ações de retenção). SLO de latência da API: p95 < 500ms.

## 7. Métricas Técnicas (do conjunto de teste)
- **Limiar MLP @0.37:** Acurácia 0.6828, Precisão 0.4510, Recall 0.8984, F1 0.6005
- **Linha de base LogReg @0.5:** Acurácia 0.7381, Precisão 0.5043, Recall 0.7807, F1 0.6128
- **Premissas de custo:** C_FN=500 (cliente perdido ≈ CLV), C_FP=100 (oferta de retenção desperdiçada)

## 8. Decisão
Pontuação acima do limiar (0.37) → cliente entra na fila de retenção para revisão humana (oferta, ligação, desconto). Limiar derivado da minimização de custos no conjunto de validação, não fixo em 0.5.

## 9. Servindo o Modelo (Serving)
- **Online:** FastAPI `/predict` (síncrono, 1 cliente por requisição).
- **Em Lote (Batch):** `scripts/run_batch.py` (vetorizado, conjunto de dados completo).

## 10. Retreinamento
Retreinamento periódico quando dados de churn observados se tornam disponíveis. Gatilhos: desvio significativo das features, degradação do recall ou mudança nas premissas de custo.

## 11. Monitoramento
- **Serviço:** `/health`, `/metrics` (Prometheus), header `X-Process-Time`.
- **Drift:** `logs/input_samples.jsonl` para monitoramento da distribuição de entrada (Evidently/NannyML).
- **Desempenho:** Recalcular métricas quando a verdade fundamental (churn real) estiver disponível.
