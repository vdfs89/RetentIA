# MODEL CARD — Classificador de Churn RetentIA

## 1. Detalhes do Modelo
- **Tarefa:** Classificação binária de churn
- **Arquitetura:** MLP (PyTorch) — [32, 16], BatchNorm + ReLU + Dropout(0.2), saída de logit
- **Linhas de Base (Baselines):** DummyClassifier (most_frequent), LogisticRegression (class_weight=balanced, max_iter=1000)
- **Versão:** 0.1.0
- **Conjunto de Dados:** IBM Telco Customer Churn (~7.043 clientes, 19 features)

## 2. Uso Pretendido
- **No escopo:** Priorizar clientes para campanhas de retenção proativas.
- **Fora do escopo:** Decisões automatizadas sem revisão humana; mercados fora do conjunto de dados original.

## 3. Dados & Features
- 3 numéricas (StandardScaler): `tenure`, `MonthlyCharges`, `TotalCharges`
- 16 categóricas (OneHotEncoder, drop=first): `gender`, `SeniorCitizen`, `Contract`, `PaymentMethod`, `InternetService`, e 11 outras
- **Ponto de atenção em TotalCharges:** 11 linhas com valor em branco = clientes com `tenure==0` (ausência estrutural, não aleatória) → imputado 0.0
- **Balanço do alvo:** churn ≈ 26,5% — desbalanceado. Tratado com `pos_weight=2.73` (MLP) e `class_weight=balanced` (LogReg).
- **Divisão:** Treino 60% / Validação 20% / Teste 20% (estratificado, seed=42)

## 4. Desempenho (Conjunto de Teste)

| Modelo         | Acurácia | Precisão | Recall | F1     | ROC-AUC | PR-AUC |
|----------------|----------|-----------|--------|--------|---------|--------|
| Dummy          | 0.7346   | 0.0000    | 0.0000 | 0.0000 | 0.5000  | 0.2654 |
| LogReg @0.5    | 0.7381   | 0.5043    | 0.7807 | 0.6128 | 0.8429  | 0.6340 |
| XGBoost @0.5   | 0.7480   | 0.5167    | 0.7861 | 0.6236 | 0.8420  | 0.6534 |
| MLP @0.37      | 0.6828   | 0.4510    | 0.8984 | 0.6005 | 0.8453  | 0.6372 |

### Lendo os resultados honestamente

O ROC-AUC é quase idêntico em todos os três modelos não-dummy (~0.842–0.845). O poder
discriminativo subjacente vem das features, não da arquitetura do modelo. As diferenças surgem
na escolha e calibração do limiar.

**XGBoost** alcança o melhor F1 (0.624) e PR-AUC (0.653) no limiar padrão de 0.5 —
tornando-se a linha de base mais forte para um equilíbrio entre precisão/recall.

**MLP @0.37** alcança o maior recall (89,8%) ao operar em um limiar otimizado por custo
(C_FN=500, C_FP=100). Este é o ponto de operação correto quando perder um cliente que faz churn custa 5x
mais do que uma oferta de retenção desperdiçada. A acurácia mais baixa (0.683) é o custo esperado
de uma detecção agressiva de churn.

**Ressalva importante:** a comparação entre os modelos em 0.5 vs MLP em 0.37 não é totalmente
justa. Aplicar o mesmo limiar otimizado por custo ao XGBoost também aumentaria seu
recall. A contribuição principal é a **estrutura de limiar sensível ao custo**, não a
superioridade da arquitetura.

## 5. Importância das Features

Ranking das features mais influentes segundo o ganho (*gain*) do XGBoost. Como os três modelos têm ROC-AUC quase idêntico (~0.842–0.845), este ranking aproxima quais sinais dirigem a predição em todos eles. Gerado por [`scripts/feature_importance.py`](../scripts/feature_importance.py); ranking completo em [`docs/feature_importance.csv`](feature_importance.csv).

| #  | Feature                          | Importância |
|----|----------------------------------|-----------|
| 1  | `Contract=Two year`              | 0.3719    |
| 2  | `Contract=One year`              | 0.1694    |
| 3  | `InternetService=Fiber optic`    | 0.1500    |
| 4  | `InternetService=No`             | 0.0597    |
| 5  | `StreamingMovies=Yes`            | 0.0374    |
| 6  | `tenure`                         | 0.0235    |
| 7  | `PaymentMethod=Electronic check` | 0.0206    |
| 8  | `OnlineSecurity=Yes`             | 0.0202    |
| 9  | `StreamingTV=Yes`                | 0.0187    |
| 10 | `PhoneService=Yes`               | 0.0130    |
| 11 | `OnlineBackup=Yes`               | 0.0119    |
| 12 | `PaperlessBilling=Yes`           | 0.0117    |
| 13 | `TechSupport=Yes`                | 0.0114    |
| 14 | `MultipleLines=Yes`              | 0.0106    |
| 15 | `TotalCharges`                   | 0.0100    |

> Nota: o OneHotEncoder usa `drop="first"`, então a categoria de referência de cada feature não aparece (ex.: `Contract=Month-to-month`, `PaymentMethod=Bank transfer`). A importância dos níveis listados é medida **em relação** a essas referências.

**Leitura de negócio:**
- O **tipo de contrato domina de forma esmagadora**: `Contract=Two year` + `Contract=One year` somam ~54% do ganho total. Isso captura o efeito do contrato month-to-month (a referência omitida), que no EDA tem churn ~42% — clientes em contratos longos cancelam muito menos.
- **`InternetService` é o segundo sinal mais forte** (Fibra Óptica + "sem internet" ≈ 21% do ganho), consistente com a maior taxa de churn em fibra óptica observada no EDA.
- **`tenure` e `PaymentMethod=Electronic check` aparecem como sinais secundários** (não no topo, mas relevantes) — alinhados ao padrão de que tenure baixo e cheque eletrônico concentram cancelamentos.
- Features de serviço (`OnlineSecurity`, `TechSupport`, `StreamingTV`) entram na cauda do ranking: secundárias, porém presentes — clientes sem esses add-ons cancelam mais.
- Isso confirma a tese do projeto: **o sinal vem das features de relacionamento e contrato, não da arquitetura do modelo** — coerente com o ROC-AUC quase idêntico entre MLP, XGBoost e LogReg.

**Implicação para retenção:** priorizar clientes em contrato month-to-month, com internet de fibra óptica, tenure baixo e pagamento via electronic check — exatamente o perfil que o frontend sinaliza como alto risco.

## 6. Análise de Custo FP vs FN

| Tipo | Cenário | Custo |
|------|----------|------|
| **FN** (falso negativo) | Cliente previsto como "permanece" mas faz churn → sem ação de retenção → cliente perdido | ≈ CLV (alto) |
| **FP** (falso positivo) | Cliente previsto como "churn" mas permanece → oferta de retenção desperdiçada | ≈ custo da campanha (baixo) |

Como C_FN >> C_FP, o limiar ótimo cai **abaixo de 0.50**, favorecendo o recall em detrimento da precisão.

- **Premissas de custo:** `C_FN = 500`, `C_FP = 100` (valores de exemplo — estimar a partir do CLV real and e do custo da campanha)
- **Derivação do limiar:** minimiza o custo total no **conjunto de validação** (não no de teste) → `t* = 0.37`
- **Verificação de sanidade de Bayes:** `t* = C_FP / (C_FP + C_FN) = 100 / 600 ≈ 0.167` — o limiar empírico é mais alto devido à calibração de probabilidade, mas a direção (< 0.5) é consistente

## 7. Limitações & Problemas Conhecidos

- **MLP não supera o XGBoost** em F1 ou PR-AUC em limiares comparáveis — documentado honestamente, não disfarçado. Em conjuntos de dados tabulares pequenos (~7k linhas), o gradient boosting é competitivo.
- **Desbalanceamento de classe:** tratado com `pos_weight` (MLP), `class_weight=balanced` (LogReg), e `scale_pos_weight` (XGBoost).
- **Teto do ROC-AUC:** todos os três modelos atingem um platô em ~0.84–0.85, consistente com a literatura sobre este conjunto de dados. Engenharia de features ou dados externos seriam necessários para ir além.
- **Sem análise por segmento:** o recall provavelmente varia entre os tipos de `Contract` e `InternetService`. Uma lacuna conhecida.
- **Sensibilidade do limiar:** o limiar ótimo depende da razão de custos. Se os custos do negócio mudarem, recalcule.
- **Sem calibração de probabilidade:** as probabilidades brutas não são calibradas. O limiar compensa, mas as verdadeiras probabilidades de churn devem ser interpretadas com cautela.

## 8. Governança
- Seed fixo (42) no numpy, torch e no gerador do DataLoader, divisões estratificadas, `preprocessor.fit()` apenas no conjunto de treino (anti-vazamento).
- Rastreamento de experimentos com MLflow: parâmetros, métricas e artefatos registrados por execução (`sqlite:///mlflow.db`).
- Monitoramento de drift: cada chamada `/predict` anexa a entrada em `logs/input_samples.jsonl`.

> **Nota de reprodutibilidade:** os artefatos versionados do MLP (`models/mlp_weights.pt`, `models/threshold.pkl = 0.37`) e as métricas reportadas foram produzidos **antes** da adição do seeding no PyTorch. Eles são mantidos como estão para que os números documentados correspondam ao modelo em deploy. O seed garante resultados determinísticos para execuções de retreinamento **futuras**; reproduzir os pesos atuais exatos exigiria a execução original (sem seed).

## 9. Justiça & Viés (Fairness & Bias)

O conjunto de dados inclui **atributos protegidos/sensíveis** usados como features do modelo: `gender`, `SeniorCitizen`, `Partner`, `Dependents`. Isso levanta considerações de justiça que devem ser abordadas antes de qualquer implantação no mundo real.

**Lacuna conhecida (ainda não auditada):** nenhuma análise de justiça de subgrupo foi realizada. As métricas relatadas são apenas agregadas — **não há** medição de recall/precisão por subgrupo demográfico (ex: recall para idosos vs. não idosos, ou por gênero), nem qualquer critério de justiça (probabilidades equalizadas, paridade demográfica) validado.

**Evidência mitigadora (parcial, não substitui uma auditoria):** o ranking de [importância das features](#5-importância-das-features) mostra que o sinal é dominado por `Contract`, `InternetService` e `tenure` — features de relacionamento/contrato — enquanto os atributos demográficos **não** aparecem nos primeiros postos. Isso sugere que o modelo não se apoia fortemente em atributos protegidos, mas não prova a ausência de impacto díspar.

**Próximos passos recomendados para produção:**
- Medir o desempenho por subgrupo (recall, precisão, custo) em `gender` e `SeniorCitizen`.
- Avaliar a remoção de atributos protegidos do conjunto de features, ou aplicar restrições de justiça / pós-processamento.
- Verificar se a fila de retenção não visa sistematicamente de forma excessiva ou insuficiente nenhum grupo protegido.

**Nota ética/regulatória:** usar `gender` ou idade (`SeniorCitizen`) para decidir quem recebe ofertas de retenção pode ter implicações legais e éticas dependendo da jurisdição. Em um ambiente de produção, isso deve ser revisado com a equipe de conformidade antes do lançamento.
