# Deploy Architecture — RetentIA

- **App:** FastAPI + Uvicorn (`src.main:app`).
- **Artifacts:** `models/preprocessor.pkl`, `models/mlp_weights.pt`, `models/threshold.pkl`.
- **Release strategy:** `render.yaml` runs `python -m src.train` at build time.
- **Cold start (free tier):** service sleeps after ~15min; first request ~50s.
- **Observability:** `X-Process-Time` header; Prometheus `/metrics`; drift log `logs/input_samples.jsonl`.

## Autenticação e Segurança (decisão de design)

O endpoint `/predict` está **aberto** nesta fase do projeto, por ser uma demonstração pública acessível via frontend estático. Esta é uma decisão consciente, não uma omissão.

### Como seria em produção

Num ambiente produtivo, `/predict` exigiria autenticação. As abordagens adequadas seriam:

- **API key** no header `X-API-Key`, validada contra um hash armazenado (nunca em texto plano). Implementável com uma dependência do FastAPI:

```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Security(api_key_header)):
    if not verify_hash(key, settings.API_KEY_HASH):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
```

- **OAuth2 bearer token** (`fastapi.security.OAuth2PasswordBearer`) para integração com um provedor de identidade corporativo.

### Endpoints públicos por design

`/health` e `/metrics` permaneceriam **públicos** mesmo em produção:
- `/health` — necessário para health checks de load balancers e orquestradores (Kubernetes liveness/readiness probes).
- `/metrics` — necessário para scraping do Prometheus.

### Princípios

- **Credenciais nunca versionadas** — viriam de variáveis de ambiente ou de um secrets manager (AWS Secrets Manager, Vault), nunca commitadas no repositório.
- **Senhas sempre com hash** — nunca em texto plano, nem em código, nem em documentação.
- **Princípio do menor privilégio** — cada consumidor da API teria sua própria key, revogável individualmente.
