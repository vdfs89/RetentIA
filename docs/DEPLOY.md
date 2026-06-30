# Arquitetura de Deploy — RetentIA

- **Aplicação:** FastAPI + Uvicorn (`src.main:app`).
- **Infraestrutura:** DigitalOcean Droplet (8GB RAM). A API roda num container Docker (uvicorn na porta interna `8000`, publicada no host como `8080` via `-p 8080:8000`). O **Nginx nativo** (não containerizado) faz reverse proxy. DNS: `retentia.vitorsilva.engineer` → registro A → IP do droplet.
- **Roteamento (Nginx):**
  - `/` → frontend estático servido diretamente pelo Nginx de `/var/www/retentia/` (não passa pela API).
  - `/predict`, `/health`, `/metrics`, `/docs` → `proxy_pass` para o container Docker (`http://127.0.0.1:8080`).
- **Artefatos:** `models/preprocessor.pkl`, `models/mlp_weights.pt`, `models/threshold.pkl`, `models/xgboost.pkl` são **commitados no repositório**. O modelo **não** é treinado no build do Docker — o `Dockerfile` apenas instala as dependências, copia `src/` + `models/` e sobe o uvicorn.
- **`render.yaml`:** permanece no repositório como **legado/inativo** — não há deploy no Render; a infra de produção é a descrita acima.
- **Observabilidade:** Header `X-Process-Time`; Prometheus `/metrics`; log de drift `logs/input_samples.jsonl`.

## Autenticação e Segurança (Decisão de Design)

O endpoint `/predict` está **aberto** nesta fase do projeto, por ser uma demonstração pública acessível via frontend estático. Esta é uma decisão consciente, não uma omissão.

### Como Seria em Produção

Num ambiente produtivo, `/predict` exigiria autenticação. As abordagens adequadas seriam:

- **Chave de API (API key)** no header `X-API-Key`, validada contra um hash armazenado (nunca em texto plano). Implementável com uma dependência do FastAPI:

```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Security(api_key_header)):
    if not verify_hash(key, settings.API_KEY_HASH):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
```

- **Token de portador OAuth2 (OAuth2 bearer token)** (`fastapi.security.OAuth2PasswordBearer`) para integração com um provedor de identidade corporativo.

### Endpoints Públicos por Design

`/health` e `/metrics` permaneceriam **públicos** mesmo em produção:
- `/health` — necessário para health checks de load balancers e orquestradores (probes de liveness/readiness do Kubernetes).
- `/metrics` — necessário para scraping do Prometheus.

### Princípios

- **Credenciais nunca versionadas** — viriam de variáveis de ambiente ou de um gerenciador de segredos (AWS Secrets Manager, Vault), nunca commitadas no repositório.
- **Senhas sempre com hash** — nunca em texto plano, nem em código, nem em documentação.
- **Princípio do menor privilégio** — cada consumidor da API teria sua própria chave, revogável individualmente.
