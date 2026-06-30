import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Histogram

from src.routes.metrics import router as metrics_router
from src.routes.predict import router as predict_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("retentia")

LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latência das requisições em segundos",
    ["endpoint"],
)

app = FastAPI(title="RetentIA API", description="Serviço de Predição de Churn com FastAPI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    LATENCY.labels(endpoint=request.url.path).observe(duration)
    response.headers["X-Process-Time"] = str(duration)
    logger.info(
        "%s %s -> %d (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


app.include_router(metrics_router)
app.include_router(predict_router)
