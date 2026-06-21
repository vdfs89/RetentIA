import time

from fastapi import FastAPI, Request
from prometheus_client import Histogram

from src.routes.metrics import router as metrics_router
from src.routes.predict import router as predict_router

LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"],
)

app = FastAPI(title="RetentIA API", description="FastAPI Churn Prediction Service")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    LATENCY.labels(endpoint=request.url.path).observe(duration)
    response.headers["X-Process-Time"] = str(duration)
    return response


app.include_router(metrics_router)
app.include_router(predict_router)
