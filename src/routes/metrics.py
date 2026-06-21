from fastapi import APIRouter, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

router = APIRouter()

REQUESTS_COUNTER = Counter("api_requests_total", "Total requests received", ["method", "endpoint"])


@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health")
def health():
    REQUESTS_COUNTER.labels(method="GET", endpoint="/health").inc()
    return {"status": "healthy"}
