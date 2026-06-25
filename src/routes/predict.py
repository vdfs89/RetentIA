import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CustomerFeatures,
    PredictionResponse,
)
from src.services.model_service import predict_batch, predict_one

logger = logging.getLogger("retentia")

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    try:
        prob, pred, threshold = predict_one(features.model_dump())
        return PredictionResponse(
            churn_probability=prob,
            churn_prediction=pred,
            threshold=threshold,
        )
    except FileNotFoundError as e:
        logger.error("Model files unavailable: %s", e)
        raise HTTPException(status_code=503, detail=f"Model files unavailable: {str(e)}")
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch_endpoint(request: BatchPredictRequest):
    try:
        samples = [s.model_dump() for s in request.samples]
        results = predict_batch(samples)
        return BatchPredictResponse(
            predictions=[PredictionResponse(**r) for r in results],
            count=len(results),
        )
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch prediction error") from e
