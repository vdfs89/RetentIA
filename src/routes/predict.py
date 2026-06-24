import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import CustomerFeatures, PredictionResponse
from src.services.model_service import predict_one

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
