import pandas as pd
import pandera as pa
import pytest

from src.validation.schemas import get_pandera_schema


def test_pandera_validation_passes():
    schema = get_pandera_schema(is_training=True)
    df = pd.DataFrame(
        {
            "gender": ["Female"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [1],
            "PhoneService": ["No"],
            "MultipleLines": ["No phone service"],
            "InternetService": ["DSL"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["Yes"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [29.85],
            "TotalCharges": [29.85],
            "Churn": ["No"],
        }
    )
    validated_df = schema.validate(df)
    assert not validated_df.empty


def test_pandera_validation_fails_on_missing_column():
    schema = get_pandera_schema(is_training=True)
    df = pd.DataFrame({"gender": ["Female"]})
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)
