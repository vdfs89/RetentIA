import logging
import os
import urllib.request

import pandas as pd

logger = logging.getLogger(__name__)

DATA_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def ingest_data(
    file_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
) -> pd.DataFrame:
    """Loads CSV, verifies gotcha structure, and clean values."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.exists(file_path):
        logger.info("Dataset not found locally. Downloading from %s...", DATA_URL)
        urllib.request.urlretrieve(DATA_URL, file_path)
        logger.info("Download complete.")

    df = pd.read_csv(file_path)

    # Gotcha: TotalCharges contains blank spaces ' ' for new customers
    blank_mask = df["TotalCharges"] == " "
    tenure_zero_mask = df["tenure"] == 0
    if not blank_mask.equals(tenure_zero_mask):
        logger.warning("Blank spaces in TotalCharges do not align perfectly with tenure == 0.")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    return df
