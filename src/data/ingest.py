import logging
import os
import urllib.request

import pandas as pd

logger = logging.getLogger(__name__)

DATA_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def ingest_data(
    file_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
) -> pd.DataFrame:
    """Carrega CSV, verifica a estrutura e limpa os valores."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.exists(file_path):
        logger.info("Dataset não encontrado localmente. Baixando de %s...", DATA_URL)
        urllib.request.urlretrieve(DATA_URL, file_path)
        logger.info("Download completo.")

    df = pd.read_csv(file_path)

    # Atenção: TotalCharges contém espaços em branco ' ' para novos clientes
    blank_mask = df["TotalCharges"] == " "
    tenure_zero_mask = df["tenure"] == 0
    if not blank_mask.equals(tenure_zero_mask):
        logger.warning(
            "Espaços em branco em TotalCharges não se alinham perfeitamente com tenure == 0."
        )

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    return df
