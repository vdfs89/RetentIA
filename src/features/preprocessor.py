import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.columns import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


class ChurnPreprocessor:
    def __init__(self):
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERICAL_FEATURES),
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )

    def fit(self, df: pd.DataFrame):
        self.transformer.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.transformer.transform(df)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
