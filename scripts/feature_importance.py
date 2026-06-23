"""Extract and display XGBoost feature importances from the trained pipeline."""

from __future__ import annotations

import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer


def get_feature_names(transformer: ColumnTransformer) -> list[str]:
    """Reconstruct feature names after ColumnTransformer (scaling + OHE).

    Handles OneHotEncoder(drop="first"): the dropped category produces no
    output column, so it must be excluded to stay aligned with the model's
    feature_importances_ vector.
    """
    names: list[str] = []
    for name, trans, cols in transformer.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            ohe = trans
            for i, (col, cats) in enumerate(zip(cols, ohe.categories_)):
                drop_idx = None
                if ohe.drop_idx_ is not None and ohe.drop_idx_[i] is not None:
                    drop_idx = int(ohe.drop_idx_[i])
                kept = [c for j, c in enumerate(cats) if j != drop_idx]
                names.extend([f"{col}={c}" for c in kept])
    return names


def main() -> None:
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("models/xgboost.pkl", "rb") as f:
        xgb = pickle.load(f)

    feature_names = get_feature_names(preprocessor.transformer)
    importances = xgb.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            f"Feature name count ({len(feature_names)}) does not match "
            f"importance count ({len(importances)}) — check OHE drop handling."
        )

    df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    pd.set_option("display.max_rows", None)
    print("\nTop 15 features by XGBoost gain importance:\n")
    print(df.head(15).to_string(index=False))

    df.to_csv("docs/feature_importance.csv", index=False)
    print("\nSaved full ranking to docs/feature_importance.csv")


if __name__ == "__main__":
    main()
