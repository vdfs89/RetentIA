import pandera as pa

from src.features.columns import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN


def get_pandera_schema(is_training: bool = True) -> pa.DataFrameSchema:
    columns = {}

    for col in NUMERICAL_FEATURES:
        columns[col] = pa.Column(pa.Float, nullable=False)

    for col in CATEGORICAL_FEATURES:
        if col == "SeniorCitizen":
            columns[col] = pa.Column(pa.Int, nullable=False)
        else:
            columns[col] = pa.Column(pa.String, nullable=False)

    if is_training:
        columns[TARGET_COLUMN] = pa.Column(
            pa.String, checks=pa.Check.isin(["Yes", "No"]), nullable=False
        )

    return pa.DataFrameSchema(columns=columns, coerce=True, strict=False)
