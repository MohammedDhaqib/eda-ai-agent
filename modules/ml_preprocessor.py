"""
Leakage-safe tabular preprocessing.

Use this inside an sklearn Pipeline. The transformer is fitted only when
Pipeline.fit() is called on training data.
"""
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(features):
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if features.empty or features.shape[1] == 0:
        raise ValueError("No feature columns are available.")
    if not features.columns.is_unique:
        raise ValueError("Feature column names must be unique.")

    frame = features.copy()
    # Replace infinities before fitting imputers.
    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    if numeric_columns:
        frame.loc[:, numeric_columns] = frame[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )

    # Select supported types. Fail rather than silently discard unsupported
    # columns such as complex values or arbitrary Python objects.
    categorical_columns = frame.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()
    supported = set(numeric_columns + categorical_columns)
    unsupported = [c for c in frame.columns if c not in supported]
    if unsupported:
        raise ValueError(
            "Unsupported feature types in columns: "
            + ", ".join(map(str, unsupported))
        )
    if not numeric_columns and not categorical_columns:
        raise ValueError("No numerical or categorical features were found.")

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="constant", fill_value="Missing",
                keep_empty_features=True
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore", sparse_output=True
            )),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
