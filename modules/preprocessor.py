"""
Data-cleaning helpers used by the Streamlit Cleaning Studio.

These functions are for exploratory data cleaning and previewing. They are
NOT substitutes for train/test-safe preprocessing pipelines in machine
learning. For ML, fit imputers, encoders and scalers on training data only,
inside a scikit-learn Pipeline or ColumnTransformer.

All functions return a copy and leave the caller's DataFrame unchanged.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _copy_dataframe(dataframe):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    return dataframe.copy(deep=True)


def _is_identifier_column(dataframe, column):
    """Conservative identifier detection for avoiding accidental transforms."""
    name = str(column).strip().lower()
    compact = name.replace(" ", "").replace("-", "").replace("_", "")
    if compact in {"id", "index", "rowid", "recordid"} or compact.endswith("id"):
        return True
    values = dataframe[column].dropna()
    return (
        len(values) >= 20
        and pd.api.types.is_numeric_dtype(dataframe[column])
        and values.nunique() / len(values) >= 0.98
    )


def handle_missing_values(dataframe, strategy="auto"):
    """Handle missing values in a copy. Strategies: auto, mean, median, mode, drop."""
    cleaned = _copy_dataframe(dataframe)
    strategy = str(strategy).strip().lower()
    allowed = {"auto", "mean", "median", "mode", "drop"}
    if strategy not in allowed:
        raise ValueError(f"Unsupported strategy '{strategy}'. Choose from {sorted(allowed)}.")
    if strategy == "drop":
        return cleaned.dropna(axis=0).copy()

    for column in cleaned.columns:
        series = cleaned[column]
        if not series.isna().any() or series.isna().all():
            continue
        numeric = (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        )
        fill_strategy = ("median" if numeric else "mode") if strategy == "auto" else strategy

        if fill_strategy in {"mean", "median"}:
            if not numeric:
                continue
            values = pd.to_numeric(series, errors="coerce")
            fill_value = values.mean() if fill_strategy == "mean" else values.median()
            if pd.notna(fill_value):
                cleaned[column] = series.fillna(fill_value)
        elif fill_strategy == "mode":
            modes = series.mode(dropna=True)
            if not modes.empty:
                cleaned[column] = series.fillna(modes.iloc[0])
    return cleaned


def handle_duplicates(dataframe, keep="first"):
    """Remove duplicate rows; return (cleaned_dataframe, duplicate_count)."""
    cleaned = _copy_dataframe(dataframe)
    if keep not in {"first", "last", False}:
        raise ValueError("keep must be 'first', 'last', or False.")
    count = int(cleaned.duplicated(keep=keep).sum())
    return cleaned.drop_duplicates(keep=keep).copy(), count


def encode_categorical_columns(dataframe, method="onehot"):
    """Encode categorical columns using onehot or compatibility label encoding."""
    cleaned = _copy_dataframe(dataframe)
    method = str(method).strip().lower()
    if method not in {"onehot", "label"}:
        raise ValueError("method must be 'onehot' or 'label'.")
    columns = cleaned.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()
    columns = [c for c in columns if not _is_identifier_column(cleaned, c)]
    if not columns:
        return cleaned
    if method == "onehot":
        return pd.get_dummies(cleaned, columns=columns, dummy_na=False, dtype=int)
    for column in columns:
        series = cleaned[column]
        categories = sorted(series.dropna().unique().tolist(), key=lambda x: str(x).casefold())
        mapping = {value: index for index, value in enumerate(categories)}
        cleaned[column] = series.map(mapping).astype("Int64")
    return cleaned


def scale_numerical_columns(dataframe, method="standard"):
    """Scale eligible numeric columns; preserve missing and infinite entries."""
    cleaned = _copy_dataframe(dataframe)
    method = str(method).strip().lower()
    if method not in {"standard", "minmax"}:
        raise ValueError("method must be 'standard' or 'minmax'.")
    columns = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    columns = [
        c for c in columns
        if not pd.api.types.is_bool_dtype(cleaned[c])
        and not _is_identifier_column(cleaned, c)
    ]
    for column in columns:
        original = pd.to_numeric(cleaned[column], errors="coerce")
        mask = original.notna() & np.isfinite(original)
        if int(mask.sum()) < 2:
            continue
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        transformed = scaler.fit_transform(original.loc[mask].to_numpy().reshape(-1, 1)).ravel()
        result = original.copy()
        result.loc[mask] = transformed
        cleaned[column] = result
    return cleaned


def extract_datetime_features(dataframe, columns=None):
    """Extract date/time components without overwriting existing columns."""
    cleaned = _copy_dataframe(dataframe)
    if columns is None:
        columns = cleaned.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    for column in columns:
        if column not in cleaned.columns:
            continue
        parsed = pd.to_datetime(cleaned[column], errors="coerce")
        if parsed.notna().sum() == 0:
            continue
        components = {
            f"{column}_year": parsed.dt.year,
            f"{column}_month": parsed.dt.month,
            f"{column}_day": parsed.dt.day,
            f"{column}_weekday": parsed.dt.dayofweek,
        }
        has_time = (
            (parsed.dt.hour.fillna(0) != 0).any()
            or (parsed.dt.minute.fillna(0) != 0).any()
            or (parsed.dt.second.fillna(0) != 0).any()
        )
        if has_time:
            components[f"{column}_hour"] = parsed.dt.hour
        for feature_name, values in components.items():
            if feature_name not in cleaned.columns:
                cleaned[feature_name] = values
    return cleaned
