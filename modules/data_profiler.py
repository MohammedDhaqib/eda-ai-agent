"""
Automated dataset profiling for the AI-Powered Autonomous Data Analyst.

The profiler is read-only: it never changes the input DataFrame.

The original return keys are preserved for compatibility with app.py:
- numerical_columns
- categorical_columns
- datetime_columns
- column_summary
- numerical_summary
- categorical_summary

Additional outputs provide column roles, constant/near-constant columns,
cardinality, likely type mismatches, sentinel values, and memory usage.
"""

import numpy as np
import pandas as pd


COMMON_SENTINELS = {
    "na", "n/a", "null", "none", "unknown", "?", "-", "--",
    "not available", "missing", "not known",
}


def _safe_name(column):
    """Convert a column label to a readable string."""
    return str(column)


def _is_datetime_series(series):
    """Detect native datetimes or object/string columns that parse as dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_timedelta64_dtype(series)
    ):
        return False

    values = series.dropna()
    if len(values) < 3:
        return False

    # Avoid classifying ordinary short numeric-like text as dates.
    sample = values.astype(str).head(1000)
    parsed = pd.to_datetime(sample, errors="coerce")
    parse_ratio = parsed.notna().mean()

    # A conservative threshold reduces accidental date detection.
    return bool(parse_ratio >= 0.80)


def _likely_numeric_text(series):
    """Return the proportion of non-empty text values parseable as numbers."""
    values = series.dropna()
    if values.empty:
        return 0.0

    converted = pd.to_numeric(values.astype(str).str.replace(",", "", regex=False),
                              errors="coerce")
    return float(converted.notna().mean())


def _sentinel_values(series):
    """Find common text sentinels and common numeric sentinel candidates."""
    found = []

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        normalized = series.dropna().astype(str).str.strip().str.lower()
        for value in sorted(COMMON_SENTINELS):
            count = int((normalized == value).sum())
            if count:
                found.append(f"{value} ({count})")

    if pd.api.types.is_numeric_dtype(series):
        numeric_sentinels = {-999, -9999, 999, 9999, -1}
        values = pd.to_numeric(series, errors="coerce")
        for value in sorted(numeric_sentinels):
            count = int((values == value).sum())
            if count:
                found.append(f"{value} ({count})")

    return found


def _column_role(series, is_datetime, is_constant, is_near_constant,
                 is_identifier, numeric_text_ratio):
    """Assign one practical primary role to a column."""
    if is_constant:
        return "constant"
    if is_datetime:
        return "datetime"
    if is_identifier:
        return "identifier"
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        unique_count = series.nunique(dropna=True)
        if unique_count <= 2:
            return "binary"
        return "numeric"
    if numeric_text_ratio >= 0.80 and series.notna().sum() >= 3:
        return "numeric stored as text"
    if is_near_constant:
        return "near-constant"
    if series.nunique(dropna=True) > 100:
        return "high-cardinality text"
    return "categorical"


def profile_dataset(dataframe):
    """
    Generate a descriptive profile of a pandas DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataset to profile.

    Returns
    -------
    dict
        Existing profile outputs plus additional diagnostic metadata.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("profile_dataset expects a pandas DataFrame.")

    row_count, column_count = dataframe.shape

    numerical_columns = []
    categorical_columns = []
    datetime_columns = []
    column_rows = []
    role_rows = []
    constant_columns = []
    near_constant_columns = []
    type_mismatch_columns = []
    sentinel_report_rows = []

    for column in dataframe.columns:
        series = dataframe[column]
        non_missing = series.dropna()
        non_missing_count = int(non_missing.shape[0])
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = (
            unique_count / non_missing_count if non_missing_count else 0.0
        )

        is_datetime = _is_datetime_series(series)
        is_numeric = (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
            and not is_datetime
        )
        is_bool = pd.api.types.is_bool_dtype(series)

        is_constant = unique_count <= 1
        is_near_constant = (
            non_missing_count >= 2
            and unique_count > 1
            and (non_missing.value_counts(normalize=True).iloc[0] >= 0.98)
        )

        name = _safe_name(column).strip().lower()
        compact_name = name.replace(" ", "").replace("-", "").replace("_", "")
        name_suggests_id = (
            compact_name in {"id", "index", "rowid", "recordid"}
            or compact_name.endswith("id")
        )
        is_identifier = (
            name_suggests_id
            or (
                non_missing_count >= 20
                and unique_ratio >= 0.98
                and (is_numeric or pd.api.types.is_string_dtype(series)
                     or pd.api.types.is_object_dtype(series))
            )
        )

        numeric_text_ratio = (
            _likely_numeric_text(series)
            if (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            )
            else 0.0
        )

        if is_datetime:
            datetime_columns.append(column)
        elif is_numeric:
            numerical_columns.append(column)
        elif not is_bool:
            categorical_columns.append(column)
        else:
            categorical_columns.append(column)

        if is_constant:
            constant_columns.append(column)
        if is_near_constant:
            near_constant_columns.append(column)

        mismatch_note = ""
        if numeric_text_ratio >= 0.80 and non_missing_count >= 3:
            type_mismatch_columns.append(column)
            mismatch_note = (
                f"Likely numeric values stored as text "
                f"({numeric_text_ratio * 100:.1f}% parseable)."
            )
        elif is_datetime and not pd.api.types.is_datetime64_any_dtype(series):
            type_mismatch_columns.append(column)
            mismatch_note = "Date-like values stored as text."

        sentinels = _sentinel_values(series)
        if sentinels:
            sentinel_report_rows.append({
                "Column": column,
                "Possible Sentinels": ", ".join(sentinels),
            })

        role = _column_role(
            series,
            is_datetime,
            is_constant,
            is_near_constant,
            is_identifier,
            numeric_text_ratio,
        )

        role_rows.append({
            "Column": column,
            "Role": role,
            "Data Type": str(series.dtype),
            "Unique Values": unique_count,
            "Unique Ratio (%)": round(unique_ratio * 100, 2),
            "Missing Values": missing_count,
            "Missing Percentage (%)": round(
                missing_count / row_count * 100, 2
            ) if row_count else 0.0,
        })

        column_rows.append({
            "Column": column,
            "Data Type": str(series.dtype),
            "Non-Null Count": non_missing_count,
            "Missing Values": missing_count,
            "Unique Values": unique_count,
            "Unique Ratio (%)": round(unique_ratio * 100, 2),
            "Column Role": role,
            "Potential Type Issue": mismatch_note,
        })

    # Preserve the familiar pandas describe-style summaries.
    if numerical_columns:
        numerical_summary = dataframe[numerical_columns].describe().T
    else:
        numerical_summary = pd.DataFrame()

    if categorical_columns:
        categorical_summary = dataframe[categorical_columns].describe(
            include="all"
        ).T
    else:
        categorical_summary = pd.DataFrame()

    column_summary = pd.DataFrame(column_rows)
    column_roles = pd.DataFrame(role_rows)
    sentinel_report = pd.DataFrame(
        sentinel_report_rows,
        columns=["Column", "Possible Sentinels"],
    )

    memory_usage_bytes = int(
        dataframe.memory_usage(index=True, deep=True).sum()
    )
    memory_usage_mb = memory_usage_bytes / (1024 ** 2)

    return {
        # Original keys retained for current app.py.
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "column_summary": column_summary,
        "numerical_summary": numerical_summary,
        "categorical_summary": categorical_summary,

        # Additional profiling outputs.
        "column_roles": column_roles,
        "constant_columns": constant_columns,
        "near_constant_columns": near_constant_columns,
        "type_mismatch_columns": type_mismatch_columns,
        "sentinel_report": sentinel_report,
        "memory_usage_bytes": memory_usage_bytes,
        "memory_usage_mb": round(memory_usage_mb, 2),
        "row_count": row_count,
        "column_count": column_count,
    }
