"""
Rule-based dataset insights for the EDA AI Agent.

All numeric observations are calculated with pandas/numpy. This module does not
ask an LLM to calculate statistics and does not modify the input DataFrame.

Compatibility:
- Keeps the public function name generate_dataset_insights.
- Returns a list of human-readable strings.
"""

import numpy as np
import pandas as pd


def _is_identifier_column(dataframe, column):
    """Conservatively identify columns that are probably row/entity IDs."""
    name = str(column).strip().lower()
    compact_name = name.replace(" ", "").replace("-", "").replace("_", "")

    if compact_name in {"id", "index", "rowid", "recordid"} or compact_name.endswith("id"):
        return True

    values = dataframe[column].dropna()
    if len(values) >= 20 and pd.api.types.is_numeric_dtype(dataframe[column]):
        return values.nunique() / len(values) >= 0.98

    return False


def _format_number(value, decimals=2):
    """Format finite numeric values safely for insight text."""
    try:
        if pd.isna(value) or not np.isfinite(value):
            return "not available"
        return f"{value:,.{decimals}f}"
    except (TypeError, ValueError):
        return "not available"


def _numeric_columns(dataframe):
    """Return numeric columns excluding booleans and likely identifiers."""
    columns = dataframe.select_dtypes(include=[np.number]).columns
    return [
        column for column in columns
        if not pd.api.types.is_bool_dtype(dataframe[column])
        and not _is_identifier_column(dataframe, column)
    ]


def _categorical_columns(dataframe):
    """Return categorical columns, including boolean columns."""
    return dataframe.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns


def _add_overview_insights(dataframe, insights):
    row_count, column_count = dataframe.shape
    insights.append(f"Dataset overview: {row_count:,} rows and {column_count:,} columns.")

    numeric_count = len(_numeric_columns(dataframe))
    categorical_count = len(_categorical_columns(dataframe))
    datetime_count = len(dataframe.select_dtypes(include=["datetime", "datetimetz"]).columns)

    insights.append(
        f"Detected {numeric_count} eligible numeric columns, "
        f"{categorical_count} categorical/boolean columns, and "
        f"{datetime_count} columns stored as datetime."
    )


def _add_missing_and_duplicate_insights(dataframe, insights):
    row_count, column_count = dataframe.shape
    total_cells = row_count * column_count
    missing_count = int(dataframe.isna().sum().sum())

    if missing_count:
        missing_pct = missing_count / total_cells * 100 if total_cells else 0
        insights.append(
            f"Missing data: {missing_count:,} of {total_cells:,} cells are "
            f"missing ({missing_pct:.2f}%)."
        )
        missing_by_column = dataframe.isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)
        for column, count in missing_by_column.items():
            pct = count / row_count * 100 if row_count else 0
            insights.append(f"'{column}' has {int(count):,} missing values ({pct:.2f}% of rows).")
    else:
        insights.append("Missing data: no missing cells were detected.")

    duplicate_count = int(dataframe.duplicated().sum())
    duplicate_pct = duplicate_count / row_count * 100 if row_count else 0
    insights.append(f"Duplicate rows: {duplicate_count:,} ({duplicate_pct:.2f}% of rows).")


def _add_numeric_insights(dataframe, insights):
    columns = _numeric_columns(dataframe)
    if not columns:
        insights.append(
            "Numeric distribution: no eligible numeric columns were available "
            "after excluding likely identifiers."
        )
        return

    for column in columns:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue

        skewness = values.skew()
        if pd.isna(skewness):
            shape = "skewness could not be estimated"
        elif skewness > 1:
            shape = "strongly right-skewed"
        elif skewness > 0.5:
            shape = "moderately right-skewed"
        elif skewness < -1:
            shape = "strongly left-skewed"
        elif skewness < -0.5:
            shape = "moderately left-skewed"
        else:
            shape = "not strongly skewed based on the skewness measure"

        insights.append(
            f"Numeric distribution for column '{column}': n={len(values):,}, "
            f"mean={_format_number(values.mean())}, "
            f"median={_format_number(values.median())}, "
            f"standard deviation={_format_number(values.std())}, "
            f"skewness={_format_number(skewness, 3)} ({shape})."
        )


def _add_categorical_insights(dataframe, insights, max_categories=30):
    """
    Store exact counts and percentages for every category when cardinality is
    modest. For high-cardinality columns, store the top categories and clearly
    state how many categories were omitted.
    """
    for column in _categorical_columns(dataframe):
        values = dataframe[column].dropna().astype(str)
        if values.empty:
            insights.append(f"Categorical distribution for column '{column}': no non-missing values.")
            continue

        counts = values.value_counts(dropna=False)
        total = int(counts.sum())
        category_count = int(len(counts))
        shown_counts = counts.head(max_categories)

        details = []
        for category, count in shown_counts.items():
            count = int(count)
            percentage = count / total * 100 if total else 0
            details.append(f"'{category}' = {count:,} ({percentage:.2f}%)")

        omitted = category_count - len(shown_counts)
        suffix = f"; {omitted:,} lower-frequency categories omitted" if omitted else ""

        insights.append(
            f"Categorical distribution for column '{column}': "
            f"{category_count:,} observed categories among {total:,} non-missing rows. "
            f"Category counts and percentages: {'; '.join(details)}{suffix}."
        )


def _add_correlation_insights(dataframe, insights, max_pairs=5):
    columns = _numeric_columns(dataframe)
    if len(columns) < 2:
        return

    numeric_data = dataframe[columns].replace([np.inf, -np.inf], np.nan)
    pearson = numeric_data.corr(method="pearson", min_periods=3)
    spearman = numeric_data.corr(method="spearman", min_periods=3)

    pairs = []
    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            p_value = pearson.loc[first, second]
            s_value = spearman.loc[first, second]
            if pd.notna(p_value):
                rank = max(abs(float(p_value)), abs(float(s_value)) if pd.notna(s_value) else 0.0)
                pairs.append((rank, first, second, float(p_value),
                              float(s_value) if pd.notna(s_value) else np.nan))

    pairs.sort(key=lambda item: item[0], reverse=True)
    meaningful_pairs = [pair for pair in pairs if pair[0] >= 0.30]

    if not meaningful_pairs:
        insights.append(
            "Correlation scan: no numeric variable pair reached an absolute "
            "Pearson or Spearman correlation of 0.30. This does not rule out "
            "non-monotonic or more complex relationships."
        )
        return

    for _, first, second, p_value, s_value in meaningful_pairs[:max_pairs]:
        insights.append(
            f"Association between '{first}' and '{second}': "
            f"Pearson r={p_value:.3f}; Spearman rho={_format_number(s_value, 3)}. "
            "These are associations, not evidence of causation."
        )


def generate_dataset_insights(dataframe):
    """Generate deterministic descriptive insights from a pandas DataFrame."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("generate_dataset_insights expects a pandas DataFrame.")

    if dataframe.empty:
        return ["The dataset is empty; no descriptive insights can be generated."]

    insights = []
    _add_overview_insights(dataframe, insights)
    _add_missing_and_duplicate_insights(dataframe, insights)
    _add_numeric_insights(dataframe, insights)
    _add_categorical_insights(dataframe, insights)
    _add_correlation_insights(dataframe, insights)

    # Do not truncate: every categorical column's distribution must remain
    # available to indexing, including columns near the end of wide datasets.
    return insights
