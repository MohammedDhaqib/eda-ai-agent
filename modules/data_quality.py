"""
Data quality assessment for the AI-Powered Autonomous Data Analyst.

This module performs deterministic checks only. It does not modify the
input DataFrame.

Compatibility note:
- Existing keys used by app.py are retained.
- `quality_score` is temporarily retained for the current Streamlit UI.
  It is marked as deprecated and should be removed when the UI is updated
  to display the transparent `quality_checks` table instead.
"""

import numpy as np
import pandas as pd


def _is_identifier_column(series, column_name):
    """Return True for columns that are very likely to be identifiers."""
    name = str(column_name).strip().lower()
    compact_name = name.replace(" ", "").replace("-", "").replace("_", "")

    common_id_names = {
        "id", "index", "rowid", "recordid", "customerid",
        "userid", "transactionid", "orderid", "flightid",
    }

    if compact_name in common_id_names or compact_name.endswith("id"):
        return True

    # A mostly unique integer column is often an index/record identifier.
    non_missing = series.dropna()
    if len(non_missing) >= 20 and pd.api.types.is_numeric_dtype(series):
        uniqueness_ratio = non_missing.nunique() / len(non_missing)
        if uniqueness_ratio >= 0.98:
            return True

    return False


def _quality_check(name, status, details, count=None, percentage=None):
    """Create one consistently structured quality-check record."""
    return {
        "Check": name,
        "Status": status,
        "Details": details,
        "Count": count,
        "Percentage": percentage,
    }


def _build_missing_report(dataframe):
    """Create a per-column missing-value report."""
    row_count = len(dataframe)
    missing_counts = dataframe.isna().sum()

    report = pd.DataFrame({
        "Column": dataframe.columns,
        "Missing Values": missing_counts.values,
    })

    if row_count:
        report["Missing Percentage"] = (
            report["Missing Values"] / row_count * 100
        ).round(2)
    else:
        report["Missing Percentage"] = 0.0

    report = report[report["Missing Values"] > 0]
    report = report.sort_values(
        by="Missing Values", ascending=False
    ).reset_index(drop=True)

    return report


def _build_outlier_report(dataframe):
    """
    Detect potential univariate outliers with the IQR rule.

    Identifier-like columns are excluded. Infinite values are excluded from
    quartile calculations. IQR flags are descriptive indicators, not proof
    that a value is erroneous. Strongly skewed columns are still reported,
    but explicitly marked for cautious interpretation.
    """
    rows = []

    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns

    for column in numeric_columns:
        series = pd.to_numeric(dataframe[column], errors="coerce")
        finite_values = series[np.isfinite(series)]

        if _is_identifier_column(series, column):
            continue

        if len(finite_values) < 4:
            continue

        q1 = finite_values.quantile(0.25)
        q3 = finite_values.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(q1) or pd.isna(q3) or pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (finite_values < lower_bound) | (
            finite_values > upper_bound
        )
        outlier_count = int(outlier_mask.sum())

        # Skewness is used as a caution flag, not to silently discard results.
        skewness = finite_values.skew()
        interpretation = (
            "Use caution: distribution is strongly skewed; IQR flags may "
            "reflect natural tail values."
            if pd.notna(skewness) and abs(skewness) > 1
            else "Potential outliers based on the 1.5 × IQR rule."
        )

        rows.append({
            "Column": column,
            "Q1": round(float(q1), 4),
            "Q3": round(float(q3), 4),
            "IQR": round(float(iqr), 4),
            "Lower Bound": round(float(lower_bound), 4),
            "Upper Bound": round(float(upper_bound), 4),
            "Outlier Count": outlier_count,
            "Outlier Percentage": round(
                outlier_count / len(finite_values) * 100, 2
            ),
            "Skewness": (
                round(float(skewness), 4)
                if pd.notna(skewness) else np.nan
            ),
            "Interpretation": interpretation,
        })

    columns = [
        "Column", "Q1", "Q3", "IQR", "Lower Bound", "Upper Bound",
        "Outlier Count", "Outlier Percentage", "Skewness", "Interpretation",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("Outlier Count", ascending=False)
        .reset_index(drop=True)
    )


def assess_data_quality(dataframe):
    """
    Assess missing values, duplicate rows, potential numeric outliers,
    and basic structural quality checks.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataset to assess. The original DataFrame is never modified.

    Returns
    -------
    dict
        Quality metrics and reports. Existing application keys are preserved.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("assess_data_quality expects a pandas DataFrame.")

    row_count, column_count = dataframe.shape
    total_cells = row_count * column_count

    # Missing values
    missing_report = _build_missing_report(dataframe)
    total_missing = int(dataframe.isna().sum().sum())
    missing_percentage = (
        total_missing / total_cells * 100 if total_cells else 0.0
    )

    # Duplicate rows
    duplicate_count = int(dataframe.duplicated().sum())
    duplicate_percentage = (
        duplicate_count / row_count * 100 if row_count else 0.0
    )

    # Potential outliers
    outlier_report = _build_outlier_report(dataframe)
    total_outliers = (
        int(outlier_report["Outlier Count"].sum())
        if not outlier_report.empty else 0
    )

    # Transparent checks
    quality_checks = []

    quality_checks.append(_quality_check(
        "Missing values",
        "PASS" if total_missing == 0 else "WARNING",
        (
            "No missing cells were detected."
            if total_missing == 0
            else f"{total_missing:,} of {total_cells:,} cells are missing "
                 f"({missing_percentage:.2f}%)."
        ),
        count=total_missing,
        percentage=round(missing_percentage, 2),
    ))

    quality_checks.append(_quality_check(
        "Duplicate rows",
        "PASS" if duplicate_count == 0 else "WARNING",
        (
            "No fully duplicated rows were detected."
            if duplicate_count == 0
            else f"{duplicate_count:,} of {row_count:,} rows are duplicates "
                 f"({duplicate_percentage:.2f}%)."
        ),
        count=duplicate_count,
        percentage=round(duplicate_percentage, 2),
    ))

    constant_columns = [
        column for column in dataframe.columns
        if dataframe[column].nunique(dropna=False) <= 1
    ]
    quality_checks.append(_quality_check(
        "Constant columns",
        "PASS" if not constant_columns else "WARNING",
        (
            "No constant columns were detected."
            if not constant_columns
            else "Constant columns: " + ", ".join(map(str, constant_columns))
        ),
        count=len(constant_columns),
    ))

    all_missing_columns = [
        column for column in dataframe.columns
        if dataframe[column].isna().all()
    ]
    quality_checks.append(_quality_check(
        "Entirely missing columns",
        "PASS" if not all_missing_columns else "WARNING",
        (
            "No entirely missing columns were detected."
            if not all_missing_columns
            else "Entirely missing columns: "
                 + ", ".join(map(str, all_missing_columns))
        ),
        count=len(all_missing_columns),
    ))

    quality_checks.append(_quality_check(
        "Potential numeric outliers",
        "INFO" if total_outliers == 0 else "REVIEW",
        (
            "No IQR outlier flags were produced for eligible numeric columns."
            if total_outliers == 0
            else f"{total_outliers:,} column-level IQR flags were detected. "
                 "These are potential unusual values, not confirmed errors. "
                 "Counts are summed across columns, so one row may be counted "
                 "more than once."
        ),
        count=total_outliers,
    ))

    # Recommendations are action-oriented and avoid automatically changing data.
    recommendations = []

    if total_missing > 0:
        recommendations.append(
            "Review missing values by column and investigate why they are "
            "missing before choosing an imputation or removal strategy."
        )

    if duplicate_count > 0:
        recommendations.append(
            "Review duplicate rows to determine whether they are accidental "
            "duplicates or valid repeated observations."
        )

    if constant_columns:
        recommendations.append(
            "Review constant columns; they usually provide no information "
            "for analysis or predictive modelling."
        )

    if all_missing_columns:
        recommendations.append(
            "Review entirely missing columns. They may need to be removed "
            "or populated from a reliable source."
        )

    if total_outliers > 0:
        recommendations.append(
            "Inspect potential outliers in context. Do not remove them "
            "automatically; they may represent valid rare observations."
        )

    if not recommendations:
        recommendations.append(
            "No missing values, duplicate rows, constant columns, or "
            "eligible IQR outlier flags were detected by these checks. "
            "This does not guarantee that the dataset is error-free."
        )

    # Temporary compatibility value for the current app.py.
    # This is the old heuristic score, not a statistically validated measure.
    # The UI will be changed to show quality_checks in a later step.
    missing_rate = missing_percentage
    duplicate_rate = duplicate_percentage
    numeric_cell_count = int(
        dataframe.select_dtypes(include=[np.number]).size
    )
    outlier_rate = (
        total_outliers / numeric_cell_count * 100
        if numeric_cell_count else 0.0
    )
    legacy_quality_score = round(
        max(
            0.0,
            min(
                100.0,
                100 - missing_rate * 0.5
                - duplicate_rate * 0.25
                - outlier_rate * 0.25,
            ),
        ),
        2,
    )

    return {
        # Existing keys retained for current app.py and LLM context builders.
        "missing_report": missing_report,
        "total_missing": total_missing,
        "duplicate_count": duplicate_count,
        "duplicate_percentage": round(duplicate_percentage, 2),
        "outlier_report": outlier_report,
        "total_outliers": total_outliers,
        "recommendations": recommendations,
        "quality_score": legacy_quality_score,

        # New transparent quality-check output.
        "quality_checks": pd.DataFrame(quality_checks),
        "missing_percentage": round(missing_percentage, 2),
        "constant_columns": constant_columns,
        "all_missing_columns": all_missing_columns,
        "quality_score_deprecated": True,
        "outlier_count_note": (
            "total_outliers is the sum of per-column IQR flags, not the "
            "number of unique rows containing an outlier."
        ),
    }
