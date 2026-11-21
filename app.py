import streamlit as st
import pandas as pd
import hashlib

from modules.insights import generate_dataset_insights
from modules.data_loader import load_csv
from modules.data_profiler import profile_dataset
from modules.data_quality import assess_data_quality
from modules.visualizer import (
    plot_histogram, plot_boxplot, interpret_boxplot,
    plot_correlation_heatmap, interpret_correlation_heatmap,
    plot_scatter, interpret_scatter_plot, plot_bar_chart
)
from modules.preprocessor import (
    handle_missing_values, handle_duplicates,
    encode_categorical_columns, scale_numerical_columns,
    extract_datetime_features
)
from modules.ml_dashboard import render_ml_dashboard
from modules.llm_engine import (
    generate_dataset_summary,
    ask_dataset_question
)

def build_llm_dataset_information(dataframe, profile, quality, dataset_insights):
    """
    Build factual dataset context using Python-calculated statistics.
    The LLM explains these facts; it must not invent or recalculate them.
    """
    sections = []
    row_count, column_count = dataframe.shape

    sections.append(
        f"DATASET SIZE: {row_count:,} total rows and {column_count:,} total columns."
    )

    # Exact data types, counted from the actual DataFrame.
    dtype_counts = dataframe.dtypes.astype(str).value_counts()
    dtype_lines = [
        f"- {dtype}: {int(count):,} columns"
        for dtype, count in dtype_counts.items()
    ]
    sections.append(
        "DATA TYPE DISTRIBUTION (these counts must add up to the total "
        f"number of columns, {column_count:,}):\n" + "\n".join(dtype_lines)
    )

    column_details = [
        f"- {column}: {dataframe[column].dtype}"
        for column in dataframe.columns
    ]
    sections.append(
        "COLUMNS AND DATA TYPES:\n" + "\n".join(column_details[:80])
    )

    missing_counts = dataframe.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if not missing_counts.empty:
        missing_lines = []
        for column, count in missing_counts.head(30).items():
            percentage = int(count) / row_count * 100 if row_count else 0
            missing_lines.append(
                f"- {column}: {int(count):,} missing out of {row_count:,} "
                f"total rows ({percentage:.2f}% of all rows)."
            )
        sections.append(
            "MISSING VALUES (denominator = total dataset rows):\n"
            + "\n".join(missing_lines)
        )
    else:
        sections.append("MISSING VALUES: No missing values detected.")

    duplicate_count = int(dataframe.duplicated().sum())
    duplicate_percentage = duplicate_count / row_count * 100 if row_count else 0
    sections.append(
        f"DUPLICATES: {duplicate_count:,} duplicate rows out of {row_count:,} "
        f"total rows ({duplicate_percentage:.2f}%)."
    )

    # Use the exact outlier results already calculated by the Data Quality module.
    total_outliers = int(quality.get("total_outliers", 0) or 0)
    outlier_report = quality.get("outlier_report")

    sections.append(
        "OUTLIER DETECTION (IQR METHOD): "
        f"{total_outliers:,} total potential outliers were detected by the "
        "Data Quality Assessment. This is the authoritative total from the "
        "application's IQR calculation. Never say that no outliers were "
        "detected when this value is greater than zero. These are potential "
        "statistical outliers, not automatically data errors."
    )

    if outlier_report is not None and not outlier_report.empty:
        sections.append(
            "IQR OUTLIER REPORT BY NUMERICAL COLUMN (use the table's exact "
            "column names and counts; do not invent missing entries):\n"
            + outlier_report.to_string(index=False, max_rows=60)
        )
    else:
        sections.append(
            "IQR OUTLIER REPORT: The report table contains no per-column rows. "
            "Do not claim that the dataset has zero outliers because the "
            "authoritative total above is supplied separately."
        )

    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    if numeric_columns:
        numeric_summary = dataframe[numeric_columns[:15]].describe().round(2)
        sections.append(
            "NUMERICAL DESCRIPTIVE STATISTICS (means are descriptive; "
            "do not label them high or low without a comparison):\n"
            + numeric_summary.to_string()
        )
    else:
        sections.append("NUMERICAL STATISTICS: No numerical columns detected.")

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    if categorical_columns:
        category_sections = []
        for column in categorical_columns[:20]:
            non_missing = dataframe[column].dropna()
            denominator = len(non_missing)
            counts = non_missing.astype("string").value_counts().head(10)

            category_sections.append(
                f"COLUMN: {column} | non-missing denominator: {denominator:,} "
                f"| missing: {int(dataframe[column].isna().sum()):,}"
            )

            if denominator == 0:
                category_sections.append("  No non-missing category values.")
                continue

            for value, count in counts.items():
                percentage = int(count) / denominator * 100
                category_sections.append(
                    f"  - {str(value)[:100]}: {int(count):,} of {denominator:,} "
                    f"non-missing values ({percentage:.2f}% of non-missing values)."
                )

            distinct_count = int(non_missing.nunique())
            if distinct_count > len(counts):
                category_sections.append(
                    f"  - {distinct_count - len(counts):,} additional distinct "
                    "categories are not listed."
                )

        sections.append(
            "CATEGORICAL VALUE COUNTS (each percentage denominator is the "
            "non-missing count for that column):\n"
            + "\n".join(category_sections)
        )
    else:
        sections.append("CATEGORICAL STATISTICS: No categorical columns detected.")

    quality_checks = quality.get("quality_checks")
    if quality_checks is not None and not quality_checks.empty:
        sections.append(
            "TRANSPARENT DATA QUALITY CHECKS (statuses are PASS, WARNING, "
            "INFO, or REVIEW; do not convert these into a single score):\n"
            + quality_checks.to_string(index=False)
        )
    sections.append(
        f"TOTAL MISSING CELLS: {quality.get('total_missing', 'Not available')}."
    )
    sections.append(
        f"DUPLICATE ROWS FROM QUALITY ASSESSMENT: "
        f"{quality.get('duplicate_count', 'Not available')}."
    )

    recommendations = quality.get("recommendations", [])
    if recommendations:
        sections.append(
            "RULE-BASED DATA QUALITY RECOMMENDATIONS:\n"
            + "\n".join(f"- {item}" for item in recommendations[:8])
        )

    if dataset_insights:
        sections.append(
            "EXISTING RULE-BASED INSIGHTS (may overlap with statistics above):\n"
            + "\n".join(f"- {item}" for item in dataset_insights[:8])
        )

    return "\n\n".join(sections)[:20000]


def build_verified_dataset_facts(dataframe, quality):
    """
    Build the factual summary with Python only.
    These figures are displayed separately from the LLM's qualitative
    explanation so small language models cannot alter the statistics.
    """
    row_count, column_count = dataframe.shape
    lines = [
        f"- Dataset size: {row_count:,} rows and {column_count:,} columns.",
        f"- Total missing cells: {int(dataframe.isna().sum().sum()):,}.",
    ]

    missing_by_column = dataframe.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0].sort_values(
        ascending=False
    )

    if missing_by_column.empty:
        lines.append("- Missing values: none detected.")
    else:
        lines.append("- Missing values by column (percentage of all dataset rows):")
        for column, count in missing_by_column.items():
            percentage = (int(count) / row_count * 100) if row_count else 0
            lines.append(
                f"  - {column}: {int(count):,} ({percentage:.2f}% of "
                f"{row_count:,} rows)."
            )

    duplicate_count = int(dataframe.duplicated().sum())
    lines.append(f"- Duplicate rows: {duplicate_count:,}.")

    dtype_counts = dataframe.dtypes.astype(str).value_counts()
    dtype_text = ", ".join(
        f"{dtype}: {int(count)}" for dtype, count in dtype_counts.items()
    )
    lines.append(f"- Data type distribution: {dtype_text}.")

    total_outliers = int(quality.get("total_outliers", 0) or 0)
    lines.append(
        f"- Potential numerical outliers detected using IQR: {total_outliers:,}."
    )

    outlier_report = quality.get("outlier_report")
    if outlier_report is not None and not outlier_report.empty:
        lines.append("- IQR outlier counts by numerical column:")
        for _, row in outlier_report.iterrows():
            lines.append(
                f"  - {row['Column']}: {int(row['Outlier Count']):,} "
                f"potential outliers."
            )

    quality_checks = quality.get("quality_checks")
    if quality_checks is not None and not quality_checks.empty:
        lines.append("- Transparent data quality checks:")
        for _, check in quality_checks.iterrows():
            lines.append(
                f"  - {check['Check']}: {check['Status']} — {check['Details']}"
            )

    numeric_columns = dataframe.select_dtypes(include=["number"]).columns
    if len(numeric_columns) > 0:
        lines.append("- Selected numerical statistics (calculated by pandas):")
        description = dataframe[numeric_columns].describe().round(2)
        for column in numeric_columns[:15]:
            if column in description.columns:
                lines.append(
                    f"  - {column}: mean={description.loc['mean', column]:,.2f}; "
                    f"median={dataframe[column].median():,.2f}."
                )

    return "\n".join(lines)


def build_verified_dataset_report(dataframe, quality):
    """
    Create the main dataset report deterministically with Python.
    All counts, percentages, metrics, and recommendations come from
    calculations on the uploaded DataFrame and quality assessment.
    """
    row_count, column_count = dataframe.shape
    missing_total = int(dataframe.isna().sum().sum())
    duplicate_total = int(dataframe.duplicated().sum())
    total_outliers = int(quality.get("total_outliers", 0) or 0)

    report = []
    report.append("## 1. Dataset Overview")
    report.append(
        f"The dataset contains **{row_count:,} rows** and "
        f"**{column_count:,} columns**."
    )

    dtype_counts = dataframe.dtypes.astype(str).value_counts()
    dtype_summary = ", ".join(
        f"{dtype}: {int(count)}" for dtype, count in dtype_counts.items()
    )
    report.append(f"Data types: {dtype_summary}.")

    report.append("\n## 2. Missing-Value Analysis")
    if missing_total == 0:
        report.append("No missing values were detected.")
    else:
        report.append(f"Total missing cells: **{missing_total:,}**.")
        missing_by_column = dataframe.isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0].sort_values(
            ascending=False
        )
        for column, count in missing_by_column.items():
            pct_rows = (int(count) / row_count * 100) if row_count else 0
            pct_cells = (
                int(count) / (row_count * column_count) * 100
                if row_count and column_count else 0
            )
            report.append(
                f"- **{column}:** {int(count):,} missing values "
                f"({pct_rows:.2f}% of dataset rows; "
                f"{pct_cells:.4f}% of all dataset cells)."
            )

    report.append("\n## 3. Duplicate-Row Analysis")
    report.append(f"Duplicate rows detected: **{duplicate_total:,}**.")
    if duplicate_total == 0:
        report.append("No duplicate-removal action is indicated by this check.")
    else:
        report.append(
            "Review duplicate records and confirm whether they are genuine "
            "duplicates before removing them."
        )

    report.append("\n## 4. IQR Outlier Analysis")
    report.append(
        f"Total potential numerical outliers detected using the IQR method: "
        f"**{total_outliers:,}**."
    )
    outlier_report = quality.get("outlier_report")
    if outlier_report is not None and not outlier_report.empty:
        for _, row in outlier_report.iterrows():
            report.append(
                f"- **{row['Column']}:** {int(row['Outlier Count']):,} "
                f"potential outliers "
                f"(lower bound: {row['Lower Bound']}; "
                f"upper bound: {row['Upper Bound']})."
            )
    if total_outliers > 0:
        report.append(
            "IQR flags statistically unusual values; it does not prove that "
            "those records are incorrect. Investigate them in context before "
            "deciding whether any correction is appropriate."
        )
    else:
        report.append("No numerical outliers were flagged by the IQR check.")

    report.append("\n## 5. Numerical Distribution")
    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        report.append("No numerical columns were available for this analysis.")
    else:
        for column in numeric_columns[:20]:
            series = dataframe[column].dropna()
            if series.empty:
                report.append(f"- **{column}:** no non-missing numeric values.")
                continue
            mean_value = float(series.mean())
            median_value = float(series.median())
            skew_value = float(series.skew()) if len(series) > 2 else float("nan")
            line = (
                f"- **{column}:** mean = {mean_value:,.2f}; "
                f"median = {median_value:,.2f}"
            )
            if pd.notna(skew_value):
                if skew_value > 0.5:
                    shape = "right-skewed"
                elif skew_value < -0.5:
                    shape = "left-skewed"
                else:
                    shape = "approximately symmetric by the skewness rule"
                line += f"; skewness = {skew_value:.2f} ({shape})"
            report.append(line + ".")

    report.append("\n## 6. Transparent Data Quality Checks")
    quality_checks = quality.get("quality_checks")
    if quality_checks is not None and not quality_checks.empty:
        for _, check in quality_checks.iterrows():
            report.append(
                f"- **{check['Check']} — {check['Status']}:** {check['Details']}"
            )
    else:
        report.append("No structured quality-check results are available.")

    report.append("\n## 7. Recommended Next Steps")
    if missing_total > 0:
        report.append(
            "- Inspect the cause of missing values and choose an appropriate "
            "imputation or exclusion strategy based on the analysis goal."
        )
    if duplicate_total > 0:
        report.append(
            "- Review duplicate rows and remove them only after confirming "
            "they are unintended duplicates."
        )
    if total_outliers > 0:
        report.append(
            "- Inspect the per-column IQR outlier results and validate unusual "
            "records against domain context; do not delete them automatically."
        )
    if missing_total == 0 and duplicate_total == 0 and total_outliers == 0:
        report.append(
            "- No missing-value, duplicate-row, or IQR-outlier actions were "
            "flagged by these checks. Continue with domain-specific validation."
        )

    return "\n".join(report)


def validate_ai_explanation(text, evidence_text="", dataframe=None, quality=None):
    """
    Check an LLM explanation for obvious contradictions and unsupported
    numeric claims. This is a guardrail, not a mathematical proof of truth.
    """
    import re

    if not isinstance(text, str) or not text.strip():
        return False, "The model returned an empty explanation."

    lowered = text.casefold()

    # Only reject these claims when the Python results contradict them.
    if dataframe is not None:
        missing_total = int(dataframe.isna().sum().sum())
        duplicate_total = int(dataframe.duplicated().sum())
    else:
        missing_total = None
        duplicate_total = None

    outlier_total = None
    if quality is not None:
        outlier_total = int(quality.get("total_outliers", 0) or 0)

    contradiction_checks = [
        (
            ["no missing values", "there are no missing values",
             "contains no missing values"],
            missing_total is not None and missing_total > 0,
            "The explanation says there are no missing values, but Python detected missing cells."
        ),
        (
            ["no duplicate rows", "there are no duplicates",
             "contains no duplicate rows"],
            duplicate_total is not None and duplicate_total > 0,
            "The explanation says there are no duplicate rows, but Python detected duplicates."
        ),
        (
            ["no outliers", "no numerical outliers",
             "there are no outliers"],
            outlier_total is not None and outlier_total > 0,
            "The explanation says there are no outliers, but Python detected potential IQR outliers."
        ),
        (
            ["all values are accurate", "the data is accurate",
             "the dataset is completely reliable", "guarantees accuracy"],
            True,
            "The explanation makes an unsupported claim about overall data accuracy or reliability."
        ),
        (
            ["remove all outliers", "delete all outliers",
             "automatically remove outliers"],
            True,
            "The explanation recommends automatically deleting outliers."
        )
    ]

    for phrases, contradiction, message in contradiction_checks:
        if contradiction and any(phrase in lowered for phrase in phrases):
            return False, message

    # Check numeric claims against numbers actually supplied to the model.
    # Strip numbered-list prefixes first so headings such as "1." are ignored.
    cleaned_lines = []
    for line in text.splitlines():
        cleaned_lines.append(
            re.sub(r"^\s*(?:[-*]\s*|\d+[.)]\s*)", "", line)
        )
    cleaned_text = "\n".join(cleaned_lines)

    def numeric_tokens(value):
        tokens = re.findall(
            r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?",
            value
        )
        normalized = set()
        for token in tokens:
            token = token.replace(",", "").replace("%", "")
            try:
                normalized.add(float(token))
            except ValueError:
                continue
        return normalized

    if evidence_text:
        allowed_numbers = numeric_tokens(str(evidence_text))
        claimed_numbers = numeric_tokens(cleaned_text)
        unsupported = sorted(
            number for number in claimed_numbers
            if number not in allowed_numbers
        )
        if unsupported:
            examples = ", ".join(
                f"{number:g}" for number in unsupported[:5]
            )
            return (
                False,
                "The explanation includes numeric claims that were not found "
                f"in the Python evidence ({examples}). Use the verified report "
                "for exact figures."
            )

    return True, ""


def answer_dataset_question_with_python(dataframe, quality, question):
    """
    Answer common factual/statistical questions directly from Python.
    Returns None when the question needs qualitative interpretation.
    """
    q = question.lower().strip()
    rows, columns = dataframe.shape

    # Handle summary/overview requests deterministically with Python so the
    # small local LLM cannot misunderstand the dataset profile or refuse.
    summary_terms = [
        "summary", "summarize", "summarise", "summarization",
        "summarisation", "overview", "sum up"
    ]
    if any(term in q for term in summary_terms) or "in one line" in q:
        missing_total = int(dataframe.isna().sum().sum())
        duplicate_total = int(dataframe.duplicated().sum())
        outlier_total = int(quality.get("total_outliers", 0) or 0)
        missing_pct = (
            missing_total / (rows * columns) * 100
            if rows and columns else 0
        )
        return (
            f"This dataset contains **{rows:,} rows** and **{columns:,} columns**, "
            f"with **{missing_total:,} missing cells** ({missing_pct:.4f}% of all "
            f"cells), **{duplicate_total:,} duplicate rows**, and "
            f"**{outlier_total:,} column-level potential IQR outlier flags**. "
            "These checks describe data conditions; they are not a single "
            "measure of overall accuracy."
        )

    if any(word in q for word in ["outlier", "outliers"]):
        total_flags = int(quality.get("total_outliers", 0) or 0)
        report = quality.get("outlier_report")
        parts = [
            "### IQR Outlier Results",
            f"**{total_flags:,} potential outlier flags** were detected "
            "across the numerical columns."
        ]

        if report is not None and not report.empty:
            nonzero = report[report["Outlier Count"] > 0]
            if not nonzero.empty:
                parts.append("**Breakdown by column**")
                for _, row in nonzero.iterrows():
                    parts.append(
                        f"- **{row['Column']}:** {int(row['Outlier Count']):,}"
                    )

            # Count distinct rows using the same eligible columns and IQR
            # bounds as the Data Quality module, so the Q&A agrees with its report.
            flagged_any = pd.Series(False, index=dataframe.index)
            for _, outlier_row in report.iterrows():
                column = outlier_row["Column"]
                if column not in dataframe.columns:
                    continue
                values = pd.to_numeric(dataframe[column], errors="coerce")
                finite_mask = values.notna() & values.isin([float("inf"), float("-inf")]).eq(False)
                lower = outlier_row["Lower Bound"]
                upper = outlier_row["Upper Bound"]
                flagged_any |= (
                    finite_mask & ((values < lower) | (values > upper))
                ).fillna(False)

            unique_rows = int(flagged_any.sum())
            parts.append(
                f"**Distinct rows affected:** {unique_rows:,}. "
                "This counts each row once, even if multiple columns flag it."
            )

        parts.append(
            "**Note:** IQR identifies statistically unusual values. "
            "They are not automatically errors and should be investigated "
            "before any changes are made."
        )
        return "\n\n".join(parts)

    if "missing" in q or "null" in q or "nan" in q:
        missing = dataframe.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        total = int(missing.sum())
        if total == 0:
            return "Python checked the uploaded dataset and found **no missing values**."
        lines = [f"Python detected **{total:,} missing cells**:"]
        for column, count in missing.items():
            pct = int(count) / rows * 100 if rows else 0
            lines.append(f"- **{column}:** {int(count):,} ({pct:.2f}% of rows)")
        return "\n".join(lines)

    if "duplicate" in q:
        count = int(dataframe.duplicated().sum())
        return f"Python detected **{count:,} duplicate rows** in the uploaded dataset."

    if "row" in q and any(word in q for word in ["how many", "number", "count", "total"]):
        return f"The uploaded dataset contains **{rows:,} rows**."

    if "column" in q and any(word in q for word in ["how many", "number", "count", "total"]):
        return f"The uploaded dataset contains **{columns:,} columns**."

    if "quality score" in q or "data quality score" in q:
        checks = quality.get("quality_checks")
        if checks is None or checks.empty:
            return "A single composite quality score is no longer used. No structured quality checks are available."
        return (
            "The application no longer combines different data-quality "
            "dimensions into one score. It reports separate checks instead:\n\n"
            + "\n".join(
                f"- **{row['Check']} — {row['Status']}:** {row['Details']}"
                for _, row in checks.iterrows()
            )
        )

    return None


st.set_page_config(
    page_title="AI EDA Agent",
    page_icon="📊",
    layout="wide"
)

st.title("🤖 AI Agent for Automated EDA")
st.write(
    "Upload any CSV dataset to begin automated exploratory "
    "data analysis and machine learning."
)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        dataframe = load_csv(uploaded_file)

        if dataframe.empty:
            st.warning("The uploaded dataset contains no rows.")
            st.stop()

        profile = profile_dataset(dataframe)
        quality = assess_data_quality(dataframe)

        st.success("Dataset uploaded and profiled successfully!")

        # DATASET PREVIEW
        st.subheader("📋 Complete Dataset Preview")
        total_rows, total_columns = dataframe.shape

        st.write(
            f"Dataset contains **{total_rows:,} rows** "
            f"and **{total_columns:,} columns**."
        )

        preview_rows = st.slider(
            "Select number of rows to preview:",
            min_value=1,
            max_value=int(total_rows),
            value=min(1000, int(total_rows)),
            step=1,
            key="dataset_preview_rows"
        )

        st.caption(
            f"Showing {preview_rows:,} of {total_rows:,} rows "
            f"and all {total_columns:,} columns."
        )
        st.dataframe(
            dataframe.head(preview_rows),
            height=500,
            width="stretch",
            hide_index=True
        )

        csv_data = dataframe.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Complete Dataset",
            data=csv_data,
            file_name="dataset.csv",
            mime="text/csv"
        )

        # AUTOMATED DATA PROFILING
        st.header("🔍 Automated Data Profiling")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Numerical Columns", len(profile["numerical_columns"]))
            st.write(profile["numerical_columns"])

        with col2:
            st.metric("Categorical Columns", len(profile["categorical_columns"]))
            st.write(profile["categorical_columns"])

        st.subheader("Column Information")
        st.dataframe(profile["column_summary"], width="stretch")

        st.subheader("Numerical Statistical Summary")
        if not profile["numerical_summary"].empty:
            st.dataframe(profile["numerical_summary"], width="stretch")
        else:
            st.info("No numerical columns found.")

        st.subheader("Categorical Statistical Summary")
        if not profile["categorical_summary"].empty:
            st.dataframe(profile["categorical_summary"], width="stretch")
        else:
            st.info("No categorical columns found.")

        st.subheader("Datetime Columns")
        if profile["datetime_columns"]:
            st.write(profile["datetime_columns"])
        else:
            st.info("No datetime columns detected.")

        st.subheader("Detected Column Roles")
        if not profile.get("column_roles", pd.DataFrame()).empty:
            st.dataframe(profile["column_roles"], width="stretch")
        else:
            st.info("No column-role information is available.")

        role_col1, role_col2, role_col3 = st.columns(3)
        with role_col1:
            st.metric("Constant Columns", len(profile.get("constant_columns", [])))
        with role_col2:
            st.metric("Near-Constant Columns", len(profile.get("near_constant_columns", [])))
        with role_col3:
            st.metric("Possible Type Mismatches", len(profile.get("type_mismatch_columns", [])))

        if profile.get("constant_columns"):
            st.caption("Constant: " + ", ".join(map(str, profile["constant_columns"])))
        if profile.get("near_constant_columns"):
            st.caption("Near-constant: " + ", ".join(map(str, profile["near_constant_columns"])))
        if profile.get("type_mismatch_columns"):
            st.caption("Possible type mismatches: " + ", ".join(map(str, profile["type_mismatch_columns"])))
        if not profile.get("sentinel_report", pd.DataFrame()).empty:
            st.subheader("Possible Sentinel Values")
            st.dataframe(profile["sentinel_report"], width="stretch")
        st.caption(
            f"Approximate DataFrame memory usage: "
            f"{profile.get('memory_usage_mb', 0):,.2f} MB"
        )

        # DATA QUALITY ASSESSMENT
        st.header("🛡️ Data Quality Assessment")
        st.subheader("Transparent Quality Checks")
        quality_checks = quality.get("quality_checks")
        if quality_checks is not None and not quality_checks.empty:
            st.dataframe(quality_checks, width="stretch", hide_index=True)
        else:
            st.warning("Structured quality-check results are unavailable.")

        st.subheader("Missing Value Analysis")
        st.metric("Total Missing Values", quality["total_missing"])
        st.dataframe(quality["missing_report"], width="stretch")

        st.subheader("Duplicate Record Analysis")
        st.metric("Duplicate Rows", quality["duplicate_count"])
        st.write(f"Duplicate Percentage: {quality['duplicate_percentage']}%")

        st.subheader("Outlier Detection Using IQR")
        st.metric("Total Potential Outliers", quality["total_outliers"])
        if not quality["outlier_report"].empty:
            st.dataframe(quality["outlier_report"], width="stretch")
        else:
            st.info("No numerical columns available for outlier detection.")

        st.subheader("Data Quality Recommendations")
        for recommendation in quality["recommendations"]:
            st.write("📌", recommendation)

        # AUTOMATED DATA VISUALIZATION
        st.header("📊 Automated Data Visualization")
        numerical_columns = dataframe.select_dtypes(
            include=["number"]
        ).columns.tolist()

        if numerical_columns:
            selected_column = st.selectbox(
                "Select a numerical column:",
                numerical_columns,
                key="histogram_column"
            )

            st.subheader("📈 Distribution Histogram")
            histogram = plot_histogram(dataframe, selected_column)
            if histogram is not None:
                st.plotly_chart(histogram, use_container_width=True)
            else:
                st.info("No valid numerical values are available for this histogram.")

            st.subheader("📦 Box Plot Analysis")
            boxplot_columns = [
                column for column in numerical_columns
                if str(column).lower() not in ["id", "unnamed: 0", "index"]
            ]

            if boxplot_columns:
                display_mode = st.radio(
                    "Choose Box Plot Display Mode:",
                    ["Single Feature", "All Numerical Features"],
                    horizontal=True,
                    key="boxplot_display_mode"
                )

                if display_mode == "Single Feature":
                    selected_box_column = st.selectbox(
                        "Select a numerical feature:",
                        boxplot_columns,
                        key="single_boxplot_column"
                    )
                    boxplot = plot_boxplot(dataframe, selected_box_column)

                    if boxplot is not None:
                        st.plotly_chart(boxplot, use_container_width=True)
                        st.subheader("🔍 Understanding the Box Plot")
                        interpretation = interpret_boxplot(
                            dataframe, selected_box_column
                        )
                        for point in interpretation:
                            st.write("📌", point)
                    else:
                        st.info("No valid values available for this feature.")

                else:
                    st.subheader("📊 Box Plots for All Numerical Features")
                    st.caption(
                        f"Displaying {len(boxplot_columns)} numerical features."
                    )
                    for column in boxplot_columns:
                        st.markdown(f"### {column}")
                        boxplot = plot_boxplot(dataframe, column)

                        if boxplot is not None:
                            st.plotly_chart(boxplot, use_container_width=True)
                            interpretation = interpret_boxplot(dataframe, column)
                            with st.expander(f"Interpretation of {column}"):
                                for point in interpretation:
                                    st.write("📌", point)
                        else:
                            st.info(f"No valid values available for {column}.")
            else:
                st.info(
                    "No suitable numerical features are available "
                    "for box plot analysis."
                )
        else:
            st.info("No numerical columns are available for visualization.")

        # CORRELATION HEATMAP
        st.subheader("🔥 Correlation Heatmap")
        correlation_method = st.radio(
            "Correlation method:",
            ["pearson", "spearman"],
            horizontal=True,
            key="correlation_method"
        )
        heatmap = plot_correlation_heatmap(
            dataframe, method=correlation_method
        )

        if heatmap is not None:
            st.plotly_chart(
                heatmap,
                use_container_width=True,
                config={"responsive": True}
            )
            st.subheader("🔍 Understanding the Correlation Heatmap")
            heatmap_interpretation = interpret_correlation_heatmap(
                dataframe, method=correlation_method
            )
            for point in heatmap_interpretation:
                st.write("📌", point)
        else:
            st.info(
                "At least two numerical columns with varying values "
                "are required for a correlation heatmap."
            )

        # SCATTER PLOT
        st.subheader("🔵 Interactive Scatter Plot")
        numerical_columns = dataframe.select_dtypes(
            include=["number"]
        ).columns.tolist()

        if len(numerical_columns) >= 2:
            col1, col2 = st.columns(2)

            with col1:
                x_column = st.selectbox(
                    "Select X-axis feature:",
                    numerical_columns,
                    key="scatter_x"
                )

            with col2:
                y_options = [
                    column for column in numerical_columns
                    if column != x_column
                ]
                y_column = st.selectbox(
                    "Select Y-axis feature:",
                    y_options,
                    key="scatter_y"
                )

            scatter_fig = plot_scatter(dataframe, x_column, y_column)
            if scatter_fig is not None:
                st.plotly_chart(scatter_fig, use_container_width=True)
                st.subheader("🔍 Understanding the Scatter Plot")
                scatter_interpretation = interpret_scatter_plot(
                    dataframe, x_column, y_column
                )
                for point in scatter_interpretation:
                    st.write("📌", point)
            else:
                st.info("No valid data available for this scatter plot.")
        else:
            st.info(
                "At least two numerical columns are required "
                "to generate a scatter plot."
            )

        # INTERACTIVE BAR CHART
        st.subheader("📊 Interactive Bar Chart")
        categorical_columns = dataframe.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()

        if categorical_columns:
            selected_category = st.selectbox(
                "Select a categorical feature:",
                categorical_columns,
                key="bar_chart_column"
            )
            bar_fig = plot_bar_chart(dataframe, selected_category)
            if bar_fig is not None:
                st.plotly_chart(bar_fig, use_container_width=True)
            else:
                st.info("No valid category data available.")
        else:
            st.info("No categorical columns are available for the bar chart.")

        # DATA CLEANING STUDIO
        st.divider()
        st.header("🧹 Data Cleaning Studio")
        st.write("Prepare your dataset for analysis and machine learning.")
        st.info(
            "Your original uploaded dataset will remain unchanged. "
            "All operations are performed on a separate copy."
        )
        st.warning(
            "Cleaning Studio is for exploratory cleaning and export. "
            "Do not use its full-dataset scaling or encoding as a substitute "
            "for ML preprocessing: fit imputers, encoders and scalers on the "
            "training split only, inside a Pipeline."
        )

        st.subheader("1. Select Cleaning Operations")

        apply_missing = st.checkbox(
            "Handle missing values", value=True, key="clean_missing"
        )
        apply_duplicates = st.checkbox(
            "Remove duplicate rows", key="clean_duplicates"
        )
        apply_datetime = st.checkbox(
            "Extract date and time features", key="clean_datetime"
        )
        apply_encoding = st.checkbox(
            "Encode categorical columns", key="clean_encoding"
        )
        apply_scaling = st.checkbox(
            "Scale numerical columns", key="clean_scaling"
        )

        if apply_missing:
            missing_strategy = st.selectbox(
                "Missing value strategy:",
                ["auto", "mean", "median", "mode", "drop"],
                key="missing_strategy"
            )

        if apply_duplicates:
            duplicate_strategy = st.selectbox(
                "Which duplicate occurrence should be kept?",
                ["first", "last", "remove all"],
                key="duplicate_strategy"
            )

        if apply_datetime:
            date_columns = st.multiselect(
                "Select date/time columns:",
                dataframe.columns.tolist(),
                key="date_columns"
            )

        if apply_encoding:
            encoding_method = st.selectbox(
                "Categorical encoding method:",
                ["onehot", "label"],
                key="encoding_method"
            )

        if apply_scaling:
            scaling_method = st.selectbox(
                "Numerical scaling method:",
                ["standard", "minmax"],
                key="scaling_method"
            )

        st.divider()

        if st.button(
                "🚀 Process Dataset",
                type="primary",
                key="process_cleaning"
        ):
            cleaned_data = dataframe.copy()
            cleaning_report = []


            def add_report(operation, status, details, rows_affected=0):
                cleaning_report.append({
                    "Operation": operation,
                    "Status": status,
                    "Details": details,
                    "Rows Affected": rows_affected
                })


            try:
                # 1. MISSING VALUES
                if apply_missing:
                    missing_before = int(cleaned_data.isna().sum().sum())
                    rows_before = len(cleaned_data)

                    cleaned_data = handle_missing_values(
                        cleaned_data,
                        strategy=missing_strategy
                    )

                    missing_after = int(cleaned_data.isna().sum().sum())
                    rows_removed = rows_before - len(cleaned_data)

                    if missing_before == 0:
                        status = "No action needed"
                        details = "No missing values were found."
                    else:
                        status = (
                            "Completed" if missing_after == 0
                            else "Partially completed"
                        )
                        details = (
                            (
                                f"Found {missing_before:,} missing values; "
                                f"{'removed ' + format(rows_removed, ',') + ' rows' if missing_strategy == 'drop' else 'filled ' + format(missing_before - missing_after, ',') + ' values'}; "
                                f"{missing_after:,} missing values remain."
                            )
                        )

                    add_report(
                        "Missing Value Handling", status, details, rows_removed
                    )
                else:
                    add_report(
                        "Missing Value Handling", "Skipped",
                        "This operation was not selected."
                    )

                # 2. DUPLICATES
                if apply_duplicates:
                    rows_before = len(cleaned_data)
                    duplicate_count = int(cleaned_data.duplicated().sum())

                    keep_value = (
                        False if duplicate_strategy == "remove all"
                        else duplicate_strategy
                    )

                    cleaned_data, detected_duplicates = handle_duplicates(
                        cleaned_data, keep=keep_value
                    )

                    rows_removed = rows_before - len(cleaned_data)
                    status = (
                        "Completed" if duplicate_count > 0
                        else "No action needed"
                    )

                    add_report(
                        "Duplicate Removal",
                        status,
                        (
                            (
                                f"Found {duplicate_count:,} duplicate rows; "
                                f"removed {rows_removed:,} rows."
                            )
                        ),
                        rows_removed
                    )
                else:
                    add_report(
                        "Duplicate Removal", "Skipped",
                        "This operation was not selected."
                    )

                # 3. DATETIME FEATURES
                if apply_datetime:
                    if date_columns:
                        columns_before = set(cleaned_data.columns)
                        cleaned_data = extract_datetime_features(
                            cleaned_data, columns=date_columns
                        )

                        new_features = [
                            column for column in cleaned_data.columns
                            if column not in columns_before
                        ]

                        if new_features:
                            add_report(
                                "Date and Time Extraction",
                                "Completed",
                                (
                                    f"Created {len(new_features)} date/time features "
                                    f"from {len(date_columns)} selected columns."

                                )
                            )
                        else:
                            add_report(
                                "Date and Time Extraction",
                                "No features created",
                                (
                                    "No valid date values were found in the "
                                    "selected columns, or the features already "
                                    "existed."
                                )
                            )
                    else:
                        add_report(
                            "Date and Time Extraction", "Skipped",
                            "No date/time columns were selected."
                        )
                else:
                    add_report(
                        "Date and Time Extraction", "Skipped",
                        "This operation was not selected."
                    )

                # 4. CATEGORICAL ENCODING
                if apply_encoding:
                    categorical_before = cleaned_data.select_dtypes(
                        include=["object", "category", "bool", "string"]
                    ).columns.tolist()

                    columns_before = set(cleaned_data.columns)

                    if categorical_before:
                        cleaned_data = encode_categorical_columns(
                            cleaned_data, method=encoding_method
                        )

                        new_columns = [
                            column for column in cleaned_data.columns
                            if column not in columns_before
                        ]

                        remaining_categorical = cleaned_data.select_dtypes(
                            include=["object", "category", "bool", "string"]
                        ).columns.tolist()

                        status = (
                            "Completed" if not remaining_categorical
                            else "Partially completed"
                        )

                        add_report(
                            "Categorical Encoding",
                            status,
                            (
                                f"Applied {encoding_method} encoding to "
                                f"{len(categorical_before)} categorical columns; "
                                f"created {len(new_columns)} new columns."

                            )
                        )
                    else:
                        add_report(
                            "Categorical Encoding", "No action needed",
                            "No categorical columns were found."
                        )
                else:
                    add_report(
                        "Categorical Encoding", "Skipped",
                        "This operation was not selected."
                    )

                # 5. NUMERICAL SCALING
                if apply_scaling:
                    numerical_before = cleaned_data.select_dtypes(
                        include=["number"]
                    ).columns.tolist()

                    if numerical_before:
                        cleaned_data = scale_numerical_columns(
                            cleaned_data, method=scaling_method
                        )

                        add_report(
                            "Numerical Scaling",
                            "Completed",
                            (
                                f"Applied {scaling_method} scaling to "
                                f"{len(numerical_before)} numerical columns."

                            )
                        )
                    else:
                        add_report(
                            "Numerical Scaling", "No action needed",
                            "No numerical columns were found."
                        )
                else:
                    add_report(
                        "Numerical Scaling", "Skipped",
                        "This operation was not selected."
                    )

                # COMPACT CLEANING ACTIVITY REPORT
                st.success("Dataset processing finished!")
                st.subheader("Cleaning Activity")

                # Show only operations the user selected.
                visible_report = [
                    item for item in cleaning_report
                    if item["Status"] != "Skipped"
                ]

                if visible_report:
                    for item in visible_report:
                        if item["Status"] == "Completed":
                            icon = "✅"
                        elif item["Status"] in [
                            "Partially completed", "No features created"
                        ]:
                            icon = "⚠️"
                        else:
                            icon = "ℹ️"

                        st.markdown(
                            f"{icon} **{item['Operation']}:** "
                            f"{item['Details']}"
                        )
                else:
                    st.info("No cleaning operations were selected.")

                # CLEANING RESULTS
                st.divider()
                st.subheader("2. Cleaning Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Rows", f"{dataframe.shape[0]:,}")
                with col2:
                    st.metric("Processed Rows", f"{cleaned_data.shape[0]:,}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Original Columns", f"{dataframe.shape[1]:,}"
                    )
                with col2:
                    st.metric(
                        "Processed Columns", f"{cleaned_data.shape[1]:,}"
                    )

                st.subheader("Cleaned Dataset Preview")
                st.dataframe(
                    cleaned_data,
                    height=500,
                    width="stretch",
                    hide_index=False
                )

                cleaned_csv = cleaned_data.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Cleaned Dataset",
                    data=cleaned_csv,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    key="download_cleaned_dataset"
                )

            except Exception as cleaning_error:
                st.error(
                    f"Data cleaning failed: {cleaning_error}"
                )

        # Rule-based insights are prepared as private context for optional
        # AI features, but are not displayed automatically.
        dataset_insights = generate_dataset_insights(dataframe)

        # ON-DEMAND DATASET REPORT + AI EXPLANATION
        # Before the user clicks, show only this button. The report, AI
        # explanation, and dataset Q&A are all hidden.
        # Hash uploaded bytes so different files with the same filename,
        # shape and columns cannot reuse stale session state.
        dataset_bytes = uploaded_file.getvalue()
        safe_signature = hashlib.sha256(dataset_bytes).hexdigest()[:20]

        report_requested_key = f"dataset_report_requested_v5_{safe_signature}"
        summary_state_key = f"explicit_ai_summary_v5_{safe_signature}"
        answer_state_key = f"explicit_ai_answer_v5_{safe_signature}"

        if report_requested_key not in st.session_state:
            st.session_state[report_requested_key] = False

        st.subheader("🤖 AI-Powered Dataset Analysis")

        generate_clicked = st.button(
            "✨ Generate AI Explanation",
            key=f"generate_ai_explanation_v5_{safe_signature}",
            use_container_width=True
        )

        if generate_clicked:
            st.session_state[report_requested_key] = True
            st.session_state.pop(summary_state_key, None)
            st.session_state.pop(answer_state_key, None)

            dataset_information = build_llm_dataset_information(
                dataframe,
                profile,
                quality,
                dataset_insights
            )

            with st.spinner("Generating your dataset report and AI explanation..."):
                summary_result = generate_dataset_summary(dataset_information)
                is_valid, validation_message = validate_ai_explanation(
                    summary_result,
                    evidence_text=dataset_information,
                    dataframe=dataframe,
                    quality=quality
                )

                if is_valid:
                    st.session_state[summary_state_key] = {
                        "status": "valid",
                        "text": summary_result
                    }
                else:
                    st.session_state[summary_state_key] = {
                        "status": "fallback",
                        "text": (
                            "The local model's explanation did not pass the "
                            "accuracy checks. The verified Python report below "
                            "is still available."
                        ),
                        "reason": validation_message
                    }

        # Nothing in this block is rendered until the user clicks the button.
        if st.session_state.get(report_requested_key, False):
            dataset_information = build_llm_dataset_information(
                dataframe,
                profile,
                quality,
                dataset_insights
            )

            st.divider()
            st.subheader("📋 Verified Dataset Analysis Report")
            st.caption(
                "Generated by Python from the uploaded dataset. Exact "
                "statistics and recommendations do not come from the LLM."
            )
            st.markdown(build_verified_dataset_report(dataframe, quality))

            saved_summary = st.session_state.get(summary_state_key)
            if saved_summary:
                st.subheader("🧠 AI Explanation")
                st.markdown(saved_summary["text"])
                if saved_summary["status"] == "fallback":
                    st.info(
                        saved_summary.get(
                            "reason",
                            "The AI explanation did not pass validation."
                        )
                    )

            st.subheader("💬 Ask a Question About This Dataset")
            with st.form(key=f"dataset_question_form_v5_{safe_signature}"):
                user_question = st.text_input(
                    "Enter your question",
                    placeholder="Example: Which columns have missing values?"
                )
                submit_question = st.form_submit_button("Ask AI Analyst")

            if submit_question:
                if not user_question.strip():
                    st.warning("Please enter a question first.")
                else:
                    with st.spinner("Analyzing your question against the uploaded dataset..."):
                        # Answer exact statistical questions with Python, not
                        # the small local LLM. This avoids rejecting correct
                        # numerical answers or allowing the model to guess.
                        answer_result = answer_dataset_question_with_python(
                            dataframe,
                            quality,
                            user_question.strip()
                        )

                        if answer_result is not None:
                            answer_source = "python"
                            answer_validation_message = ""
                        else:
                            answer_result = ask_dataset_question(
                                dataset_information,
                                user_question.strip()
                            )
                            # Do not apply the summary validator to Q&A:
                            # valid answers often need numerical values.
                            # The model is grounded in the supplied dataset
                            # profile and can answer in natural language.
                            answer_is_valid, answer_validation_message = (
                                validate_ai_explanation(
                                    answer_result,
                                    evidence_text=dataset_information,
                                    dataframe=dataframe,
                                    quality=quality
                                )
                            )
                            if answer_is_valid:
                                answer_source = "ai"
                            else:
                                answer_source = "ai_validation_fallback"
                                answer_result = (
                                    "The AI response did not pass the evidence "
                                    "checks. Please use the Python-verified "
                                    "report for exact facts.\n\n"
                                    + answer_validation_message
                                )

                        st.session_state[answer_state_key] = {
                            "question": user_question.strip(),
                            "answer": answer_result,
                            "source": answer_source,
                            "validation_message": answer_validation_message
                        }

            saved_answer = st.session_state.get(answer_state_key)
            if saved_answer:
                st.markdown(f"**Your question:** {saved_answer['question']}")
                source_label = (
                    "Python-verified answer"
                    if saved_answer.get("source") == "python"
                    else (
                        "AI answer (not accepted by evidence checks)"
                        if saved_answer.get("source") == "ai_validation_fallback"
                        else "AI Analyst"
                    )
                )
                st.markdown(f"**{source_label}:**")
                st.markdown(saved_answer["answer"])
                if saved_answer.get("validation_message"):
                    st.info(saved_answer["validation_message"])

        # MACHINE LEARNING STUDIO
        # ML dashboard intentionally receives the original uploaded data.
        # Each model must perform train/test-safe preprocessing internally.
        render_ml_dashboard(dataframe)

    except Exception as error:
        st.error(f"An error occurred: {error}")
