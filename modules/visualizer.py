"""
Plotly visualizations and deterministic chart interpretations for EDA.
Public function names and default signatures are preserved for app.py.
"""
import numpy as np
import pandas as pd
import plotly.express as px

MAX_SCATTER_POINTS = 10_000
MAX_CATEGORIES = 30


def _is_identifier_column(dataframe, column):
    name = str(column).strip().lower()
    compact = name.replace(" ", "").replace("-", "").replace("_", "")
    if compact in {"id", "index", "rowid", "recordid"} or compact.endswith("id"):
        return True
    # Do not classify a numeric feature as an ID from uniqueness alone:
    # continuous measurements can legitimately have nearly all unique values.
    return False


def _finite_numeric_series(dataframe, column):
    return pd.to_numeric(dataframe[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()


def _correlation_data(dataframe):
    columns = dataframe.select_dtypes(include=[np.number]).columns
    eligible = [
        col for col in columns
        if not pd.api.types.is_bool_dtype(dataframe[col])
        and not _is_identifier_column(dataframe, col)
        and dataframe[col].nunique(dropna=True) > 1
    ]
    return dataframe[eligible].replace([np.inf, -np.inf], np.nan)


def plot_histogram(dataframe, column):
    """Generate an interactive histogram for a numerical column."""
    if column not in dataframe.columns:
        return None
    values = _finite_numeric_series(dataframe, column)
    if values.empty:
        return None
    fig = px.histogram(
        x=values, title=f"Distribution of {column}", nbins=30,
        color_discrete_sequence=["#00C8FF"], template="plotly_dark"
    )
    fig.update_traces(
        marker_line_color="#FFFFFF", marker_line_width=0.7,
        hovertemplate=(
            f"<b>{column} Distribution</b><br>Value: %{{x}}<br>"
            "Number of records: %{y}<extra></extra>"
        )
    )
    fig.update_layout(
        height=380, title_x=0.5, bargap=0.08,
        xaxis_title=str(column), yaxis_title="Number of records",
        showlegend=False
    )
    return fig


def plot_boxplot(dataframe, column):
    """Generate a vertical box plot; return None if no valid numeric data."""
    if column not in dataframe.columns:
        return None
    values = _finite_numeric_series(dataframe, column)
    if values.empty:
        return None

    plot_column = str(column)
    clean_data = pd.DataFrame({plot_column: values})
    q1, median, q3 = values.quantile(.25), values.median(), values.quantile(.75)
    minimum, maximum, mean = values.min(), values.max(), values.mean()

    fig = px.box(
        clean_data, y=plot_column, orientation="v",
        title=f"Box Plot of {column}",
        color_discrete_sequence=["#00C8FF"], template="plotly_dark",
        points=False
    )
    fig.update_traces(
        width=.30, line=dict(color="#00C8FF", width=2),
        fillcolor="rgba(0, 200, 255, 0.25)", boxmean=False,
        hovertemplate=f"<b>{column}</b><br>Value: %{{y:,.2f}}<extra></extra>"
    )
    fig.update_layout(
        height=480,
        title=dict(text=f"Box Plot of {column}", x=.5,
                   font=dict(size=20, color="white")),
        showlegend=False, hovermode="closest",
        hoverlabel=dict(bgcolor="#161B22", bordercolor="#00C8FF",
                        font=dict(color="white", size=14, family="Arial")),
        xaxis=dict(title="", showticklabels=False, zeroline=False),
        yaxis=dict(title=dict(text=plot_column, font=dict(color="white")),
                   gridcolor="rgba(255,255,255,0.12)", zeroline=False),
        margin=dict(l=65, r=35, t=75, b=125)
    )
    fig.add_annotation(
        x=.5, y=-.27, xref="paper", yref="paper",
        text=(
            f"<b>MIN:</b> {minimum:,.2f} &nbsp; | &nbsp; "
            f"<b>Q1:</b> {q1:,.2f} &nbsp; | &nbsp; "
            f"<b>MEDIAN:</b> {median:,.2f} &nbsp; | &nbsp; "
            f"<b>Q3:</b> {q3:,.2f} &nbsp; | &nbsp; "
            f"<b>MAX:</b> {maximum:,.2f} &nbsp; | &nbsp; "
            f"<b>MEAN:</b> {mean:,.2f}"
        ),
        showarrow=False, xanchor="center", align="center",
        font=dict(color="#00C8FF", size=11)
    )
    return fig


def interpret_boxplot(dataframe, column):
    """Return a beginner-friendly statistical summary for a box plot."""
    if column not in dataframe.columns:
        return ["The selected column is not available."]
    values = _finite_numeric_series(dataframe, column)
    if values.empty:
        return ["No valid numerical values available."]

    q1, q3 = values.quantile(.25), values.quantile(.75)
    median, minimum, maximum, mean = (
        values.median(), values.min(), values.max(), values.mean()
    )
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = values[(values < lower) | (values > upper)]
    skewness = values.skew()

    result = [
        f"Valid observations: {len(values):,}.",
        f"Minimum: {minimum:,.2f}.",
        f"Q1: {q1:,.2f} — 25% of observations are at or below this value.",
        f"Median: {median:,.2f} — the middle value.",
        f"Q3: {q3:,.2f} — 75% of observations are at or below this value.",
        f"Maximum: {maximum:,.2f}.",
        f"Mean: {mean:,.2f}.",
        f"IQR: {iqr:,.2f} — the middle 50% spread.",
        (f"Skewness: {skewness:.3f}." if pd.notna(skewness)
         else "Skewness could not be estimated."),
        f"Lower IQR limit: {lower:,.2f}.",
        f"Upper IQR limit: {upper:,.2f}.",
    ]
    if outliers.empty:
        result.append("No potential outliers were detected using the 1.5 × IQR rule.")
    else:
        result.append(
            f"{len(outliers):,} potential outlier(s) detected "
            f"({len(outliers) / len(values) * 100:.2f}% of valid values)."
        )
        result.append(f"First 20 flagged values: {outliers.head(20).tolist()}.")
    if pd.notna(skewness) and abs(skewness) > 1:
        result.append(
            "The distribution is strongly skewed. IQR flags may include "
            "natural tail observations; inspect them in context."
        )
    result.append(
        "An IQR flag indicates an unusual value, not necessarily an error. "
        "Investigate before deciding whether to remove it."
    )
    return result


def plot_correlation_heatmap(dataframe, method="pearson"):
    """Generate a numeric Pearson or Spearman correlation heatmap."""
    method = str(method).lower().strip()
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    numeric = _correlation_data(dataframe)
    if numeric.shape[1] < 2:
        return None
    matrix = numeric.corr(method=method, min_periods=3)
    n_features = len(matrix.columns)
    fig = px.imshow(
        matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale=["#B2182B", "#F7F7F7", "#00C8FF"],
        zmin=-1, zmax=1,
        title=f"{method.title()} Correlation Between Numerical Features",
        template="plotly_dark"
    )
    fig.update_layout(
        autosize=True, height=max(600, n_features * 45),
        title=dict(
            text=f"{method.title()} Correlation Between Numerical Features",
            x=.5, xanchor="center", font=dict(size=21, color="white")
        ),
        xaxis=dict(title="Feature 1 (Columns)", automargin=True, tickangle=-45),
        yaxis=dict(title="Feature 2 (Rows)", automargin=True, autorange="reversed"),
        coloraxis_colorbar=dict(title="Correlation", len=.85),
        margin=dict(l=120, r=100, t=100, b=150)
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Feature Relationship</b><br>Feature 1: %{x}<br>"
            "Feature 2: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
        ),
        xgap=1, ygap=1
    )
    return fig


def interpret_correlation_heatmap(dataframe, method="pearson"):
    """Describe strongest positive and negative pairwise correlations."""
    method = str(method).lower().strip()
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    numeric = _correlation_data(dataframe)
    if numeric.shape[1] < 2:
        return ["At least two varying eligible numerical features are required."]

    matrix = numeric.corr(method=method, min_periods=3)
    pairs = []
    cols = matrix.columns.tolist()
    for i, first in enumerate(cols):
        for second in cols[i + 1:]:
            value = matrix.loc[first, second]
            if pd.notna(value):
                n = int(numeric[[first, second]].dropna().shape[0])
                pairs.append((float(value), first, second, n))
    if not pairs:
        return ["No valid correlations could be calculated."]

    positive = max(pairs, key=lambda item: item[0])
    negative = min(pairs, key=lambda item: item[0])
    result = [
        f"Correlation method: {method.title()}.",
        (f"Strongest positive pair: {positive[1]} and {positive[2]} "
         f"({positive[0]:.3f}; n={positive[3]:,})."),
        (f"Strongest negative pair: {negative[1]} and {negative[2]} "
         f"({negative[0]:.3f}; n={negative[3]:,})."),
        "Correlation describes association, not causation. Pairwise sample "
        "sizes can differ when values are missing."
    ]
    if max(abs(positive[0]), abs(negative[0])) < .30:
        result.append(
            "The strongest absolute correlation is below 0.30; detected "
            "linear or monotonic associations are weak."
        )
    return result


def plot_scatter(dataframe, x_column, y_column, max_points=MAX_SCATTER_POINTS,
                 random_state=42):
    """Generate a scatter plot, deterministically sampling large datasets."""
    if x_column not in dataframe.columns or y_column not in dataframe.columns:
        return None
    clean = dataframe[[x_column, y_column]].copy()
    for col in (x_column, y_column):
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None

    total = len(clean)
    if max_points is not None and total > max_points:
        clean = clean.sample(n=max_points, random_state=random_state)
    subtitle = (
        f"Showing {len(clean):,} of {total:,} valid observations"
        if len(clean) < total else f"{total:,} valid observations"
    )
    fig = px.scatter(
        clean, x=x_column, y=y_column, title=f"{y_column} vs {x_column}",
        template="plotly_dark", opacity=.75,
        color_discrete_sequence=["#00C8FF"]
    )
    fig.update_traces(
        marker=dict(size=7),
        hovertemplate=(
            f"<b>{x_column}:</b> %{{x}}<br>"
            f"<b>{y_column}:</b> %{{y}}<extra></extra>"
        )
    )
    fig.update_layout(
        height=500,
        title=dict(text=f"{y_column} vs {x_column}<br><sup>{subtitle}</sup>",
                   x=.5, xanchor="center"),
        xaxis_title=str(x_column), yaxis_title=str(y_column),
        hoverlabel=dict(bgcolor="#161B22", bordercolor="#00C8FF",
                        font=dict(color="white", size=13)),
        margin=dict(l=80, r=50, t=100, b=80)
    )
    return fig


def interpret_scatter_plot(dataframe, x_column, y_column):
    """Interpret Pearson and Spearman association for a selected pair."""
    if x_column not in dataframe.columns or y_column not in dataframe.columns:
        return ["One or both selected columns are not available."]
    clean = dataframe[[x_column, y_column]].copy()
    for col in (x_column, y_column):
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return ["At least three valid paired observations are needed."]

    pearson = clean[x_column].corr(clean[y_column], method="pearson")
    spearman = clean[x_column].corr(clean[y_column], method="spearman")
    if pd.isna(pearson):
        return ["Pearson correlation cannot be calculated; one or both columns may be constant."]

    strength = (
        "very weak" if abs(pearson) < .20 else
        "weak" if abs(pearson) < .40 else
        "moderate" if abs(pearson) < .60 else
        "strong" if abs(pearson) < .80 else "very strong"
    )
    direction = "positive" if pearson > 0 else "negative" if pearson < 0 else "no linear"
    result = [
        f"Pearson correlation (linear association): {pearson:.3f}.",
        (f"Spearman correlation (rank-based association): {spearman:.3f}."
         if pd.notna(spearman) else "Spearman correlation could not be calculated."),
        f"Pearson relationship: {strength} {direction}.",
        f"Number of valid paired observations: {len(clean):,}."
    ]
    if pd.notna(spearman) and abs(spearman - pearson) >= .20:
        result.append(
            "Pearson and Spearman differ noticeably. The relationship may "
            "not be well described by a straight line or may be influenced by outliers."
        )
    result.append(
        "Correlation measures association, not causation. Inspect the scatter "
        "plot for non-linear patterns and outliers."
    )
    return result


def plot_bar_chart(dataframe, column, top_n=MAX_CATEGORIES):
    """Plot category frequencies; group remaining categories when numerous."""
    if column not in dataframe.columns:
        return None
    values = dataframe[column].dropna()
    if values.empty:
        return None
    counts = values.astype(str).value_counts()
    if top_n is not None and top_n > 0 and len(counts) > top_n:
        remainder = int(counts.iloc[top_n:].sum())
        counts = counts.head(top_n).copy()
        counts.loc["Other (remaining categories)"] = remainder

    chart_data = counts.rename_axis(str(column)).reset_index(name="Count")
    fig = px.bar(
        chart_data, x=str(column), y="Count",
        title=f"Distribution of {column}", template="plotly_dark",
        color="Count", color_continuous_scale=["#164E63", "#00C8FF"],
        text="Count"
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            f"<b>{column}:</b> %{{x}}<br>"
            "<b>Number of records:</b> %{y:,}<extra></extra>"
        )
    )
    fig.update_layout(
        height=500, title=dict(text=f"Distribution of {column}", x=.5,
                               xanchor="center"),
        xaxis_title=str(column), yaxis_title="Number of records",
        showlegend=False, coloraxis_showscale=False,
        margin=dict(l=70, r=50, t=90, b=100)
    )
    return fig
