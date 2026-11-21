from modules.langchain_engine import EDALangChainEngine
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import pandas as pd
import numpy as np
import io
import json

from sklearn.model_selection import train_test_split

from modules.classification import train_classification_models
from modules.regression import train_regression_models

from modules.data_profiler import profile_dataset
from modules.data_quality import assess_data_quality
from modules.insights import generate_dataset_insights

from modules.visualizer import (
    plot_histogram,
    plot_boxplot,
    interpret_boxplot,
    plot_correlation_heatmap,
    interpret_correlation_heatmap,
    plot_scatter,
    interpret_scatter_plot,
    plot_bar_chart
)

from modules.ml_preprocessor import build_preprocessor
from modules.llm_engine import generate_response
from modules.rag_dataset_builder import index_dataset_insights
from modules.rag_engine import EDARetrievalEngine
from modules.rag_qa import answer_dataset_question

from modules.preprocessor import (
    handle_missing_values,
    handle_duplicates,
    encode_categorical_columns,
    scale_numerical_columns,
    extract_datetime_features
)


app = FastAPI(
    title="EDA AI Agent API",
    description="API for automated exploratory data analysis",
    version="1.0.0"
)


# --------------------------------------------------
# LANGCHAIN REQUEST MODEL
# --------------------------------------------------

class LangChainQuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=20)



# --------------------------------------------------
# GLOBAL DATASET STORAGE
# --------------------------------------------------

uploaded_dataset = None
uploaded_filename = None

processed_dataset = None

# Stores the latest leakage-safe ML preprocessing result in memory.
ml_preprocessing_result = None

# Latest model-training results are kept in memory for status checks.
classification_training_result = None
regression_training_result = None

# The active ChromaDB collection for the currently uploaded dataset.
rag_collection_name = None

# Cached LangChain engine for the active RAG collection.
langchain_engine = None


# --------------------------------------------------
# JSON CONVERSION HELPER
# --------------------------------------------------

def make_json_safe(value):

    if isinstance(value, pd.DataFrame):
        return make_json_safe(value.to_dict(orient="records"))

    if isinstance(value, pd.Series):
        return make_json_safe(value.to_dict())

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if value is None:
        return None

    if isinstance(value, float) and not np.isfinite(value):
        return None

    if not isinstance(value, (dict, list, tuple)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

    return value


# --------------------------------------------------
# DATASET VALIDATION
# --------------------------------------------------

def check_dataset():

    if uploaded_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload a dataset first using /upload."
        )



def is_identifier_feature(column_name):
    """Identify obvious row/index and record-ID columns for ML exclusion."""
    name = str(column_name).strip().lower()

    if name.startswith("unnamed:"):
        return True

    if name in {"id", "index", "row_id", "record_id"}:
        return True

    if name.endswith("_id") or name.endswith(" id"):
        return True

    return False


def get_preprocessing_source(source):

    if source not in {"original", "processed"}:
        raise HTTPException(
            status_code=400,
            detail="Source must be 'original' or 'processed'."
        )

    check_dataset()

    if source == "original":
        return uploaded_dataset

    if processed_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No processed dataset exists. Run a preprocessing operation first."
        )

    return processed_dataset


def save_processed_dataset(dataframe, operation, source, details=None):

    global processed_dataset

    processed_dataset = dataframe.copy(deep=True)

    return {
        "message": "Preprocessing completed successfully.",
        "filename": uploaded_filename,
        "operation": operation,
        "source": source,
        "rows": int(processed_dataset.shape[0]),
        "columns": int(processed_dataset.shape[1]),
        "column_names": processed_dataset.columns.tolist(),
        "preview": make_json_safe(
            processed_dataset.head(10).to_dict(orient="records")
        ),
        "details": details or {},
        "download_endpoint": "/preprocessing/download"
    }


# --------------------------------------------------
# PLOTLY FIGURE CONVERSION
# --------------------------------------------------

def convert_figure_to_json(figure):

    if figure is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to generate visualization. Check the selected columns and their data."
        )

    return json.loads(figure.to_json())


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to EDA AI Agent API",
        "status": "running"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):

    global uploaded_dataset
    global uploaded_filename
    global processed_dataset
    global ml_preprocessing_result
    global classification_training_result
    global regression_training_result
    global rag_collection_name
    global langchain_engine

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    try:
        contents = await file.read()

        dataframe = pd.read_csv(io.BytesIO(contents))

        if dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty."
            )

        if dataframe.columns.duplicated().any():
            raise HTTPException(
                status_code=400,
                detail="The CSV contains duplicate column names."
            )

        uploaded_dataset = dataframe
        uploaded_filename = file.filename

        # Reset previous preprocessing results
        processed_dataset = None
        ml_preprocessing_result = None
        classification_training_result = None
        regression_training_result = None
        rag_collection_name = None
        langchain_engine = None

        return {
            "message": "Dataset uploaded successfully",
            "filename": uploaded_filename,
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "column_names": dataframe.columns.tolist(),
            "preview": make_json_safe(
                dataframe.head().to_dict(orient="records")
            )
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading CSV file: {str(error)}"
        )


# --------------------------------------------------
# DATASET PROFILE
# --------------------------------------------------

@app.get("/profile")
def get_dataset_profile():

    check_dataset()

    try:
        profile = profile_dataset(uploaded_dataset)

        return make_json_safe({

            "filename": uploaded_filename,

            "overview": {
                "rows": uploaded_dataset.shape[0],
                "columns": uploaded_dataset.shape[1],
                "memory_usage_mb": profile.get("memory_usage_mb"),
                "memory_usage_bytes": profile.get("memory_usage_bytes")
            },

            "column_types": {
                "numerical": profile.get("numerical_columns", []),
                "categorical": profile.get("categorical_columns", []),
                "datetime": profile.get("datetime_columns", [])
            },

            "column_summary": profile.get("column_summary", {}),

            "column_roles": profile.get("column_roles", {}),

            "diagnostics": {
                "constant_columns": profile.get(
                    "constant_columns", []
                ),

                "near_constant_columns": profile.get(
                    "near_constant_columns", []
                ),

                "type_mismatch_columns": profile.get(
                    "type_mismatch_columns", []
                ),

                "sentinel_report": profile.get(
                    "sentinel_report", {}
                )
            },

            "numerical_summary": profile.get(
                "numerical_summary", {}
            ),

            "categorical_summary": profile.get(
                "categorical_summary", {}
            )

        })

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error profiling dataset: {str(error)}"
        )


# --------------------------------------------------
# DATA QUALITY ASSESSMENT
# --------------------------------------------------

@app.get("/quality")
def get_data_quality():

    check_dataset()

    try:
        quality_results = assess_data_quality(uploaded_dataset)

        return make_json_safe({

            "filename": uploaded_filename,

            "dataset_overview": {
                "rows": uploaded_dataset.shape[0],
                "columns": uploaded_dataset.shape[1]
            },

            "quality_checks": quality_results.get(
                "quality_checks", pd.DataFrame()
            ),

            "missing_report": quality_results.get(
                "missing_report", pd.DataFrame()
            ),

            "missing_summary": {
                "total_missing": quality_results.get(
                    "total_missing", 0
                ),

                "missing_percentage": quality_results.get(
                    "missing_percentage", 0
                )
            },

            "duplicate_summary": {
                "duplicate_count": quality_results.get(
                    "duplicate_count", 0
                ),

                "duplicate_percentage": quality_results.get(
                    "duplicate_percentage", 0
                )
            },

            "outlier_report": quality_results.get(
                "outlier_report", pd.DataFrame()
            ),

            "outlier_summary": {
                "total_outliers": quality_results.get(
                    "total_outliers", 0
                ),

                "count_note": quality_results.get(
                    "outlier_count_note", ""
                )
            },

            "constant_columns": quality_results.get(
                "constant_columns", []
            ),

            "all_missing_columns": quality_results.get(
                "all_missing_columns", []
            ),

            "recommendations": quality_results.get(
                "recommendations", []
            ),

            "legacy_quality_score": quality_results.get(
                "quality_score"
            ),

            "quality_score_deprecated": quality_results.get(
                "quality_score_deprecated", True
            )

        })

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error assessing data quality: {str(error)}"
        )


# --------------------------------------------------
# DATASET INSIGHTS
# --------------------------------------------------

@app.get("/insights")
def get_dataset_insights():

    check_dataset()

    try:
        insights = generate_dataset_insights(uploaded_dataset)

        return {
            "filename": uploaded_filename,

            "dataset_overview": {
                "rows": int(uploaded_dataset.shape[0]),
                "columns": int(uploaded_dataset.shape[1])
            },

            "total_insights": len(insights),

            "insights": insights
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating dataset insights: {str(error)}"
        )


# ==================================================
# VISUALIZATION ENDPOINTS
# ==================================================

# --------------------------------------------------
# HISTOGRAM
# --------------------------------------------------

@app.get("/visualizations/histogram")
def get_histogram(column: str):

    check_dataset()

    if column not in uploaded_dataset.columns:
        raise HTTPException(
            status_code=404,
            detail=f"Column '{column}' was not found."
        )

    try:
        figure = plot_histogram(uploaded_dataset, column)

        return {
            "filename": uploaded_filename,
            "chart_type": "histogram",
            "column": column,
            "chart": convert_figure_to_json(figure)
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating histogram: {str(error)}"
        )


# --------------------------------------------------
# BOXPLOT
# --------------------------------------------------

@app.get("/visualizations/boxplot")
def get_boxplot(column: str):

    check_dataset()

    if column not in uploaded_dataset.columns:
        raise HTTPException(
            status_code=404,
            detail=f"Column '{column}' was not found."
        )

    try:
        figure = plot_boxplot(uploaded_dataset, column)

        interpretation = interpret_boxplot(
            uploaded_dataset,
            column
        )

        return {
            "filename": uploaded_filename,
            "chart_type": "boxplot",
            "column": column,
            "chart": convert_figure_to_json(figure),
            "interpretation": interpretation
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating boxplot: {str(error)}"
        )


# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

@app.get("/visualizations/correlation")
def get_correlation_heatmap(method: str = "pearson"):

    check_dataset()

    method = method.lower().strip()

    if method not in ["pearson", "spearman"]:
        raise HTTPException(
            status_code=400,
            detail="Method must be either 'pearson' or 'spearman'."
        )

    try:
        figure = plot_correlation_heatmap(
            uploaded_dataset,
            method=method
        )

        interpretation = interpret_correlation_heatmap(
            uploaded_dataset,
            method=method
        )

        return {
            "filename": uploaded_filename,
            "chart_type": "correlation_heatmap",
            "method": method,
            "chart": convert_figure_to_json(figure),
            "interpretation": interpretation
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating correlation heatmap: {str(error)}"
        )


# --------------------------------------------------
# SCATTER PLOT
# --------------------------------------------------

@app.get("/visualizations/scatter")
def get_scatter_plot(
    x_column: str,
    y_column: str
):

    check_dataset()

    for column in [x_column, y_column]:

        if column not in uploaded_dataset.columns:
            raise HTTPException(
                status_code=404,
                detail=f"Column '{column}' was not found."
            )

    try:
        figure = plot_scatter(
            uploaded_dataset,
            x_column,
            y_column
        )

        interpretation = interpret_scatter_plot(
            uploaded_dataset,
            x_column,
            y_column
        )

        return {
            "filename": uploaded_filename,
            "chart_type": "scatter_plot",
            "x_column": x_column,
            "y_column": y_column,
            "chart": convert_figure_to_json(figure),
            "interpretation": interpretation
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating scatter plot: {str(error)}"
        )


# --------------------------------------------------
# BAR CHART
# --------------------------------------------------

@app.get("/visualizations/bar")
def get_bar_chart(column: str, top_n: int = 30):

    check_dataset()

    if column not in uploaded_dataset.columns:
        raise HTTPException(
            status_code=404,
            detail=f"Column '{column}' was not found."
        )

    if top_n < 1:
        raise HTTPException(
            status_code=400,
            detail="top_n must be at least 1."
        )

    try:
        figure = plot_bar_chart(
            uploaded_dataset,
            column,
            top_n=top_n
        )

        return {
            "filename": uploaded_filename,
            "chart_type": "bar_chart",
            "column": column,
            "chart": convert_figure_to_json(figure)
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating bar chart: {str(error)}"
        )


# ==================================================
# DATA PREPROCESSING ENDPOINTS
# ==================================================

# --------------------------------------------------
# 1. HANDLE MISSING VALUES
# --------------------------------------------------

@app.post("/preprocessing/missing")
def preprocess_missing_values(
    strategy: str = "auto",
    source: str = "original"
):

    dataframe = get_preprocessing_source(source)

    try:
        cleaned = handle_missing_values(
            dataframe,
            strategy=strategy
        )

        missing_before = int(dataframe.isna().sum().sum())
        missing_after = int(cleaned.isna().sum().sum())

        return save_processed_dataset(
            cleaned,
            "missing_value_handling",
            source,
            {
                "strategy": strategy,
                "missing_values_before": missing_before,
                "missing_values_after": missing_after,
                "missing_values_handled": missing_before - missing_after
            }
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Error handling missing values: {str(error)}"
        )


# --------------------------------------------------
# 2. HANDLE DUPLICATES
# --------------------------------------------------

@app.post("/preprocessing/duplicates")
def preprocess_duplicates(
    keep: str = "first",
    source: str = "original"
):

    dataframe = get_preprocessing_source(source)

    if keep not in {"first", "last", "none"}:
        raise HTTPException(
            status_code=400,
            detail="keep must be 'first', 'last', or 'none'."
        )

    try:
        keep_option = False if keep == "none" else keep

        cleaned, duplicate_count = handle_duplicates(
            dataframe,
            keep=keep_option
        )

        return save_processed_dataset(
            cleaned,
            "duplicate_removal",
            source,
            {
                "keep": keep,
                "duplicates_removed": duplicate_count,
                "rows_before": len(dataframe),
                "rows_after": len(cleaned)
            }
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Error removing duplicates: {str(error)}"
        )


# --------------------------------------------------
# 3. ENCODE CATEGORICAL COLUMNS
# --------------------------------------------------

@app.post("/preprocessing/encoding")
def preprocess_encoding(
    method: str = "onehot",
    source: str = "original"
):

    dataframe = get_preprocessing_source(source)

    if method not in {"onehot", "label"}:
        raise HTTPException(
            status_code=400,
            detail="method must be 'onehot' or 'label'."
        )

    try:
        cleaned = encode_categorical_columns(
            dataframe,
            method=method
        )

        return save_processed_dataset(
            cleaned,
            "categorical_encoding",
            source,
            {
                "method": method,
                "columns_before": dataframe.shape[1],
                "columns_after": cleaned.shape[1],
                "new_columns_created": (
                    cleaned.shape[1] - dataframe.shape[1]
                )
            }
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Error encoding categorical columns: {str(error)}"
        )


# --------------------------------------------------
# 4. SCALE NUMERICAL COLUMNS
# --------------------------------------------------

@app.post("/preprocessing/scaling")
def preprocess_scaling(
    method: str = "standard",
    source: str = "original"
):

    dataframe = get_preprocessing_source(source)

    if method not in {"standard", "minmax"}:
        raise HTTPException(
            status_code=400,
            detail="method must be 'standard' or 'minmax'."
        )

    try:
        cleaned = scale_numerical_columns(
            dataframe,
            method=method
        )

        return save_processed_dataset(
            cleaned,
            "numerical_scaling",
            source,
            {
                "method": method,
                "message": (
                    "Eligible numerical columns were scaled. "
                    "Identifier columns and missing/infinite entries "
                    "were preserved."
                )
            }
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Error scaling numerical columns: {str(error)}"
        )


# --------------------------------------------------
# 5. EXTRACT DATETIME FEATURES
# --------------------------------------------------

@app.post("/preprocessing/datetime")
def preprocess_datetime(
    columns: list[str] = Query(default=None),
    source: str = "original"
):

    dataframe = get_preprocessing_source(source)

    if columns is not None:
        missing_columns = [
            column for column in columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=404,
                detail=f"Columns not found: {missing_columns}"
            )

    try:
        cleaned = extract_datetime_features(
            dataframe,
            columns=columns
        )

        return save_processed_dataset(
            cleaned,
            "datetime_feature_extraction",
            source,
            {
                "selected_columns": columns,
                "columns_before": dataframe.shape[1],
                "columns_after": cleaned.shape[1],
                "new_features_created": (
                    cleaned.shape[1] - dataframe.shape[1]
                )
            }
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Error extracting datetime features: {str(error)}"
        )


# --------------------------------------------------
# DOWNLOAD PROCESSED DATASET
# --------------------------------------------------

@app.get("/preprocessing/download")
def download_processed_dataset():

    if processed_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No processed dataset is available. Run preprocessing first."
        )

    try:
        csv_buffer = io.StringIO()

        processed_dataset.to_csv(
            csv_buffer,
            index=False
        )

        csv_buffer.seek(0)

        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    'attachment; filename="processed_dataset.csv"'
            }
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading processed dataset: {str(error)}"
        )


# --------------------------------------------------
# RESET PROCESSED DATASET
# --------------------------------------------------

@app.post("/preprocessing/reset")
def reset_processed_dataset():

    global processed_dataset

    check_dataset()

    processed_dataset = None

    return {
        "message": "Processed dataset reset successfully.",
        "filename": uploaded_filename,
        "original_dataset_preserved": True
    }


# ==================================================
# LEAKAGE-SAFE MACHINE LEARNING PREPROCESSING
# ==================================================

@app.post("/ml/preprocessing/split")
def ml_preprocessing_split(
    target_column: str,
    test_size: float = Query(default=0.2, gt=0.0, lt=1.0),
    random_state: int = Query(default=42, ge=0)
):
    """
    Split the uploaded dataset into training and test sets.

    The preprocessor is fitted ONLY on X_train. X_test is transformed
    using those training-derived parameters to prevent data leakage.
    The target column is kept separate and is never passed to the
    feature preprocessor.
    """
    global ml_preprocessing_result

    check_dataset()

    if target_column not in uploaded_dataset.columns:
        raise HTTPException(
            status_code=404,
            detail=f"Target column '{target_column}' was not found."
        )

    if uploaded_dataset.shape[0] < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two rows are required for a train/test split."
        )

    if uploaded_dataset[target_column].isna().any():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' contains missing values. "
                "Handle or remove missing target rows before ML preprocessing."
            )
        )

    try:
        X = uploaded_dataset.drop(columns=[target_column]).copy()
        y = uploaded_dataset[target_column].copy()

        # Exclude obvious identifiers from predictive features. Keep the
        # original uploaded dataset unchanged; this affects only ML inputs.
        excluded_identifier_columns = [
            column for column in X.columns
            if is_identifier_feature(column)
        ]
        if excluded_identifier_columns:
            X = X.drop(columns=excluded_identifier_columns)

        if X.shape[1] == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No usable feature columns remain after excluding the "
                    "target and identifier columns."
                )
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        if X_train.empty or X_test.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The selected test_size produced an empty training or "
                    "test set. Use a larger dataset or adjust test_size."
                )
            )

        # Critical leakage-safe order:
        # fit_transform training data; transform test data only.
        preprocessor = build_preprocessor(X_train)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except (AttributeError, ValueError):
            feature_names = []

        ml_preprocessing_result = {
            "preprocessor": preprocessor,
            "X_train": X_train_transformed,
            "X_test": X_test_transformed,
            "y_train": y_train.copy(),
            "y_test": y_test.copy(),
            "target_column": target_column,
            "feature_names": feature_names
        }

        return {
            "message": (
                "Leakage-safe ML preprocessing completed. "
                "The preprocessor was fitted only on training features."
            ),
            "filename": uploaded_filename,
            "target_column": target_column,
            "test_size": test_size,
            "random_state": random_state,
            "training_rows": int(X_train.shape[0]),
            "testing_rows": int(X_test.shape[0]),
            "original_feature_count": int(
                uploaded_dataset.shape[1] - 1
            ),
            "excluded_identifier_columns": excluded_identifier_columns,
            "features_used_for_training": X.columns.tolist(),
            "transformed_feature_count": int(X_train_transformed.shape[1]),
            "training_matrix_shape": list(X_train_transformed.shape),
            "testing_matrix_shape": list(X_test_transformed.shape),
            "training_target_shape": [int(y_train.shape[0])],
            "testing_target_shape": [int(y_test.shape[0])],
            "feature_names": feature_names,
            "leakage_prevention": {
                "preprocessor_fitted_on": "X_train only",
                "test_data_used_during_fit": False,
                "target_included_in_features": False,
                "identifier_columns_excluded": excluded_identifier_columns
            }
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Error during ML preprocessing: {str(error)}"
        )


@app.get("/ml/preprocessing/status")
def ml_preprocessing_status():
    """Report whether a leakage-safe train/test preprocessing run exists."""
    check_dataset()

    if ml_preprocessing_result is None:
        return {
            "ready": False,
            "message": (
                "No ML preprocessing result exists. "
                "Run POST /ml/preprocessing/split first."
            )
        }

    return {
        "ready": True,
        "target_column": ml_preprocessing_result["target_column"],
        "training_matrix_shape": list(
            ml_preprocessing_result["X_train"].shape
        ),
        "testing_matrix_shape": list(
            ml_preprocessing_result["X_test"].shape
        ),
        "transformed_feature_count": int(
            ml_preprocessing_result["X_train"].shape[1]
        )
    }

# ==================================================
# CLASSIFICATION MODEL TRAINING API
# ==================================================

@app.post("/ml/classification/train")
def ml_classification_train(
    target_column: str,
    test_size: float = Query(default=0.2, ge=0.1, le=0.4),
    random_state: int = Query(default=42, ge=0)
):
    """
    Train leakage-safe classification pipelines on the uploaded dataset.

    The classification module creates its own stratified train/test split
    when possible and fits preprocessing only within training data.
    """
    global classification_training_result

    check_dataset()

    try:
        training_output = train_classification_models(
            uploaded_dataset,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state
        )

        model_summaries = {}
        successful = []

        for model_name, result in training_output["results"].items():
            if "error" in result:
                model_summaries[model_name] = {
                    "status": "failed",
                    "error": result["error"]
                }
                continue

            summary = {
                "status": "success",
                "metrics": result["metrics"],
                "confusion_matrix": result["confusion_matrix"],
                "classes": result["classes"],
                "classification_report": result["classification_report"],
                "cross_validation": result.get("cv")
            }
            model_summaries[model_name] = make_json_safe(summary)
            successful.append((model_name, result))

        if not successful:
            raise HTTPException(
                status_code=500,
                detail="All classification models failed."
            )

        # Select the strongest non-baseline model by macro F1.
        non_baseline = [
            item for item in successful
            if not item[0].lower().startswith("dummy")
        ]
        candidates = non_baseline or successful
        best_model_name, best_model_result = max(
            candidates,
            key=lambda item: item[1]["metrics"].get("macro_f1", 0.0)
        )

        classification_training_result = training_output

        return make_json_safe({
            "message": "Classification model training completed.",
            "filename": uploaded_filename,
            "target_column": target_column,
            "test_size": training_output["test_size"],
            "random_state": training_output["random_state"],
            "training_rows": len(training_output["X_train"]),
            "testing_rows": len(training_output["X_test"]),
            "excluded_columns": training_output["removed_columns"],
            "best_model": best_model_name,
            "best_model_metrics": best_model_result["metrics"],
            "models": model_summaries,
            "leakage_prevention": {
                "preprocessing_fitted_with_training_data_only": True,
                "cross_validation_performed_on_training_data_only": True,
                "held_out_test_set_used_for_final_evaluation": True
            }
        })

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Classification training failed: {str(error)}"
        )


# ==================================================
# REGRESSION MODEL TRAINING API
# ==================================================

@app.post("/ml/regression/train")
def ml_regression_train(
    target_column: str,
    test_size: float = Query(default=0.2, ge=0.1, le=0.4),
    random_state: int = Query(default=42, ge=0)
):
    """
    Train leakage-safe regression pipelines on the uploaded dataset.

    Each model and cross-validation fold fits its own preprocessing
    pipeline using training data only. The held-out test set is reserved
    for final evaluation.
    """
    global regression_training_result

    check_dataset()

    try:
        training_output = train_regression_models(
            uploaded_dataset,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state
        )

        model_summaries = {}
        successful = []

        for model_name, result in training_output["results"].items():
            if "error" in result:
                model_summaries[model_name] = {
                    "status": "failed",
                    "error": result["error"]
                }
                continue

            summary = {
                "status": "success",
                "test_metrics": {
                    "MAE": result.get("MAE"),
                    "MSE": result.get("MSE"),
                    "RMSE": result.get("RMSE"),
                    "R2": result.get("R2")
                },
                "training_metrics": result.get("training_metrics"),
                "cross_validation": result.get("cv"),
                "experiment_parameters": result.get("experiment_params"),
                "best_parameters": result.get("best_params"),
                "best_cv_rmse": result.get("best_cv_rmse"),
                "selected_features": result.get("selected_features"),
                "selected_feature_count": result.get("selected_feature_count"),
                "total_preprocessed_features": result.get(
                    "total_preprocessed_features"
                ),
                "feature_importance": result.get("feature_importance")
            }
            model_summaries[model_name] = make_json_safe(summary)
            successful.append((model_name, result))

        if not successful:
            raise HTTPException(
                status_code=500,
                detail="All regression models failed."
            )

        # Prefer a real predictive model over dummy baselines when selecting
        # the headline model; lower held-out RMSE is better.
        non_baseline = [
            item for item in successful
            if not item[0].lower().startswith("dummy")
        ]
        candidates = non_baseline or successful
        best_model_name, best_model_result = min(
            candidates,
            key=lambda item: item[1].get("RMSE", float("inf"))
        )

        regression_training_result = training_output

        return make_json_safe({
            "message": "Regression model training completed.",
            "filename": uploaded_filename,
            "target_column": target_column,
            "test_size": training_output["test_size"],
            "random_state": training_output["random_state"],
            "training_rows": len(training_output["X_train"]),
            "testing_rows": len(training_output["X_test"]),
            "excluded_columns": training_output["removed_columns"],
            "best_model": best_model_name,
            "best_model_test_metrics": {
                "MAE": best_model_result.get("MAE"),
                "MSE": best_model_result.get("MSE"),
                "RMSE": best_model_result.get("RMSE"),
                "R2": best_model_result.get("R2")
            },
            "models": model_summaries,
            "leakage_prevention": {
                "preprocessing_fitted_with_training_data_only": True,
                "cross_validation_performed_on_training_data_only": True,
                "held_out_test_set_used_for_final_evaluation": True
            }
        })

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Regression training failed: {str(error)}"
        )


@app.get("/ml/training/status")
def ml_training_status():
    """Check whether classification or regression training has been run."""
    check_dataset()

    return {
        "classification_ready": classification_training_result is not None,
        "regression_ready": regression_training_result is not None,
        "filename": uploaded_filename
    }



# ==================================================
# AI-POWERED MACHINE LEARNING EXPLANATIONS
# ==================================================

ML_EXPLANATION_SYSTEM_PROMPT = """
You are the explanation assistant inside EDA AI Agent.

Explain only the Python-computed model-training evidence supplied by the API.
Python is the source of truth for all calculations.

Rules:
- Do not invent metrics, feature names, model results, or dataset facts.
- Do not calculate or estimate new statistics.
- Explain metrics in beginner-friendly language.
- Distinguish model performance from proof of real-world usefulness.
- Do not claim correlation or feature importance proves causation.
- For classification, explain class-level results and confusion matrices only
  when those values are supplied.
- For regression, explain MAE, RMSE and R-squared using the supplied values.
- If a requested detail is absent, say it was not included in the results.
- Mention possible overfitting only when the supplied training/test or
  cross-validation evidence supports that observation.
- Keep the answer focused, practical, and structured with short headings.
"""


def build_ml_explanation_prompt(task_type, training_output, user_question=None):
    """Create an LLM prompt from stored, Python-generated model results."""
    evidence = {
        "task_type": task_type,
        "target_column": training_output.get("target_column"),
        "test_size": training_output.get("test_size"),
        "random_state": training_output.get("random_state"),
        "removed_columns": training_output.get("removed_columns", []),
        "training_rows": (
            len(training_output["X_train"])
            if training_output.get("X_train") is not None else None
        ),
        "testing_rows": (
            len(training_output["X_test"])
            if training_output.get("X_test") is not None else None
        ),
        "models": {}
    }

    for model_name, result in training_output.get("results", {}).items():
        if not isinstance(result, dict):
            continue

        if "error" in result:
            evidence["models"][model_name] = {
                "status": "failed",
                "error": str(result["error"])
            }
            continue

        if task_type == "classification":
            evidence["models"][model_name] = {
                "status": "success",
                "metrics": result.get("metrics"),
                "confusion_matrix": result.get("confusion_matrix"),
                "classes": result.get("classes"),
                "classification_report": result.get("classification_report"),
                "cross_validation": result.get("cv")
            }
        else:
            evidence["models"][model_name] = {
                "status": "success",
                "test_metrics": {
                    "MAE": result.get("MAE"),
                    "MSE": result.get("MSE"),
                    "RMSE": result.get("RMSE"),
                    "R2": result.get("R2")
                },
                "training_metrics": result.get("training_metrics"),
                "cross_validation": result.get("cv"),
                "feature_importance": result.get("feature_importance"),
                "selected_features": result.get("selected_features"),
                "best_parameters": result.get("best_params")
            }

    prompt = f"""
Explain these completed {task_type} model-training results.

Use the supplied evidence only:
<python_model_evidence>
{json.dumps(make_json_safe(evidence), ensure_ascii=False)[:14000]}
</python_model_evidence>
"""
    if user_question and user_question.strip():
        prompt += f"""

User's specific question:
<user_question>
{user_question.strip()[:1000]}
</user_question>
"""
    else:
        prompt += """
Please explain:
1. Which model performed best based on the supplied evaluation results.
2. What the main evaluation metrics mean.
3. How the models compare.
4. Any evidence of overfitting or consistency from training and
   cross-validation results, if available.
5. Practical next steps.
"""
    prompt += "\nDo not calculate any additional values."
    return prompt


@app.post("/ml/classification/explain")
def explain_classification_results(
    question: str = Query(default=None, max_length=1000)
):
    """Ask the local LLM to explain the latest classification results."""
    check_dataset()

    if classification_training_result is None:
        raise HTTPException(
            status_code=400,
            detail="Run POST /ml/classification/train before requesting an explanation."
        )

    prompt = build_ml_explanation_prompt(
        "classification",
        classification_training_result,
        question
    )
    explanation = generate_response(
        prompt,
        system_prompt=ML_EXPLANATION_SYSTEM_PROMPT,
        num_predict=420
    )

    return {
        "message": "Classification results explanation generated.",
        "filename": uploaded_filename,
        "target_column": classification_training_result.get("target_column"),
        "explanation": explanation,
        "evidence_source": "Python-generated classification evaluation results",
        "llm_model": "llama3.2:1b"
    }


@app.post("/ml/regression/explain")
def explain_regression_results(
    question: str = Query(default=None, max_length=1000)
):
    """Ask the local LLM to explain the latest regression results."""
    check_dataset()

    if regression_training_result is None:
        raise HTTPException(
            status_code=400,
            detail="Run POST /ml/regression/train before requesting an explanation."
        )

    prompt = build_ml_explanation_prompt(
        "regression",
        regression_training_result,
        question
    )
    explanation = generate_response(
        prompt,
        system_prompt=ML_EXPLANATION_SYSTEM_PROMPT,
        num_predict=420
    )

    return {
        "message": "Regression results explanation generated.",
        "filename": uploaded_filename,
        "target_column": regression_training_result.get("target_column"),
        "explanation": explanation,
        "evidence_source": "Python-generated regression evaluation results",
        "llm_model": "llama3.2:1b"
    }

# ==================================================


def add_categorical_rag_evidence(dataframe, collection_name):
    """
    Add explicit, Python-computed category counts to the active ChromaDB
    collection. This gives RAG searchable evidence for questions such as
    'Which department has the most students?'.
    """
    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns

    documents = []
    metadatas = []
    ids = []

    for column_index, column in enumerate(categorical_columns):
        values = dataframe[column].dropna().astype(str).str.strip()
        values = values[values != ""]

        if values.empty:
            continue

        counts = values.value_counts()
        total = int(counts.sum())
        category_count = int(len(counts))
        top_count = int(counts.iloc[0])
        top_categories = [
            str(category)
            for category, count in counts.items()
            if int(count) == top_count
        ]

        details = []
        for category, count in counts.head(30).items():
            count = int(count)
            percentage = count / total * 100 if total else 0
            details.append(
                f"'{category}' = {count:,} ({percentage:.2f}%)"
            )

        omitted = max(0, category_count - len(details))
        omitted_note = (
            f"; {omitted:,} lower-frequency categories omitted"
            if omitted else ""
        )

        top_text = ", ".join(f"'{item}'" for item in top_categories)
        document = (
            f"Categorical distribution for column '{column}': "
            f"{category_count:,} observed categories among {total:,} "
            f"non-missing rows. "
            f"The most frequent category in column '{column}' is "
            f"{top_text}, with {top_count:,} rows. "
            f"Category counts and percentages: {'; '.join(details)}"
            f"{omitted_note}."
        )

        documents.append(document)
        metadatas.append({
            "source": "python_categorical_distribution",
            "column_name": str(column),
        })
        ids.append(f"eda_category_{column_index}")

    if not documents:
        return 0

    engine = EDARetrievalEngine()
    collection = engine.client.get_collection(name=collection_name)

    # Remove only category documents from the previous indexing run.
    existing = collection.get()
    old_category_ids = [
        item_id for item_id in existing.get("ids", [])
        if str(item_id).startswith("eda_category_")
    ]
    if old_category_ids:
        collection.delete(ids=old_category_ids)

    embeddings = engine.embedding_model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(documents)


# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ==================================================

@app.post("/rag/index")
def index_uploaded_dataset_for_rag():
    """
    Generate Python-computed dataset insights and index them in ChromaDB.

    Run this once after uploading a dataset, and again after uploading
    a different dataset.
    """
    global rag_collection_name

    check_dataset()

    collection_name = "eda_uploaded_dataset_knowledge"

    try:
        # Start from a clean collection for every indexing run. The dataset
        # insight builder uses predictable document IDs (document_0, etc.);
        # upserting a smaller new dataset without clearing the collection can
        # leave higher-numbered documents from the previous dataset behind.
        collection_manager = EDARetrievalEngine()
        existing_collections = collection_manager.client.list_collections()
        existing_names = {
            item.name if hasattr(item, "name") else str(item)
            for item in existing_collections
        }
        collection_was_reset = collection_name in existing_names
        if collection_was_reset:
            collection_manager.client.delete_collection(
                name=collection_name
            )

        indexing_result = index_dataset_insights(
            uploaded_dataset,
            collection_name=collection_name
        )

        categorical_documents_indexed = add_categorical_rag_evidence(
            uploaded_dataset,
            collection_name
        )
        indexing_result["categorical_documents_indexed"] = (
            categorical_documents_indexed
        )
        indexing_result["total_documents_available"] = (
            int(indexing_result.get("documents_indexed", 0))
            + categorical_documents_indexed
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"RAG indexing failed: {str(error)}"
        ) from error

    rag_collection_name = collection_name

    return {
        "message": "Uploaded dataset successfully indexed for RAG.",
        "filename": uploaded_filename,
        "previous_collection_cleared": collection_was_reset,
        **indexing_result
    }


@app.post("/rag/ask")
def ask_rag_dataset_question(
    question: str = Query(..., min_length=1, max_length=1000),
    top_k: int = Query(default=3, ge=1, le=10)
):
    """
    Answer a question using retrieved evidence from the uploaded dataset.
    """
    check_dataset()

    if rag_collection_name is None:
        raise HTTPException(
            status_code=400,
            detail="Run POST /rag/index after uploading the dataset before asking RAG questions."
        )

    try:
        result = answer_dataset_question(
            question=question,
            collection_name=rag_collection_name,
            top_k=top_k
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"RAG question answering failed: {str(error)}"
        ) from error

    return {
        "filename": uploaded_filename,
        **result
    }


@app.get("/rag/status")
def rag_index_status():
    """Report whether the current uploaded dataset has been indexed for RAG."""
    check_dataset()

    return {
        "filename": uploaded_filename,
        "indexed": rag_collection_name is not None,
        "collection_name": rag_collection_name
    }


# ==================================================
# LANGCHAIN QUESTION ANSWERING API
# ==================================================

@app.post("/langchain/ask")
def ask_with_langchain(request: LangChainQuestionRequest):
    """
    Answer a dataset question using LangChain, Hugging Face embeddings,
    the active ChromaDB collection, and the configured Ollama model.

    Upload a dataset and run POST /rag/index before using this endpoint.
    The existing /rag/ask endpoint remains unchanged.
    """
    global langchain_engine

    check_dataset()

    if rag_collection_name is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Run POST /rag/index after uploading the dataset "
                "before asking LangChain questions."
            )
        )

    try:
        # Reuse the engine for repeated questions. Reinitialize it after
        # a new upload so it always points to the active collection.
        if langchain_engine is None:
            langchain_engine = EDALangChainEngine(
                collection_name=rag_collection_name
            )

        result = langchain_engine.answer_question(
            question=request.question,
            top_k=request.top_k
        )

        return {
            "filename": uploaded_filename,
            **result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"LangChain question answering failed: {str(error)}"
        ) from error

