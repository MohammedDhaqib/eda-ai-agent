import re
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def detect_anomalies(
    dataframe,
    contamination=0.05,
    random_state=42
):
    """
    Detect anomalies using Isolation Forest.

    Steps:
        1. Select numerical features.
        2. Exclude identifier and generated index columns.
        3. Handle missing and infinite values.
        4. Scale numerical features.
        5. Train Isolation Forest.
        6. Identify normal observations and anomalies.
        7. Calculate anomaly scores.

    Returns:
        Dictionary containing the trained model,
        processed results and anomaly statistics.
    """

    # Validate input
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "The dataset is empty."
        )

    if not 0 < contamination <= 0.5:
        raise ValueError(
            "Contamination must be greater than 0 and at most 0.5."
        )

    # Select numerical features
    numerical_data = dataframe.select_dtypes(
        include=["number"]
    ).copy()

    # Exclude common identifier columns
    excluded_names = {
        "id",
        "index",
        "unnamed: 0",
        "unnamed:0"
    }

    numerical_columns = [
        column
        for column in numerical_data.columns
        if (
            str(column).strip().lower() not in excluded_names
            and not re.fullmatch(
                r"Unnamed:\s*\d+",
                str(column).strip(),
                flags=re.IGNORECASE
            )
        )
    ]

    numerical_data = numerical_data[numerical_columns]

    if numerical_data.empty:
        raise ValueError(
            "No suitable numerical columns found for anomaly detection."
        )

    # Replace infinite values with missing values
    numerical_data = numerical_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove columns containing only missing values or constants
    valid_columns = [
        column
        for column in numerical_data.columns
        if (
            numerical_data[column].notna().any()
            and numerical_data[column].nunique(dropna=True) > 1
        )
    ]

    numerical_data = numerical_data[valid_columns]

    if numerical_data.empty:
        raise ValueError(
            "No numerical features with sufficient variation found."
        )

    # Keep rows containing at least one observed numerical value
    valid_rows = numerical_data.notna().any(axis=1)

    numerical_data = numerical_data.loc[valid_rows]

    if len(numerical_data) < 5:
        raise ValueError(
            "At least 5 usable rows are required for anomaly detection."
        )

    # Build preprocessing pipeline
    preprocessing_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    processed_data = preprocessing_pipeline.fit_transform(
        numerical_data
    )

    # Train Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )

    predictions = model.fit_predict(processed_data)

    # Calculate anomaly scores
    anomaly_scores = model.decision_function(
        processed_data
    )

    # Convert predictions:
    # 1 = Normal observation
    # -1 = Anomaly

    result_data = dataframe.loc[
        numerical_data.index
    ].copy()

    result_data["Anomaly Status"] = [
        "Anomaly" if prediction == -1 else "Normal"
        for prediction in predictions
    ]

    result_data["Anomaly Score"] = anomaly_scores

    # Calculate anomaly statistics
    anomaly_count = int(
        (predictions == -1).sum()
    )

    normal_count = int(
        (predictions == 1).sum()
    )

    anomaly_percentage = (
        anomaly_count / len(predictions)
    ) * 100

    return {
        "model": model,
        "preprocessor": preprocessing_pipeline,
        "results": result_data,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "anomaly_percentage": anomaly_percentage,
        "feature_columns": numerical_data.columns.tolist(),
        "row_indices": numerical_data.index.tolist()
    }