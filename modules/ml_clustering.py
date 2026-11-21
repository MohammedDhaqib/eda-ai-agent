import re
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def perform_clustering(
    dataframe,
    n_clusters=3,
    random_state=42
):
    """
    Perform K-Means clustering on a dataset.

    Steps:
        1. Select numerical features.
        2. Remove identifier and generated index columns.
        3. Handle missing and infinite values.
        4. Impute missing values.
        5. Scale numerical features.
        6. Train K-Means.
        7. Calculate silhouette score.

    Returns:
        Dictionary containing cluster assignments,
        trained model, preprocessing pipeline and evaluation results.
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

    if (
        not isinstance(n_clusters, (int, np.integer))
        or isinstance(n_clusters, (bool, np.bool_))
        or n_clusters < 2
    ):
        raise ValueError(
            "Number of clusters must be an integer of at least 2."
        )

    # Select numerical columns
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

    if numerical_data.shape[1] == 0:
        raise ValueError(
            "No suitable numerical columns found for clustering."
        )

    # Replace infinite values with missing values
    numerical_data = numerical_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove columns with no usable values or no variation
    valid_columns = [
        column
        for column in numerical_data.columns
        if (
            numerical_data[column].notna().any()
            and numerical_data[column].nunique(dropna=True) > 1
        )
    ]

    numerical_data = numerical_data[valid_columns]

    if numerical_data.shape[1] == 0:
        raise ValueError(
            "No numerical features with sufficient variation found."
        )

    # Remove rows with no observed feature values
    numerical_data = numerical_data.dropna(
        axis=0,
        how="all"
    )

    if len(numerical_data) <= n_clusters:
        raise ValueError(
            "The number of usable rows must be greater than "
            "the requested number of clusters."
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

    # Train K-Means
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    cluster_labels = model.fit_predict(
        processed_data
    )

    # Count distinct clusters actually produced
    unique_clusters = np.unique(cluster_labels)

    actual_cluster_count = len(unique_clusters)

    # Calculate silhouette score when valid
    if 1 < actual_cluster_count < len(cluster_labels):

        score = silhouette_score(
            processed_data,
            cluster_labels
        )

    else:
        score = None

    # Preserve original rows and their indices
    clustered_data = dataframe.loc[
        numerical_data.index
    ].copy()

    clustered_data["Cluster"] = cluster_labels

    return {
        "model": model,
        "preprocessor": preprocessing_pipeline,
        "clustered_data": clustered_data,
        "cluster_labels": cluster_labels,
        "silhouette_score": score,
        "n_clusters": n_clusters,
        "actual_cluster_count": actual_cluster_count,
        "feature_columns": numerical_data.columns.tolist(),
        "row_indices": numerical_data.index.tolist()
    }