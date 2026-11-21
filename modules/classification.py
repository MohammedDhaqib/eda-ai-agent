"""Leakage-safe classification training with baselines and evaluation.

Preprocessing and cross-validation are fitted using training data only.
The held-out test set is reserved for final model evaluation.
"""

import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from modules.ml_preprocessor import build_preprocessor


_ID_NAMES = {"id", "index", "rowid", "recordid", "passengerid", "customerid"}


def _is_identifier_name(column):
    """Conservatively identify columns whose names clearly indicate an ID."""
    name = re.sub(r"[^a-z0-9]", "", str(column).strip().lower())
    return name in _ID_NAMES or name.endswith("identifier")


def _validate_parameters(test_size, random_state):
    """Validate split parameters and return normalized values."""
    try:
        test_size = float(test_size)
    except (TypeError, ValueError):
        raise ValueError("test_size must be a number between 0.1 and 0.4.") from None

    if not np.isfinite(test_size) or not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")

    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise ValueError("random_state must be a non-negative integer.")
    if not 0 <= int(random_state) <= 2**32 - 1:
        raise ValueError("random_state must be between 0 and 2**32 - 1.")

    return test_size, int(random_state)


def _metrics(y_true, predictions, probabilities, classes):
    """Calculate test metrics, including ROC-AUC when it is defined."""
    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "roc_auc": None,
    }

    try:
        if probabilities is not None and len(classes) == 2:
            y_binary = (np.asarray(y_true) == classes[1]).astype(int)
            result["roc_auc"] = float(roc_auc_score(y_binary, probabilities[:, 1]))
        elif probabilities is not None and len(classes) > 2:
            result["roc_auc"] = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    labels=classes,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    except (ValueError, TypeError, IndexError):
        # ROC-AUC is unavailable for some small or unusual test partitions.
        result["roc_auc"] = None

    return result


def train_classification_models(
    dataframe, target_column, test_size=0.2, random_state=42
):
    """Train classification pipelines and return dashboard-compatible results."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("The uploaded dataset is empty.")
    if not dataframe.columns.is_unique:
        raise ValueError(
            "Duplicate column names were found. Rename duplicate columns "
            "before running classification."
        )
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    test_size, random_state = _validate_parameters(test_size, random_state)

    data = dataframe.copy()
    data = data.dropna(subset=[target_column])
    if len(data) < 10:
        raise ValueError("At least 10 rows with a non-missing target are required.")

    y = data[target_column]
    if y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    X = data.drop(columns=[target_column])
    removed_columns = []

    unnamed_columns = [
        column for column in X.columns
        if str(column).strip().lower().startswith("unnamed:")
    ]
    identifier_columns = [
        column for column in X.columns if _is_identifier_name(column)
    ]
    excluded_columns = list(dict.fromkeys(unnamed_columns + identifier_columns))
    if excluded_columns:
        X = X.drop(columns=excluded_columns)
        removed_columns.extend(excluded_columns)

    unsupported_columns = [
        column
        for column in X.columns
        if (
            pd.api.types.is_datetime64_any_dtype(X[column])
            or pd.api.types.is_timedelta64_dtype(X[column])
            or pd.api.types.is_complex_dtype(X[column])
        )
    ]
    if unsupported_columns:
        X = X.drop(columns=unsupported_columns)
        removed_columns.extend(unsupported_columns)

    # Do not inspect the complete feature data to remove constant or
    # all-missing columns. Their handling belongs inside the training-fitted
    # preprocessing pipeline, avoiding test-partition-informed decisions.
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after exclusions.")

    numeric_columns = X.select_dtypes(include=["number"]).columns
    if len(numeric_columns):
        X.loc[:, numeric_columns] = X[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )

    class_counts = y.value_counts()
    number_of_classes = int(y.nunique())
    can_stratify = (
        int(class_counts.min()) >= 2
        and int(np.ceil(len(y) * test_size)) >= number_of_classes
        and int(np.floor(len(y) * (1 - test_size))) >= number_of_classes
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )

    if y_train.nunique() < 2:
        raise ValueError(
            "The training split contains fewer than two classes. "
            "Add more examples for each class or adjust the test size."
        )

    unseen_test_classes = set(pd.unique(y_test)) - set(pd.unique(y_train))
    if unseen_test_classes:
        unseen_labels = ", ".join(sorted(map(str, unseen_test_classes)))
        raise ValueError(
            "Some target classes occur only in the test split: "
            f"{unseen_labels}. Add more examples of these classes."
        )

    min_class_train = int(y_train.value_counts().min())
    cv_splits = min(5, min_class_train) if min_class_train >= 2 else 0

    models = {
        "Dummy Classifier (Most Frequent)": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
    }

    results = {}
    for model_name, estimator in models.items():
        try:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(X_train)),
                    ("model", estimator),
                ]
            )

            cv_summary = None
            if cv_splits >= 2:
                cv = StratifiedKFold(
                    n_splits=cv_splits,
                    shuffle=True,
                    random_state=random_state,
                )
                scoring = {
                    "accuracy": "accuracy",
                    "balanced_accuracy": "balanced_accuracy",
                    "f1_weighted": "f1_weighted",
                }
                scores = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,
                    error_score="raise",
                )
                cv_summary = {
                    key: {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1))
                        if len(values) > 1 else 0.0,
                    }
                    for key, values in scores.items()
                    if key.startswith("test_")
                }

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            probabilities = (
                pipeline.predict_proba(X_test)
                if hasattr(pipeline, "predict_proba") else None
            )
            classes = np.asarray(pipeline.classes_)

            results[model_name] = {
                "model": pipeline,
                "predictions": predictions,
                "metrics": _metrics(y_test, predictions, probabilities, classes),
                "confusion_matrix": confusion_matrix(
                    y_test, predictions, labels=classes
                ),
                "classification_report": classification_report(
                    y_test,
                    predictions,
                    labels=classes,
                    zero_division=0,
                    output_dict=True,
                ),
                "classes": classes,
                "cv": cv_summary,
            }
        except Exception as error:
            results[model_name] = {"error": str(error)}

    successful_models = [
        value for value in results.values()
        if isinstance(value, dict) and "error" not in value
    ]
    if not successful_models:
        details = "; ".join(
            f"{name}: {value.get('error', 'Unknown error')}"
            for name, value in results.items()
        )
        raise RuntimeError("All classification models failed. " + details)

    return {
        "results": results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_column": target_column,
        "removed_columns": removed_columns,
        "random_state": random_state,
        "test_size": test_size,
        "split_indices": {
            "train": X_train.index.tolist(),
            "test": X_test.index.tolist(),
        },
    }
