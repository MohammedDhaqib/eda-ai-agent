"""Leakage-safe regression training with Random Forest generalization experiments."""
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from modules.ml_preprocessor import build_preprocessor

_ID_NAMES = {"id", "index", "rowid", "recordid", "passengerid", "customerid"}


def _is_identifier_name(column):
    name = re.sub(r"[^a-z0-9]", "", str(column).strip().lower())
    return name in _ID_NAMES or name.endswith("identifier")


def _validate_parameters(test_size, random_state):
    try:
        test_size = float(test_size)
    except (TypeError, ValueError):
        raise ValueError("test_size must be a number between 0.1 and 0.4.") from None
    if not np.isfinite(test_size) or not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)):
        raise ValueError("random_state must be a non-negative integer.")
    random_state = int(random_state)
    if not 0 <= random_state <= 2**32 - 1:
        raise ValueError("random_state must be between 0 and 2**32 - 1.")
    return test_size, random_state


def _score(y_true, predictions):
    mse = mean_squared_error(y_true, predictions)
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, predictions)) if len(y_true) >= 2 else None,
    }


def _positive_cv_summary(scores):
    summary = {}
    for key, values in scores.items():
        if not (key.startswith("test_") or key.startswith("train_")):
            continue
        metric = key.removeprefix("test_").removeprefix("train_")
        split = "test" if key.startswith("test_") else "train"
        values = np.asarray(values, dtype=float)
        if metric in {"mae", "rmse"}:
            values = -values
        summary.setdefault(metric, {})[split] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "fold_scores": [float(value) for value in values],
        }
    flattened = {}
    for metric, splits in summary.items():
        if "test" in splits:
            flattened[metric] = splits["test"]
        if "train" in splits:
            flattened[f"train_{metric}"] = splits["train"]
    return flattened


def _make_cv(n_rows, random_state):
    """Only use folds with at least two validation observations (valid R²)."""
    n_splits = min(5, n_rows // 2)
    if n_splits < 2:
        return None
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _train_tuned_random_forest(X_train, X_test, y_train, y_test, random_state):
    """Fit preprocessing, feature selection and tuning only within training CV."""
    cv_splits = min(3, len(X_train) // 2)
    if cv_splits < 2:
        raise ValueError(
            "Not enough training rows for tuning with validation folds of at least "
            "two rows. The tuned model was skipped."
        )
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    selector = SelectFromModel(
        estimator=RandomForestRegressor(
            n_estimators=60, max_features=1.0, random_state=random_state, n_jobs=1
        ),
        threshold=-np.inf,
        max_features=lambda X: max(1, int(np.ceil(X.shape[1] * 0.70))),
    )
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("feature_selection", selector),
        ("model", RandomForestRegressor(random_state=random_state, n_jobs=1)),
    ])
    parameter_distributions = {
        "model__n_estimators": [100, 150],
        "model__max_depth": [None, 6, 10, 16],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4, 8, 12],
        "model__max_features": [1.0, "sqrt", 0.7],
    }
    search = RandomizedSearchCV(
        pipeline, parameter_distributions, n_iter=8,
        scoring="neg_root_mean_squared_error", cv=cv,
        random_state=random_state, n_jobs=1, refit=True,
        error_score="raise", return_train_score=False,
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    train_predictions = best_pipeline.predict(X_train)
    predictions = best_pipeline.predict(X_test)
    best_index = int(search.best_index_)
    fold_rmse = [
        float(-search.cv_results_[f"split{i}_test_score"][best_index])
        for i in range(cv_splits)
    ]
    tuned_cv = {
        "rmse": {
            "mean": float(np.mean(fold_rmse)),
            "std": float(np.std(fold_rmse, ddof=1)) if len(fold_rmse) > 1 else 0.0,
            "fold_scores": fold_rmse,
        }
    }

    selected_features, total_features = [], 0
    try:
        names = np.asarray(best_pipeline.named_steps["preprocessor"].get_feature_names_out())
        support = best_pipeline.named_steps["feature_selection"].get_support()
        total_features = int(len(names))
        selected_features = [str(n).replace("num__", "").replace("cat__", "") for n in names[support]]
    except (AttributeError, ValueError, TypeError):
        try:
            total_features = int(best_pipeline.named_steps["feature_selection"].n_features_in_)
        except (AttributeError, TypeError, ValueError):
            pass

    return {
        "model": best_pipeline, "predictions": predictions,
        **_score(y_test, predictions),
        "training_metrics": _score(y_train, train_predictions),
        "cv": tuned_cv,
        "best_params": search.best_params_,
        "best_cv_rmse": float(-search.best_score_),
        "selected_features": selected_features,
        "selected_feature_count": int(best_pipeline.named_steps["feature_selection"].get_support().sum()),
        "total_preprocessed_features": total_features,
        "cv_splits": cv_splits,
    }



def _train_rf_experiment(X_train, X_test, y_train, y_test, random_state, cv, config):
    """Evaluate a fixed Random Forest configuration without touching the test set during CV."""
    estimator = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=1,
        **config,
    )
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", estimator),
    ])
    cv_summary = None
    if cv is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(
                pipeline, X_train, y_train, cv=cv,
                scoring={
                    "mae": "neg_mean_absolute_error",
                    "rmse": "neg_root_mean_squared_error",
                    "r2": "r2",
                },
                return_train_score=True,
                n_jobs=1,
                error_score="raise",
            )
        cv_summary = _positive_cv_summary(scores)
    pipeline.fit(X_train, y_train)
    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)
    return {
        "model": pipeline,
        "predictions": test_predictions,
        **_score(y_test, test_predictions),
        "training_metrics": _score(y_train, train_predictions),
        "cv": cv_summary,
        "experiment_params": dict(config),
    }


def train_regression_models(dataframe, target_column, test_size=0.2, random_state=42):
    """Train regression models and return the dashboard-compatible result shape."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("The uploaded dataset is empty.")
    if not dataframe.columns.is_unique:
        raise ValueError("Duplicate column names were found. Rename duplicate columns before running regression.")
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")
    test_size, random_state = _validate_parameters(test_size, random_state)

    data = dataframe.copy()
    unnamed = [c for c in data.columns if str(c).strip().lower().startswith("unnamed:")]
    data = data.drop(columns=unnamed, errors="ignore")
    if target_column not in data.columns:
        raise ValueError("Target column was removed as an unnamed index column.")
    data = data.dropna(subset=[target_column])
    if not pd.api.types.is_numeric_dtype(data[target_column]):
        raise ValueError("Regression requires a numerical target.")
    target_values = pd.to_numeric(data[target_column], errors="coerce")
    try:
        finite_mask = np.isfinite(target_values.to_numpy(dtype=float))
    except (TypeError, ValueError):
        raise ValueError("The regression target must contain finite numeric values.") from None
    data = data.loc[finite_mask].copy()
    if len(data) < 10:
        raise ValueError(f"At least 10 usable rows are required; found {len(data)}.")
    if data[target_column].nunique() < 2:
        raise ValueError("The regression target must contain at least two distinct values.")

    y = data[target_column].astype(float)
    X = data.drop(columns=[target_column])
    identifiers = [c for c in X.columns if _is_identifier_name(c)]
    unsupported = [c for c in X.columns if (
        pd.api.types.is_datetime64_any_dtype(X[c])
        or pd.api.types.is_timedelta64_dtype(X[c])
        or pd.api.types.is_complex_dtype(X[c])
    )]
    # Avoid using full-dataset constant/all-missing checks: they can inspect test rows.
    removed_columns = list(dict.fromkeys(identifiers + unsupported))
    X = X.drop(columns=removed_columns, errors="ignore")
    numeric = X.select_dtypes(include=["number"]).columns
    if len(numeric):
        X.loc[:, numeric] = X[numeric].replace([np.inf, -np.inf], np.nan)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")
    if len(X) != len(y):
        raise RuntimeError("Internal data alignment error: feature and target row counts differ.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if len(X_train) < 3:
        raise ValueError("Not enough training rows for regression.")
    cv = _make_cv(len(X_train), random_state)

    models = {
        "Dummy Regressor (Mean)": DummyRegressor(strategy="mean"),
        "Dummy Regressor (Median)": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150, random_state=random_state, n_jobs=1
        ),
    }
    results = {}
    for model_name, estimator in models.items():
        try:
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ])
            cv_summary = None
            if cv is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_validate(
                        pipeline, X_train, y_train, cv=cv,
                        scoring={
                            "mae": "neg_mean_absolute_error",
                            "rmse": "neg_root_mean_squared_error",
                            "r2": "r2",
                        },
                        return_train_score=True, n_jobs=1, error_score="raise",
                    )
                cv_summary = _positive_cv_summary(scores)
            pipeline.fit(X_train, y_train)
            train_predictions = pipeline.predict(X_train)
            predictions = pipeline.predict(X_test)
            result = {
                "model": pipeline, "predictions": predictions,
                **_score(y_test, predictions),
                "training_metrics": _score(y_train, train_predictions),
                "cv": cv_summary,
            }
            if model_name == "Random Forest Regressor":
                try:
                    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
                    importances = pipeline.named_steps["model"].feature_importances_
                    if len(names) == len(importances):
                        result["feature_importance"] = pd.DataFrame({
                            "Feature": [str(n).replace("num__", "").replace("cat__", "") for n in names],
                            "Importance": importances,
                        }).sort_values("Importance", ascending=False).reset_index(drop=True)
                except (AttributeError, ValueError, TypeError):
                    result["feature_importance"] = pd.DataFrame(columns=["Feature", "Importance"])
            results[model_name] = result
        except Exception as error:
            results[model_name] = {"error": str(error)}

    # Controlled experiments isolate common Random Forest regularization choices.
    rf_experiments = {
        "Random Forest (Shallow)": {
            "max_depth": 8, "min_samples_split": 5,
            "min_samples_leaf": 5, "max_features": 0.8,
        },
        "Random Forest (Regularized)": {
            "max_depth": 16, "min_samples_split": 10,
            "min_samples_leaf": 8, "max_features": 0.8,
        },
        "Random Forest (Feature Subsampling)": {
            "max_depth": None, "min_samples_split": 2,
            "min_samples_leaf": 2, "max_features": "sqrt",
        },
    }
    for experiment_name, config in rf_experiments.items():
        try:
            results[experiment_name] = _train_rf_experiment(
                X_train, X_test, y_train, y_test,
                random_state, cv, config,
            )
        except Exception as error:
            results[experiment_name] = {"error": str(error)}

    try:
        results["Tuned Random Forest (Feature Selection)"] = _train_tuned_random_forest(
            X_train, X_test, y_train, y_test, random_state
        )
    except Exception as error:
        results["Tuned Random Forest (Feature Selection)"] = {"error": str(error)}

    successful = [v for v in results.values() if isinstance(v, dict) and "error" not in v]
    if not successful:
        details = "; ".join(f"{name}: {value.get('error', 'Unknown error')}" for name, value in results.items())
        raise RuntimeError("All regression models failed. " + details)

    return {
        "results": results,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "target_column": target_column,
        "removed_columns": list(dict.fromkeys(removed_columns)),
        "identifier_experiment": None,
        "random_state": random_state,
        "test_size": test_size,
        "split_indices": {"train": X_train.index.tolist(), "test": X_test.index.tolist()},
    }
