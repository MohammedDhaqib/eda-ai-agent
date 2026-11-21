import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# 1. NEURAL NETWORK
# ==================================================

class TabularNeuralNetwork(nn.Module):

    def __init__(self, input_size, output_size, task="classification"):
        super().__init__()

        self.task = task

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.network(x)


# ==================================================
# 2. PREPROCESSING
# ==================================================

def prepare_features(X_fit, X_validation, X_test):

    X_fit = X_fit.copy()
    X_validation = X_validation.copy()
    X_test = X_test.copy()

    numeric_columns = X_fit.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for frame in [X_fit, X_validation, X_test]:
        frame[numeric_columns] = frame[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )

    usable_columns = [
        column for column in X_fit.columns
        if not X_fit[column].isna().all()
    ]

    X_fit = X_fit[usable_columns]
    X_validation = X_validation[usable_columns]
    X_test = X_test[usable_columns]

    if X_fit.empty:
        raise ValueError("No usable feature columns remain.")

    numeric_columns = X_fit.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_fit.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    transformers = []

    if numeric_columns:
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numerical", numerical_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("No numerical or categorical features found.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    X_fit_processed = preprocessor.fit_transform(X_fit)
    X_validation_processed = preprocessor.transform(X_validation)
    X_test_processed = preprocessor.transform(X_test)

    X_fit_processed = np.nan_to_num(
        np.asarray(X_fit_processed, dtype=np.float32)
    )

    X_validation_processed = np.nan_to_num(
        np.asarray(X_validation_processed, dtype=np.float32)
    )

    X_test_processed = np.nan_to_num(
        np.asarray(X_test_processed, dtype=np.float32)
    )

    return (
        X_fit_processed,
        X_validation_processed,
        X_test_processed,
        preprocessor
    )


# ==================================================
# 3. TRAINING HISTORY INTERPRETATION
# ==================================================

def interpret_training_history(history):

    if not history:
        return {
            "status": "Unavailable",
            "explanation": "Training history is unavailable.",
            "overfitting_status": "Unavailable",
            "validation_explanation": "Validation history unavailable.",
            "recommendations": []
        }

    training_losses = [item["loss"] for item in history]
    validation_losses = [
        item["validation_loss"] for item in history
    ]

    first_training = training_losses[0]
    last_training = training_losses[-1]

    first_validation = validation_losses[0]
    last_validation = validation_losses[-1]

    recommendations = []

    if last_training < first_training:
        status = "Learning detected"
        explanation = (
            "Training loss decreased. The neural network is learning "
            "patterns from the training dataset."
        )
    else:
        status = "Learning needs attention"
        explanation = (
            "Training loss did not decrease. Consider adjusting "
            "the learning rate or model settings."
        )
        recommendations.append(
            "Experiment with the learning rate and training settings."
        )

    if (
        last_validation > last_training * 1.25
        and last_validation > first_validation
    ):
        overfitting = "Possible overfitting"
        recommendations.append(
            "Try fewer epochs, stronger regularization or more training data."
        )
    elif last_validation < first_validation:
        overfitting = "No clear overfitting signal"
    else:
        overfitting = "Validation performance needs attention"
        recommendations.append(
            "Inspect the validation-loss curve and adjust the model."
        )

    validation_explanation = (
        "Validation loss decreased during training, which is encouraging."
        if last_validation < first_validation
        else "Validation loss did not improve overall."
    )

    best_epoch = min(
        range(len(validation_losses)),
        key=lambda index: validation_losses[index]
    ) + 1

    epochs_trained = len(history)
    stopped_early = bool(history[-1].get("stopped_early", False))

    if stopped_early:
        recommendations.append(
            f"Early stopping activated after {epochs_trained} epochs. "
            f"The best validation loss occurred at epoch {best_epoch}; "
            "the model's best weights were restored."
        )

    return {
        "status": status,
        "explanation": explanation,
        "overfitting_status": overfitting,
        "validation_explanation": validation_explanation,
        "recommendations": recommendations,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "stopped_early": stopped_early
    }


# ==================================================
# 4. CLASSIFICATION INTERPRETATION
# ==================================================

def interpret_classification_results(results):

    accuracy = results["accuracy"]
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1_score"]

    observations = []
    recommendations = []

    if accuracy >= 0.90:
        observations.append(
            "The model achieved very high classification accuracy."
        )
    elif accuracy >= 0.75:
        observations.append(
            "The model achieved reasonably good classification accuracy."
        )
    elif accuracy >= 0.60:
        observations.append(
            "The model has moderate classification performance."
        )
    else:
        observations.append(
            "The model has low classification accuracy."
        )
        recommendations.append(
            "Review features, target labels and data quality."
        )

    if precision < recall:
        observations.append(
            "Precision is lower than recall. Some positive predictions "
            "may be incorrect."
        )
    elif recall < precision:
        observations.append(
            "Recall is lower than precision. The model may miss "
            "some actual positive cases."
        )

    if f1 < 0.60:
        recommendations.append(
            "Investigate class imbalance and consider class-weighted training."
        )

    return {
        "summary": " ".join(observations),
        "accuracy_interpretation": (
            f"The model correctly classified approximately "
            f"{accuracy * 100:.2f}% of test observations."
        ),
        "precision_interpretation": f"Weighted precision is {precision:.4f}.",
        "recall_interpretation": f"Weighted recall is {recall:.4f}.",
        "f1_interpretation": f"Weighted F1-score is {f1:.4f}.",
        "recommendations": recommendations
    }


# ==================================================
# 5. REGRESSION INTERPRETATION
# ==================================================

def interpret_regression_results(results):

    mae = results["mae"]
    rmse = results["rmse"]
    r2 = results["r2_score"]

    observations = []
    recommendations = []

    if r2 is None:
        observations.append("R² could not be calculated.")
        r2_interpretation = "R² is unavailable for this test dataset."
    elif r2 >= 0.80:
        observations.append(
            "The model explains a large proportion of target variation."
        )
        r2_interpretation = (
            f"R² is {r2:.4f}. The model explains approximately "
            f"{r2 * 100:.2f}% of target variation."
        )
    elif r2 >= 0.50:
        observations.append(
            "The model explains a moderate proportion of target variation."
        )
        r2_interpretation = (
            f"R² is {r2:.4f}. The model explains approximately "
            f"{r2 * 100:.2f}% of target variation."
        )
    elif r2 >= 0:
        observations.append(
            "The model explains only a limited proportion of target variation."
        )
        r2_interpretation = (
            f"R² is {r2:.4f}. Consider exploring additional predictive features."
        )
        recommendations.append(
            "Explore additional predictive features and alternative algorithms."
        )
    else:
        observations.append(
            "The model performs worse than the test-set mean baseline."
        )
        r2_interpretation = (
            f"R² is {r2:.4f}. Review feature relevance and model settings."
        )
        recommendations.append(
            "Review feature relevance, target quality and model settings."
        )

    if rmse > 1.5 * mae:
        observations.append(
            "RMSE is considerably higher than MAE, suggesting some large errors."
        )
        recommendations.append(
            "Inspect prediction errors for unusually large mistakes."
        )

    return {
        "summary": " ".join(observations),
        "mae_interpretation": (
            f"Predictions differ from actual values by approximately "
            f"{mae:.4f} units on average."
        ),
        "rmse_interpretation": (
            f"RMSE is {rmse:.4f}. It gives greater weight to larger errors."
        ),
        "r2_interpretation": r2_interpretation,
        "recommendations": recommendations
    }


# ==================================================
# 6. NEURAL NETWORK TRAINING
# ==================================================

def train_neural_network(
    X_train,
    y_train,
    X_test,
    y_test,
    task="classification",
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    patience=7,
    min_delta=0.0001
):

    if len(X_train) < 5:
        raise ValueError(
            "At least 5 training observations are required."
        )

    indices = np.arange(len(X_train))

    stratify_values = None

    if task == "classification":
        counts = pd.Series(y_train).value_counts()

        if (
            counts.min() >= 2
            and int(len(indices) * 0.2) >= len(counts)
            and int(len(indices) * 0.8) >= len(counts)
        ):
            stratify_values = np.asarray(y_train)

    fit_indices, validation_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=stratify_values
    )

    X_fit = X_train.iloc[fit_indices]
    X_validation = X_train.iloc[validation_indices]

    y_fit = np.asarray(y_train)[fit_indices]
    y_validation = np.asarray(y_train)[validation_indices]

    if task == "classification":
        if not np.array_equal(
            np.unique(y_train),
            np.unique(y_fit)
        ):
            raise ValueError(
                "Internal training split is missing a target class. "
                "Use more observations per class."
            )

    (
        X_fit_processed,
        X_validation_processed,
        X_test_processed,
        preprocessor
    ) = prepare_features(
        X_fit,
        X_validation,
        X_test
    )

    X_fit_tensor = torch.tensor(
        X_fit_processed,
        dtype=torch.float32
    )

    X_validation_tensor = torch.tensor(
        X_validation_processed,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test_processed,
        dtype=torch.float32
    )

    if task == "classification":

        y_fit_tensor = torch.tensor(y_fit, dtype=torch.long)

        y_validation_tensor = torch.tensor(
            y_validation,
            dtype=torch.long
        )

        y_test_array = np.asarray(y_test, dtype=np.int64)

        output_size = len(np.unique(y_train))

        criterion = nn.CrossEntropyLoss()

    else:

        y_fit_tensor = torch.tensor(
            np.asarray(y_fit, dtype=np.float32).reshape(-1, 1)
        )

        y_validation_tensor = torch.tensor(
            np.asarray(y_validation, dtype=np.float32).reshape(-1, 1)
        )

        y_test_array = np.asarray(y_test, dtype=np.float32)

        output_size = 1

        criterion = nn.MSELoss()

    dataset = TensorDataset(
        X_fit_tensor,
        y_fit_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    torch.manual_seed(42)

    model = TabularNeuralNetwork(
        input_size=X_fit_tensor.shape[1],
        output_size=output_size,
        task=task
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = []

    # Monitor validation loss for early stopping. The held-out test set
    # is not used for checkpoint selection or stopping decisions.
    best_validation_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    patience = max(1, int(patience))
    min_delta = max(0.0, float(min_delta))

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for batch_X, batch_y in loader:

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / max(1, len(loader))

        model.eval()
        with torch.no_grad():
            validation_outputs = model(X_validation_tensor)
            validation_loss = criterion(
                validation_outputs,
                y_validation_tensor
            ).item()

        history.append({
            "epoch": epoch + 1,
            "loss": average_loss,
            "validation_loss": validation_loss
        })

        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_model_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            stopped_early = True
            break

    if best_model_state is None:
        raise RuntimeError(
            "Training did not produce a valid validation checkpoint."
        )

    # Restore the best validation checkpoint before test evaluation.
    model.load_state_dict(best_model_state)
    model.eval()

    for item in history:
        item["best_epoch"] = best_epoch
        item["epochs_trained"] = len(history)
        item["stopped_early"] = stopped_early
        item["best_validation_loss"] = best_validation_loss

    with torch.no_grad():
        test_outputs = model(X_test_tensor)

        if task == "classification":
            predictions = torch.argmax(
                test_outputs,
                dim=1
            ).numpy()
        else:
            predictions = test_outputs.numpy().flatten()

    return model, predictions, history, preprocessor


# ==================================================
# 7. PYTORCH CLASSIFICATION
# ==================================================

def train_pytorch_classification(
    dataframe,
    target_column,
    split_data=None,
    test_size=0.2,
    epochs=50,
    patience=7,
    min_delta=0.0001
):

    if split_data is None:
        raise ValueError(
            "Pass the scikit-learn training output using split_data."
        )

    X_train = split_data["X_train"]
    X_test = split_data["X_test"]

    y_train = split_data["y_train"]
    y_test = split_data["y_test"]

    label_encoder = LabelEncoder()

    label_encoder.fit(y_train)

    y_train_encoded = label_encoder.transform(y_train)

    try:
        y_test_encoded = label_encoder.transform(y_test)
    except ValueError as error:
        raise ValueError(
            "Test data contains a class absent from training data."
        ) from error

    if len(np.unique(y_train_encoded)) < 2:
        raise ValueError("Training data must contain at least two classes.")

    model, predictions, history, preprocessor = train_neural_network(
        X_train,
        y_train_encoded,
        X_test,
        y_test_encoded,
        task="classification",
        epochs=epochs,
        patience=patience,
        min_delta=min_delta
    )

    results = {
        "accuracy": accuracy_score(y_test_encoded, predictions),
        "precision": precision_score(
            y_test_encoded,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_test_encoded,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test_encoded,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

    return {
        "model": model,
        "preprocessor": preprocessor,
        "results": results,
        "history": history,
        "predictions": label_encoder.inverse_transform(predictions),
        "y_test": np.asarray(y_test),
        "classes": label_encoder.classes_,
        "target_column": target_column,
        "interpretation": interpret_classification_results(results),
        "training_interpretation": interpret_training_history(history)
    }


# ==================================================
# 8. PYTORCH REGRESSION
# ==================================================

def train_pytorch_regression(
    dataframe,
    target_column,
    split_data=None,
    test_size=0.2,
    epochs=50,
    patience=7,
    min_delta=0.0001
):

    if split_data is None:
        raise ValueError(
            "Pass the scikit-learn training output using split_data."
        )

    X_train = split_data["X_train"]
    X_test = split_data["X_test"]

    y_train = pd.to_numeric(
        split_data["y_train"],
        errors="coerce"
    )

    y_test = pd.to_numeric(
        split_data["y_test"],
        errors="coerce"
    )

    if not np.isfinite(y_train).all():
        raise ValueError("Training target contains non-finite values.")

    if not np.isfinite(y_test).all():
        raise ValueError("Testing target contains non-finite values.")

    if y_train.nunique() < 2:
        raise ValueError("Regression target must contain varying values.")

    model, predictions, history, preprocessor = train_neural_network(
        X_train,
        y_train.to_numpy(dtype=np.float32),
        X_test,
        y_test.to_numpy(dtype=np.float32),
        task="regression",
        epochs=epochs,
        patience=patience,
        min_delta=min_delta
    )

    mse = mean_squared_error(y_test, predictions)

    results = {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2_score": (
            r2_score(y_test, predictions)
            if len(y_test) >= 2
            else None
        )
    }

    return {
        "model": model,
        "preprocessor": preprocessor,
        "results": results,
        "history": history,
        "predictions": predictions,
        "y_test": y_test.to_numpy(),
        "target_column": target_column,
        "interpretation": interpret_regression_results(results),
        "training_interpretation": interpret_training_history(history)
    }
