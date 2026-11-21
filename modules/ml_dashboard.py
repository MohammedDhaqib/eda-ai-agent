import streamlit as st

import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

import torch

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve

from modules.classification import train_classification_models

from modules.regression import train_regression_models

from modules.ml_clustering import perform_clustering

from modules.anomaly_detection import detect_anomalies

from modules.ml_compatibility import inspect_target

from modules.pytorch_model import (
    train_pytorch_classification,

    train_pytorch_regression
)

# ============================================================

# 1. METRIC CONFIGURATION

# ============================================================


METRIC_NAMES = {

    "accuracy": "Accuracy",

    "precision": "Precision",

    "recall": "Recall",

    "f1_score": "F1 Score",

    "roc_auc": "ROC-AUC",

    "mae": "MAE",

    "mse": "MSE",

    "rmse": "RMSE",

    "r2_score": "R² Score",

    "r2": "R² Score"

}

CLASSIFICATION_METRICS = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1 Score",

    "ROC-AUC"

]

REGRESSION_METRICS = [

    "MAE",

    "MSE",

    "RMSE",

    "R² Score"

]


# ============================================================

# 2. TARGET COMPATIBILITY

# ============================================================


def display_compatibility(dataframe, target_column, task):
    """Check whether a target column is suitable for a task."""

    inspection = inspect_target(

        dataframe,

        target_column

    )

    if task == "Classification":

        supported = inspection.get(

            "classification_supported",

            False

        )

        reasons = inspection.get(

            "classification_reasons",

            []

        )



    else:

        supported = inspection.get(

            "regression_supported",

            False

        )

        reasons = inspection.get(

            "regression_reasons",

            []

        )

    warnings = inspection.get("warnings", [])

    st.subheader("Target Compatibility")

    data_type = inspection.get(

        "data_type",

        inspection.get("dtype", "Unknown")

    )

    st.write(f"**Target column:** {target_column}")

    st.write(f"**Detected data type:** {data_type}")

    if supported:

        st.success(

            f"This target is suitable for {task.lower()}."

        )



    else:

        st.error(

            f"This target is not currently suitable for {task.lower()}."

        )

    for reason in reasons:
        st.warning(str(reason))

    for warning in warnings:
        st.info(str(warning))

    return supported


# ============================================================

# 3. SAFE METRIC HANDLING

# ============================================================


def safe_metric(value):
    """Convert metric values into safe numeric values."""

    if value is None:
        return np.nan

    try:

        numeric_value = float(value)

        if not np.isfinite(numeric_value):
            return np.nan

        return numeric_value



    except (TypeError, ValueError):

        return np.nan


def format_metric_name(metric):
    """Normalize metric names from both sklearn and PyTorch."""

    key = str(metric).strip().lower()

    aliases = {

        "accuracy": "Accuracy",

        "accuracy_score": "Accuracy",

        "precision": "Precision",

        "precision_score": "Precision",

        "weighted_precision": "Precision",

        "recall": "Recall",

        "recall_score": "Recall",

        "weighted_recall": "Recall",

        "f1": "F1 Score",

        "f1_score": "F1 Score",

        "weighted_f1": "F1 Score",

        "roc_auc": "ROC-AUC",

        "roc_auc_score": "ROC-AUC",

        "mae": "MAE",

        "mean_absolute_error": "MAE",

        "mse": "MSE",

        "mean_squared_error": "MSE",

        "rmse": "RMSE",

        "root_mean_squared_error": "RMSE",

        "r2": "R² Score",

        "r2_score": "R² Score",

        "r² score": "R² Score"

    }

    return aliases.get(

        key,

        str(metric).replace("_", " ").title()

    )


# ============================================================

# 4. EXTRACT MODEL METRICS

# ============================================================


def extract_model_metrics(model_output):
    """

    Extract metrics from nested classification results

    or flat regression results.

    """

    if not isinstance(model_output, dict):
        return {}

    # Classification structure:

    # {

    #     "model": ...,

    #     "metrics": {...},

    #     "predictions": ...,

    #     ...

    # }

    if isinstance(model_output.get("metrics"), dict):

        metrics = model_output["metrics"]



    else:

        # Regression may return metrics directly.

        metrics = model_output

    excluded_keys = {

        "model",

        "predictions",

        "confusion_matrix",

        "classification_report",

        "history",

        "preprocessor",

        "interpretation",

        "training_interpretation",

        "classes",

        "target_column",

        "y_test",

        "best_params",

        "best_cv_rmse",

        "selected_features",

        "selected_feature_count",

        "total_preprocessed_features",

        "best_epoch",

        "epochs_trained",

        "stopped_early"

    }

    extracted = {}

    for metric, value in metrics.items():

        if metric in excluded_keys:
            continue

        display_name = format_metric_name(metric)

        numeric_value = safe_metric(value)

        if not np.isnan(numeric_value):
            extracted[display_name] = numeric_value

    return extracted


# ============================================================

# 5. CREATE MODEL COMPARISON TABLE

# ============================================================


def create_comparison_dataframe(results, task):
    """

    Create a clean comparison table.



    Supports nested classification results and

    flat regression results.

    """

    rows = []

    if not isinstance(results, dict):
        return pd.DataFrame()

    for model_name, model_output in results.items():

        if not isinstance(model_output, dict):
            continue

        if "error" in model_output:
            st.warning(

                f"{model_name} failed: {model_output['error']}"

            )

            continue

        metrics = extract_model_metrics(model_output)

        if not metrics:
            continue

        row = {"Model": model_name}

        row.update(metrics)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    comparison = pd.DataFrame(rows)

    comparison = comparison.set_index("Model")

    if task == "Classification":

        ordered_columns = [

            column

            for column in CLASSIFICATION_METRICS

            if column in comparison.columns

        ]



    else:

        ordered_columns = [

            column

            for column in REGRESSION_METRICS

            if column in comparison.columns

        ]

    comparison = comparison.reindex(

        columns=ordered_columns

    )

    return comparison


# ============================================================

# 6. ADD PYTORCH TO MODEL COMPARISON

# ============================================================


def add_pytorch_to_comparison(

        comparison,

        pytorch_results,

        task

):
    """Add PyTorch metrics to the existing comparison table."""

    if not isinstance(pytorch_results, dict):
        return comparison

    pytorch_metrics = extract_model_metrics(

        pytorch_results

    )

    if not pytorch_metrics:
        return comparison

    pytorch_row = {

        metric: pytorch_metrics.get(metric, np.nan)

        for metric in (

            CLASSIFICATION_METRICS

            if task == "Classification"

            else REGRESSION_METRICS

        )

    }

    pytorch_dataframe = pd.DataFrame(

        [pytorch_row],

        index=["PyTorch Neural Network"]

    )

    if comparison is None or comparison.empty:
        return pytorch_dataframe

    comparison = pd.concat(

        [comparison, pytorch_dataframe],

        axis=0

    )

    return comparison


# ============================================================

# 7. DISPLAY MODEL COMPARISON

# ============================================================


def display_model_comparison(comparison, task):
    """Display a formatted model comparison table."""

    if comparison is None or comparison.empty:
        st.warning("No model comparison results are available.")

        return

    st.subheader(f"{task} Model Comparison")

    formatted_table = comparison.style.format(

        "{:.4f}",

        na_rep="N/A"

    )

    st.dataframe(

        formatted_table,

        use_container_width=True

    )


# ============================================================

# 8. MODEL COMPARISON INTERPRETATION

# ============================================================


def display_comparison_interpretation(comparison, task):
    """Identify the best-performing model for each metric."""

    if comparison is None or comparison.empty:
        return

    st.subheader("Model Comparison Interpretation")

    if task == "Classification":

        preferences = {

            "Accuracy": "higher",

            "Precision": "higher",

            "Recall": "higher",

            "F1 Score": "higher",

            "ROC-AUC": "higher"

        }



    else:

        preferences = {

            "MAE": "lower",

            "MSE": "lower",

            "RMSE": "lower",

            "R² Score": "higher"

        }

    for metric, direction in preferences.items():

        if metric not in comparison.columns:
            continue

        values = pd.to_numeric(

            comparison[metric],

            errors="coerce"

        ).dropna()

        if values.empty:
            continue

        if direction == "higher":

            best_model = values.idxmax()



        else:

            best_model = values.idxmin()

        best_value = values.loc[best_model]

        st.write(

            f"**{metric}:** {best_model} performed best "

            f"with a score of {best_value:.4f}."

        )

    if task == "Classification":

        st.caption(

            "Higher accuracy, precision, recall, F1 Score and "

            "ROC-AUC generally indicate better classification "

            "performance."

        )



    else:

        st.caption(

            "Lower MAE, MSE and RMSE generally indicate smaller "

            "prediction errors. Higher R² generally indicates "

            "that the model explains more target variation."

        )


# ============================================================

# 9. PYTORCH PERFORMANCE INTERPRETATION

# ============================================================


def display_pytorch_interpretation(interpretation):
    """Display PyTorch performance interpretation."""

    if not isinstance(interpretation, dict):
        return

    st.subheader("PyTorch Model Interpretation")

    summary = interpretation.get("summary")

    if summary:
        st.info(summary)

    display_names = {

        "accuracy_interpretation": "Accuracy",

        "precision_interpretation": "Precision",

        "recall_interpretation": "Recall",

        "f1_interpretation": "F1 Score",

        "mae_interpretation": "MAE",

        "rmse_interpretation": "RMSE",

        "r2_interpretation": "R² Score"

    }

    for key, display_name in display_names.items():

        explanation = interpretation.get(key)

        if explanation:
            st.write(

                f"**{display_name} Interpretation:** {explanation}"

            )

    recommendations = interpretation.get(

        "recommendations",

        []

    )

    if recommendations:

        st.markdown("**Recommendations**")

        for recommendation in recommendations:
            st.write(f"- {recommendation}")


# ============================================================

# 10. TRAINING INTERPRETATION

# ============================================================


def display_training_interpretation(training_interpretation):
    """Explain neural-network training behaviour."""

    if not isinstance(training_interpretation, dict):
        return

    st.subheader("Neural Network Training Analysis")

    status = training_interpretation.get("status")

    explanation = training_interpretation.get("explanation")

    overfitting_status = training_interpretation.get(

        "overfitting_status"

    )

    validation_explanation = training_interpretation.get(

        "validation_explanation"

    )

    if status:
        st.write(f"**Training Status:** {status}")

    if explanation:
        st.info(explanation)

    if overfitting_status:
        st.write(

            f"**Overfitting Assessment:** {overfitting_status}"

        )

    if validation_explanation:
        st.write(

            f"**Validation Analysis:** {validation_explanation}"

        )

    recommendations = training_interpretation.get(

        "recommendations",

        []

    )

    if recommendations:

        st.markdown("**Suggested Improvements**")

        for recommendation in recommendations:
            st.write(f"- {recommendation}")


# ============================================================

# 11. PYTORCH TRAINING HISTORY GRAPH

# ============================================================


def display_training_history(history):
    """

    Display the training/validation loss graph and explain what it means

    in beginner-friendly language.

    """

    if history is None or len(history) == 0:
        st.warning("Training history is unavailable.")

        return

    # Convert either supported history format into a DataFrame.

    if isinstance(history, list):

        history_data = pd.DataFrame(history).rename(

            columns={

                "epoch": "Epoch",

                "loss": "Training Loss",

                "validation_loss": "Validation Loss"

            }

        )



    elif isinstance(history, dict):

        training_loss = history.get("loss", [])

        validation_loss = history.get("validation_loss", [])

        if not training_loss:
            st.warning("Training loss data is unavailable.")

            return

        history_data = pd.DataFrame({

            "Epoch": range(1, len(training_loss) + 1),

            "Training Loss": training_loss

        })

        if len(validation_loss) == len(training_loss):
            history_data["Validation Loss"] = validation_loss



    else:

        st.warning("Unrecognized training history format.")

        return

    required_columns = ["Epoch", "Training Loss"]

    if history_data.empty or not all(

            column in history_data.columns

            for column in required_columns

    ):
        st.warning("Training history is missing required values.")

        return

    plot_columns = [

        column for column in ["Training Loss", "Validation Loss"]

        if column in history_data.columns

    ]

    history_data["Epoch"] = pd.to_numeric(

        history_data["Epoch"], errors="coerce"

    )

    for column in plot_columns:
        history_data[column] = pd.to_numeric(

            history_data[column], errors="coerce"

        )

    history_data = history_data.dropna(

        subset=["Epoch"] + plot_columns

    )

    if history_data.empty:
        st.warning("No valid numeric loss values are available.")

        return

    history_data = history_data.sort_values("Epoch")

    st.subheader("PyTorch Training History")

    figure = px.line(

        history_data,

        x="Epoch",

        y=plot_columns,

        markers=True,

        title="Training vs Validation Loss"

    )

    figure.update_layout(

        xaxis_title="Training Epoch",

        yaxis_title="Loss (lower is generally better)",

        legend_title="Loss Type",

        hovermode="x unified"

    )

    st.plotly_chart(figure, use_container_width=True)

    st.caption(

        "An epoch is one complete pass through the neural network's "

        "training data. Loss is the model's prediction error: lower "

        "values generally mean better predictions."

    )

    # Explain the curve using its actual first and last values.

    st.markdown("### How to read this graph")

    training_values = history_data["Training Loss"].to_numpy()

    first_train = float(training_values[0])

    last_train = float(training_values[-1])

    train_change = last_train - first_train

    st.write(

        f"- **Training loss:** changed from {first_train:.4f} "

        f"at the beginning to {last_train:.4f} at the end."

    )

    if "Validation Loss" not in history_data.columns:
        st.info(

            "Only training loss was recorded. A validation-loss curve "

            "is needed to assess how well the model generalizes to "

            "unseen data during training."

        )

        return

    validation_values = history_data["Validation Loss"].to_numpy()

    first_val = float(validation_values[0])

    last_val = float(validation_values[-1])

    val_change = last_val - first_val

    st.write(

        f"- **Validation loss:** changed from {first_val:.4f} "

        f"at the beginning to {last_val:.4f} at the end."

    )

    # Small tolerance prevents tiny numerical fluctuations from being

    # described as meaningful improvement or deterioration.

    tolerance = max(abs(first_train), abs(first_val), 1.0) * 0.01

    train_improved = train_change < -tolerance

    train_worsened = train_change > tolerance

    validation_improved = val_change < -tolerance

    validation_worsened = val_change > tolerance

    if train_improved and validation_improved:

        st.success(

            " **Good sign — the model is learning and generalizing.** "

            "Both lines went down. The model reduced its error on the "

            "data it learned from and on the separate validation data."

        )

        st.write(

            "**What this means:** The model appears to be learning "

            "useful patterns that also help on data it did not train on."

        )



    elif train_improved and validation_worsened:

        st.warning(

            "**Possible overfitting.** Training loss went down, but "

            "validation loss went up."

        )

        st.write(

            "**What this means:** The model is getting better at the "

            "training examples but worse on unseen validation examples. "

            "It may be memorizing training patterns instead of learning "

            "patterns that generalize."

        )

        st.write(

            "**What to try:** Reduce the number of epochs, use stronger "

            "regularization, simplify the network, or collect more data."

        )



    elif train_improved and not validation_worsened:

        st.info(

            "**Some learning is happening.** Training loss decreased, "

            "while validation loss changed only slightly."

        )

        st.write(

            "**What this means:** The model learned the training data, "

            "but its improvement on unseen data is limited so far. "

            "More epochs may or may not help; watch the validation curve."

        )



    elif train_worsened or validation_worsened:

        st.warning(

            "**The loss curves need attention.** At least one loss "

            "increased overall instead of improving."

        )

        st.write(

            "**What this means:** The model may not be learning reliably. "

            "Consider checking the learning rate, feature quality, target "

            "values, and dataset size."

        )



    else:

        st.info(

            "**The model appears to have reached a plateau.** "

            "Neither loss changed much between the first and last epoch."

        )

        st.write(

            "**What this means:** Training may have stopped making "

            "meaningful progress. Different model settings or features "

            "may be needed."

        )

    # Explain the distance between the two curves at the end.

    final_gap = last_val - last_train

    if final_gap > tolerance:

        st.write(

            f"**End-of-training gap:** validation loss is {final_gap:.4f} "

            "higher than training loss. A noticeable gap can indicate "

            "that unseen data is harder for the model."

        )

    elif final_gap < -tolerance:

        st.write(

            f"**End-of-training gap:** validation loss is "

            f"{abs(final_gap):.4f} lower than training loss. This can "

            "happen, for example, because dropout is active during "

            "training; it is not automatically a problem."

        )

    else:

        st.write(

            "**End-of-training gap:** the two losses are close, which "

            "means their final error levels are similar."

        )

    st.caption(

        "Important: the graph is a diagnostic aid, not proof of model "

        "quality. Use the final test metrics above to judge performance "

        "on the held-out test dataset."

    )


# ============================================================

# OPTIONAL ROC CURVE ANALYSIS

# ============================================================


def display_optional_roc_analysis(sklearn_results, pytorch_output, X_test, y_test):
    """Display optional One-vs-Rest ROC analysis and threshold metrics."""

    show_roc = st.checkbox(

        "Show optional ROC Curve Analysis",

        value=False,

        help="Display a ROC curve and explore the classification threshold.",

        key="show_optional_roc"

    )

    if not show_roc:
        return

    model_probabilities = {}

    for model_name, model_output in sklearn_results.items():

        if not isinstance(model_output, dict) or "error" in model_output:
            continue

        fitted_pipeline = model_output.get("model")

        if fitted_pipeline is None or not hasattr(fitted_pipeline, "predict_proba"):
            continue

        try:

            probabilities = fitted_pipeline.predict_proba(X_test)

            classes = np.asarray(fitted_pipeline.classes_)

            if probabilities.ndim == 2 and probabilities.shape[1] == len(classes):
                model_probabilities[model_name] = {

                    "probabilities": probabilities,

                    "classes": classes

                }

        except Exception as error:

            st.warning(f"Could not obtain ROC probabilities for {model_name}: {error}")

    if isinstance(pytorch_output, dict):

        pytorch_model = pytorch_output.get("model")

        pytorch_preprocessor = pytorch_output.get("preprocessor")

        pytorch_classes = pytorch_output.get("classes")

        if (

                pytorch_model is not None

                and pytorch_preprocessor is not None

                and pytorch_classes is not None

        ):

            try:

                processed_test = pytorch_preprocessor.transform(X_test)

                processed_test = np.nan_to_num(

                    np.asarray(processed_test, dtype=np.float32)

                )

                test_tensor = torch.tensor(processed_test, dtype=torch.float32)

                pytorch_model.eval()

                with torch.no_grad():

                    logits = pytorch_model(test_tensor)

                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()

                pytorch_classes = np.asarray(pytorch_classes)

                if probabilities.shape[1] == len(pytorch_classes):
                    model_probabilities["PyTorch Neural Network"] = {

                        "probabilities": probabilities,

                        "classes": pytorch_classes

                    }

            except Exception as error:

                st.warning(f"Could not obtain ROC probabilities for PyTorch: {error}")

    if not model_probabilities:
        st.warning("No models produced usable class probabilities.")

        return

    model_name = st.selectbox(

        "Select a model for ROC analysis",

        list(model_probabilities.keys()),

        key="roc_model_selection"

    )

    selected_model = model_probabilities[model_name]

    probabilities = selected_model["probabilities"]

    classes = selected_model["classes"]

    if len(classes) < 2:
        st.warning("ROC analysis requires at least two classes.")

        return

    class_labels = [str(value) for value in classes]

    selected_class_label = st.selectbox(

        "Select the class to evaluate",

        class_labels,

        key="roc_class_selection"

    )

    class_index = class_labels.index(selected_class_label)

    selected_class = classes[class_index]

    y_binary = (np.asarray(y_test) == selected_class).astype(int)

    positive_count = int(y_binary.sum())

    negative_count = int(len(y_binary) - positive_count)

    if positive_count == 0 or negative_count == 0:
        st.warning(

            "The selected class needs both positive and negative "

            "examples in the test data. ROC cannot be calculated "

            "for this class with the current test split."

        )

        return

    if class_index >= probabilities.shape[1]:
        st.warning("The selected class has no matching probability column.")

        return

    # AUC summarizes ranking performance across all thresholds.

    # The slider below selects one operating threshold for predictions.

    selected_probabilities = probabilities[:, class_index]

    false_positive_rate, true_positive_rate, _ = roc_curve(

        y_binary,

        selected_probabilities

    )

    roc_auc_value = auc(false_positive_rate, true_positive_rate)

    st.subheader("Choose a Classification Threshold")

    st.write(

        "The threshold decides when the selected class is predicted. "

        "For example, at 0.50 the model predicts this class when its "

        "probability is at least 50%."

    )

    threshold = st.slider(

        "Classification threshold",

        min_value=0.00,

        max_value=1.00,

        value=0.50,

        step=0.01,

        key="roc_classification_threshold",

        help=(

            "Lower thresholds usually catch more positive cases but may "

            "create more false positives. Higher thresholds usually reduce "

            "false positives but may miss more positive cases."

        )

    )

    predicted_positive = (selected_probabilities >= threshold).astype(int)

    true_positive = int(np.sum((y_binary == 1) & (predicted_positive == 1)))

    false_positive = int(np.sum((y_binary == 0) & (predicted_positive == 1)))

    true_negative = int(np.sum((y_binary == 0) & (predicted_positive == 0)))

    false_negative = int(np.sum((y_binary == 1) & (predicted_positive == 0)))

    accuracy = (

        (true_positive + true_negative) / len(y_binary)

        if len(y_binary) else 0.0

    )

    precision = (

        true_positive / (true_positive + false_positive)

        if (true_positive + false_positive) else 0.0

    )

    recall = (

        true_positive / (true_positive + false_negative)

        if (true_positive + false_negative) else 0.0

    )

    specificity = (

        true_negative / (true_negative + false_positive)

        if (true_negative + false_positive) else 0.0

    )

    f1_score = (

        2 * precision * recall / (precision + recall)

        if (precision + recall) else 0.0

    )

    operating_fpr = (

        false_positive / (false_positive + true_negative)

        if (false_positive + true_negative) else 0.0

    )

    operating_tpr = recall

    figure = go.Figure()

    figure.add_trace(

        go.Scatter(

            x=false_positive_rate,

            y=true_positive_rate,

            mode="lines",

            name=f"{selected_class_label} vs Rest (AUC = {roc_auc_value:.4f})"

        )

    )

    figure.add_trace(

        go.Scatter(

            x=[0, 1],

            y=[0, 1],

            mode="lines",

            name="Random Classifier",

            line={"dash": "dash"}

        )

    )

    figure.add_trace(

        go.Scatter(

            x=[operating_fpr],

            y=[operating_tpr],

            mode="markers",

            name=f"Selected threshold = {threshold:.2f}",

            marker={"size": 12, "symbol": "circle"}

        )

    )

    figure.update_layout(

        title=f"ROC Curve — {model_name}",

        xaxis_title="False Positive Rate",

        yaxis_title="True Positive Rate",

        xaxis={"range": [0, 1]},

        yaxis={"range": [0, 1]},

        legend_title="ROC Analysis",

        hovermode="closest"

    )

    st.plotly_chart(figure, use_container_width=True)

    st.metric("ROC-AUC Score (unchanged by threshold)", f"{roc_auc_value:.4f}")

    st.subheader(f"Performance at Threshold {threshold:.2f}")

    metric_columns = st.columns(5)

    metric_columns[0].metric("Accuracy", f"{accuracy:.3f}")

    metric_columns[1].metric("Precision", f"{precision:.3f}")

    metric_columns[2].metric("Recall", f"{recall:.3f}")

    metric_columns[3].metric("Specificity", f"{specificity:.3f}")

    metric_columns[4].metric("F1 Score", f"{f1_score:.3f}")

    st.subheader("Prediction Breakdown")

    confusion_columns = st.columns(4)

    confusion_columns[0].metric("True Positives", f"{true_positive:,}")

    confusion_columns[1].metric("False Positives", f"{false_positive:,}")

    confusion_columns[2].metric("True Negatives", f"{true_negative:,}")

    confusion_columns[3].metric("False Negatives", f"{false_negative:,}")

    st.subheader("Threshold Interpretation")

    st.write(

        f"At threshold **{threshold:.2f}**, the model labels an observation "

        f"as **{selected_class_label}** when its predicted probability for "

        "that class is at least the selected threshold."

    )

    if recall > specificity:

        st.write(

            "At this threshold, recall is higher than specificity: the model "

            "catches a larger share of the selected class, but may also flag "

            "more observations from other classes."

        )

    elif specificity > recall:

        st.write(

            "At this threshold, specificity is higher than recall: the model "

            "avoids more false alarms, but may miss more observations from "

            "the selected class."

        )

    else:

        st.write(

            "At this threshold, recall and specificity are equal."

        )

    st.caption(

        "The highlighted point on the ROC curve shows the selected threshold. "

        "Changing the threshold changes the confusion-matrix metrics and the "

        "highlighted operating point, but it does not change ROC-AUC. "

        "For multiclass targets, this is a One-vs-Rest calculation: the "

        "selected class is positive and all remaining classes are negative. "

        "Metrics are calculated on the held-out test set."

    )

    if roc_auc_value >= 0.90:

        rating = "Excellent discrimination"

        explanation = "The model separates this class from the remaining classes very well across thresholds."

    elif roc_auc_value >= 0.80:

        rating = "Good discrimination"

        explanation = "The model generally distinguishes this class from the remaining classes well."

    elif roc_auc_value >= 0.70:

        rating = "Fair discrimination"

        explanation = "The model has some ability to distinguish this class, but there is room for improvement."

    elif roc_auc_value > 0.50:

        rating = "Weak discrimination"

        explanation = "The model performs better than random ranking, but its ability to distinguish this class is limited."

    elif np.isclose(roc_auc_value, 0.50):

        rating = "About random"

        explanation = "The model's ranking performance is approximately the same as random guessing for this one-vs-rest task."

    else:

        rating = "Below random ranking"

        explanation = "The model ranks positive examples below negative examples more often than expected. Check the class mapping and model."

    st.subheader("ROC Curve Interpretation")

    st.write(f"**Assessment:** {rating}")

    st.write(explanation)

    st.write(

        f"For this calculation, **{positive_count}** test observations "

        f"belonged to class **{selected_class_label}**, and "

        f"**{negative_count}** belonged to the remaining classes."

    )


# ============================================================

# 12. CLASSIFICATION

# ============================================================


def run_classification(

        dataframe,

        target_column,

        test_size,

        epochs

):
    st.header("Classification")

    if not display_compatibility(

            dataframe,

            target_column,

            "Classification"

    ):
        return

    # Keep trained results in Streamlit session state. Without this cache,

    # changing the ROC checkbox or its dropdowns would rerun the app and

    # hide the results because the Train button is no longer clicked.

    dataset_signature = (

        f"{len(dataframe)}|{len(dataframe.columns)}|"

        f"{'|'.join(map(str, dataframe.columns))}"

    )

    cache_key = (

        f"ml_classification_cache::{target_column}::"

        f"{test_size}::{epochs}::{dataset_signature}"

    )

    train_clicked = st.button(

        "Train Classification Models",

        key="train_classification"

    )

    if train_clicked:

        with st.spinner("Training classification models..."):

            try:

                # ------------------------------------------------

                # SCIKIT-LEARN MODELS

                # ------------------------------------------------

                output = train_classification_models(

                    dataframe,

                    target_column,

                    test_size=test_size,

                    random_state=42

                )

                sklearn_results = output["results"]

                st.success(

                    "Scikit-learn classification models trained."

                )

                comparison = create_comparison_dataframe(

                    sklearn_results,

                    "Classification"

                )

                # ------------------------------------------------

                # PYTORCH MODEL

                # ------------------------------------------------

                st.subheader("PyTorch Neural Network")

                with st.spinner("Training PyTorch neural network..."):

                    pytorch_output = train_pytorch_classification(

                        dataframe,

                        target_column,

                        split_data=output,

                        epochs=epochs

                    )

                st.success(

                    "PyTorch classification model trained."

                )

                # Save the fitted models and test split so interactive

                # ROC controls continue working after Streamlit reruns.

                st.session_state[cache_key] = {

                    "output": output,

                    "pytorch_output": pytorch_output

                }



            except Exception as error:

                st.error(

                    f"Classification training failed: {error}"

                )

                return

    cached_results = st.session_state.get(cache_key)

    if cached_results is None:
        st.info(

            "Choose the target column and click "

            "'Train Classification Models' to begin."

        )

        return

    output = cached_results["output"]

    pytorch_output = cached_results["pytorch_output"]

    sklearn_results = output["results"]

    # ------------------------------------------------

    # COMBINE MODEL RESULTS

    # ------------------------------------------------

    comparison = create_comparison_dataframe(

        sklearn_results,

        "Classification"

    )

    pytorch_metrics = pytorch_output.get(

        "results",

        {}

    )

    comparison = add_pytorch_to_comparison(

        comparison,

        pytorch_metrics,

        "Classification"

    )

    # ------------------------------------------------

    # DISPLAY COMPARISON

    # ------------------------------------------------

    display_model_comparison(

        comparison,

        "Classification"

    )

    display_comparison_interpretation(

        comparison,

        "Classification"

    )

    # Optional ROC analysis (hidden unless selected by the user).

    # Results remain available when the checkbox/dropdowns are changed.

    display_optional_roc_analysis(

        sklearn_results,

        pytorch_output,

        output["X_test"],

        output["y_test"]

    )

    # ------------------------------------------------

    # PYTORCH INTERPRETATION

    # ------------------------------------------------

    display_pytorch_interpretation(

        pytorch_output.get(

            "interpretation",

            {}

        )

    )

    # ------------------------------------------------

    # TRAINING HISTORY

    # ------------------------------------------------

    display_training_history(

        pytorch_output.get(

            "history",

            []

        )

    )

    display_training_interpretation(

        pytorch_output.get(

            "training_interpretation",

            {}

        )

    )

    st.success(

        "Scikit-learn and PyTorch used the same "

        "outer training and testing observations."

    )


# ============================================================


# 13. REGRESSION

# ============================================================


# ============================================================

# RANDOM FOREST FEATURE IMPORTANCE

# ============================================================



def display_regression_cv_summary(sklearn_results):
    """Display training-only cross-validation stability for regression models."""
    rows = []
    fold_rows = []

    for model_name, model_output in (sklearn_results or {}).items():
        if not isinstance(model_output, dict) or "error" in model_output:
            continue

        cv_summary = model_output.get("cv")
        if not isinstance(cv_summary, dict):
            continue

        row = {"Model": model_name}
        for metric in ("mae", "rmse", "r2"):
            metric_data = cv_summary.get(metric)
            if not isinstance(metric_data, dict):
                continue

            mean_value = safe_metric(metric_data.get("mean"))
            std_value = safe_metric(metric_data.get("std"))
            if np.isfinite(mean_value):
                row[f"{metric.upper()} mean"] = mean_value
            if np.isfinite(std_value):
                row[f"{metric.upper()} std"] = std_value

            scores = metric_data.get("fold_scores", [])
            if isinstance(scores, (list, tuple, np.ndarray)):
                for fold_number, score in enumerate(scores, start=1):
                    score_value = safe_metric(score)
                    if np.isfinite(score_value):
                        fold_rows.append({
                            "Model": model_name,
                            "Metric": metric.upper(),
                            "Fold": fold_number,
                            "Score": score_value
                        })

        if len(row) > 1:
            rows.append(row)

    if not rows:
        st.info(
            "Cross-validation summaries are unavailable for this run. "
            "They are calculated on the training split only."
        )
        return

    st.subheader("Cross-Validation Stability")
    st.caption(
        "These scores come from cross-validation within the training split. "
        "Mean is the average across folds; standard deviation describes "
        "variation between folds. Lower MAE/RMSE and higher R² are generally "
        "preferred. A large standard deviation suggests less stable results."
    )

    summary = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        summary.style.format("{:.4f}", na_rep="N/A"),
        use_container_width=True
    )

    if fold_rows:
        with st.expander("View fold-by-fold cross-validation scores"):
            fold_frame = pd.DataFrame(fold_rows)
            st.dataframe(
                fold_frame.pivot_table(
                    index=["Model", "Metric"],
                    columns="Fold",
                    values="Score"
                ).style.format("{:.4f}", na_rep="N/A"),
                use_container_width=True
            )

    st.caption(
        "Cross-validation results estimate performance variability during "
        "model development; they are not a replacement for the final "
        "held-out test evaluation."
    )


def display_regression_feature_importance(feature_importance):
    """Display the Random Forest's fitted feature importances."""

    if not isinstance(feature_importance, pd.DataFrame) or feature_importance.empty:
        st.info("Feature importance is unavailable for this regression run.")

        return

    st.subheader("Random Forest Feature Importance")

    st.caption(

        "Importance scores describe how much the fitted Random Forest used "

        "each preprocessed feature to reduce prediction error. They are not "

        "causal effects and do not prove that a feature causes Age."

    )

    importance_data = feature_importance.copy().sort_values(

        "Importance", ascending=False

    )

    top_n = min(20, len(importance_data))

    displayed = importance_data.head(top_n).sort_values(

        "Importance", ascending=True

    )

    figure = px.bar(

        displayed,

        x="Importance",

        y="Feature",

        orientation="h",

        title=f"Top {top_n} Features Used by Random Forest",

        labels={

            "Importance": "Relative feature importance",

            "Feature": "Preprocessed feature"

        }

    )

    figure.update_layout(yaxis={"automargin": True})

    st.plotly_chart(figure, use_container_width=True)

    with st.expander("View all feature importance scores"):
        st.dataframe(

            importance_data.reset_index(drop=True),

            use_container_width=True

        )

    st.caption(

        "One-hot encoded categories appear as separate features. "

        "Correlated features can share or redistribute importance. "

        "These scores are model-specific, not causal effects."

    )


def display_identifier_experiment(experiment):
    """Show Random Forest test metrics with and without identifier columns."""

    if not isinstance(experiment, dict):
        return

    if "error" in experiment:
        st.warning(

            "The ID-column comparison could not be completed: "

            + str(experiment["error"])

        )

        return

    with_id = experiment.get("with_id")

    without_id = experiment.get("without_id")

    if not isinstance(with_id, dict) or not isinstance(without_id, dict):
        return

    st.subheader("Identifier Column Check: Random Forest")

    st.caption(

        "Both runs use the same train/test observations and Random Forest "

        "settings. The second run excludes the ID column."

    )

    comparison = pd.DataFrame(

        [with_id, without_id],

        index=["With ID", "Without ID"]

    )

    st.dataframe(

        comparison.style.format("{:.4f}", na_rep="N/A"),

        use_container_width=True

    )

    delta_r2 = without_id.get("R2")

    original_r2 = with_id.get("R2")

    if delta_r2 is not None and original_r2 is not None:

        change = float(delta_r2) - float(original_r2)

        if change > 0.005:

            st.success(

                f"Excluding ID improved test R² by {change:.4f}. "

                "The ID was not helping generalization."

            )

        elif change < -0.005:

            st.info(

                f"Test R² decreased by {abs(change):.4f} after excluding ID. "

                "This does not mean an identifier is a valid real-world "

                "predictor; it may encode dataset ordering or collection "

                "artifacts. Keep it excluded unless its meaning is justified."

            )

        else:

            st.info(

                "Test R² changed very little after excluding ID. "

                "The identifier has little effect on this test result."

            )


def display_regression_tuning(tuned_result):
    """Show feature-selection and tuning details only when requested."""

    show_details = st.checkbox(
        "Show Feature Selection & Hyperparameter Tuning",
        value=False,
        key="show_regression_feature_selection_tuning"
    )

    if not show_details:
        return

    if not isinstance(tuned_result, dict):
        st.info("Feature-selection tuning results are not available.")
        return

    if "error" in tuned_result:
        st.warning(

            "Feature-selection tuning could not be completed: "

            + str(tuned_result["error"])

        )

        return

    st.subheader("Feature Selection & Hyperparameter Tuning")

    st.write(

        "A SelectFromModel step selected features using only the training data. "

        "RandomizedSearchCV then tuned the Random Forest inside 3-fold "

        "cross-validation on the training split. The held-out test split was "

        "not used to choose features or hyperparameters."

    )

    st.metric(

        "Features retained",

        f"{tuned_result.get('selected_feature_count', 0)} / "

        f"{tuned_result.get('total_preprocessed_features', 0)}"

    )

    st.metric(

        "Best cross-validation RMSE",

        f"{tuned_result.get('best_cv_rmse', float('nan')):.4f}"

    )

    with st.expander("View best hyperparameters"):

        st.json(tuned_result.get("best_params", {}))

    selected = tuned_result.get("selected_features", [])

    with st.expander("View features selected by the model"):

        if selected:

            st.dataframe(

                pd.DataFrame({"Selected feature": selected}),

                use_container_width=True

            )

        else:

            st.info("No selected feature names were returned.")

    st.caption(

        "Compare this tuned model's held-out test metrics with the baseline. "

        "Cross-validation RMSE is a training-time selection score; the test "

        "RMSE is the final held-out evaluation. A tuned model is not guaranteed "

        "to outperform the baseline."

    )



def display_tuned_random_forest_comparison(sklearn_results):
    """Show the baseline-vs-tuned comparison only when requested."""

    show_comparison = st.checkbox(
        "Show Baseline vs Tuned Model",
        value=False,
        key="show_baseline_vs_tuned_random_forest"
    )

    if not show_comparison:
        return

    baseline = sklearn_results.get("Random Forest Regressor", {})
    tuned = sklearn_results.get("Tuned Random Forest (Feature Selection)", {})

    st.subheader("Random Forest: Baseline vs Tuned Model")

    if not isinstance(baseline, dict) or "error" in baseline:
        st.warning("Baseline Random Forest results are unavailable.")
        return

    if not isinstance(tuned, dict) or "error" in tuned:
        error_text = (
            tuned.get("error", "Tuned model results are unavailable.")
            if isinstance(tuned, dict)
            else "Tuned model results are unavailable."
        )
        st.warning(f"Could not compare the tuned model: {error_text}")
        return

    rows = []
    for key, label in [("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R²")]:
        base_value = baseline.get(key)
        tuned_value = tuned.get(key)
        rows.append({
            "Metric": label,
            "Baseline Random Forest": (
                float(base_value) if base_value is not None else np.nan
            ),
            "Tuned Random Forest": (
                float(tuned_value) if tuned_value is not None else np.nan
            ),
        })

    comparison = pd.DataFrame(rows).set_index("Metric")
    st.dataframe(
        comparison.style.format("{:.4f}", na_rep="N/A"),
        use_container_width=True
    )

    st.caption(
        "Both models were evaluated on the same held-out test observations. "
        "Feature selection and hyperparameter tuning used only the training split."
    )

    base_rmse = baseline.get("RMSE")
    tuned_rmse = tuned.get("RMSE")
    base_r2 = baseline.get("R2")
    tuned_r2 = tuned.get("R2")

    if base_rmse is not None and tuned_rmse is not None:
        rmse_change = float(tuned_rmse) - float(base_rmse)
        if rmse_change < -1e-9:
            st.success(
                f"The tuned model reduced test RMSE by {abs(rmse_change):.4f} "
                "compared with the baseline."
            )
        elif rmse_change > 1e-9:
            st.warning(
                f"The tuned model increased test RMSE by {rmse_change:.4f}. "
                "For this split, the baseline Random Forest performed better."
            )
        else:
            st.info("Both Random Forest models have the same test RMSE.")

    if base_r2 is not None and tuned_r2 is not None:
        r2_change = float(tuned_r2) - float(base_r2)
        st.write(f"**Change in test R²:** {r2_change:+.4f} (higher is better).")

    st.caption(
        "Use held-out test metrics—not cross-validation RMSE alone—to decide "
        "which model performed better on this test split."
    )



def display_regression_learning_curve(
        baseline_result,
        X_train,
        y_train,
        cache_key
):
    """Display training/validation learning curves with fold variability."""
    show_curve = st.checkbox(
        "Show Learning Curves",
        value=False,
        key="show_regression_learning_curves"
    )

    if not show_curve:
        return

    if not isinstance(baseline_result, dict) or "model" not in baseline_result:
        st.info("A fitted baseline Random Forest is required for learning curves.")
        return

    if X_train is None or y_train is None or len(X_train) < 12:
        st.warning(
            "At least 12 training rows are recommended to calculate "
            "a 3-fold learning curve."
        )
        return

    curve_cache_key = f"{cache_key}::learning_curve_v2"

    if curve_cache_key not in st.session_state:
        with st.spinner("Calculating the Random Forest learning curve..."):
            try:
                train_sizes, train_scores, validation_scores = learning_curve(
                    estimator=baseline_result["model"],
                    X=X_train,
                    y=y_train,
                    cv=3,
                    scoring="neg_root_mean_squared_error",
                    train_sizes=np.linspace(0.25, 1.0, 4),
                    n_jobs=1,
                    shuffle=True,
                    random_state=42
                )

                train_rmse = -np.asarray(train_scores, dtype=float)
                validation_rmse = -np.asarray(validation_scores, dtype=float)

                curve_data = pd.DataFrame({
                    "Training examples": train_sizes,
                    "Training RMSE": np.mean(train_rmse, axis=1),
                    "Training RMSE SD": (
                        np.std(train_rmse, axis=1, ddof=1)
                        if train_rmse.shape[1] > 1
                        else np.zeros(len(train_sizes))
                    ),
                    "Validation RMSE": np.mean(validation_rmse, axis=1),
                    "Validation RMSE SD": (
                        np.std(validation_rmse, axis=1, ddof=1)
                        if validation_rmse.shape[1] > 1
                        else np.zeros(len(train_sizes))
                    )
                })
                st.session_state[curve_cache_key] = curve_data

            except Exception as error:
                st.error(f"Could not calculate the learning curve: {error}")
                return

    curve_data = st.session_state[curve_cache_key]

    st.subheader("Random Forest Learning Curves")
    st.caption(
        "Each point is the mean of 3 training-only cross-validation folds. "
        "The shaded bands show ±1 standard deviation between folds. "
        "Lower RMSE is better. The held-out test set is not used to build "
        "this curve."
    )

    figure = go.Figure()

    for name, mean_col, sd_col in [
        ("Training", "Training RMSE", "Training RMSE SD"),
        ("Validation", "Validation RMSE", "Validation RMSE SD")
    ]:
        x_values = curve_data["Training examples"].to_numpy(dtype=float)
        means = curve_data[mean_col].to_numpy(dtype=float)
        deviations = curve_data[sd_col].to_numpy(dtype=float)
        lower = np.maximum(0, means - deviations)
        upper = means + deviations

        figure.add_trace(go.Scatter(
            x=x_values,
            y=upper,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
            name=f"{name} variability upper"
        ))
        figure.add_trace(go.Scatter(
            x=x_values,
            y=lower,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(100, 100, 100, 0.12)",
            showlegend=False,
            hoverinfo="skip",
            name=f"{name} ±1 SD"
        ))
        figure.add_trace(go.Scatter(
            x=x_values,
            y=means,
            mode="lines+markers",
            name=f"{name} RMSE"
        ))

    figure.update_layout(
        title="Training vs Validation RMSE",
        xaxis_title="Number of training examples",
        yaxis_title="RMSE (lower is better)",
        hovermode="x unified",
        legend_title="Curve"
    )
    st.plotly_chart(figure, use_container_width=True)

    display_data = curve_data.rename(columns={
        "Training RMSE SD": "Training RMSE SD",
        "Validation RMSE SD": "Validation RMSE SD"
    })
    st.dataframe(
        display_data.style.format({
            "Training RMSE": "{:.4f}",
            "Training RMSE SD": "{:.4f}",
            "Validation RMSE": "{:.4f}",
            "Validation RMSE SD": "{:.4f}"
        }),
        use_container_width=True
    )

    final_row = curve_data.iloc[-1]
    final_gap = float(
        final_row["Validation RMSE"] - final_row["Training RMSE"]
    )
    st.metric("Final learning-curve gap (validation − training RMSE)",
              f"{final_gap:.4f}")

    if final_gap > 0 and final_gap > float(final_row["Validation RMSE SD"]):
        st.warning(
            "The final validation error is higher than training error by "
            "more than one validation-fold standard deviation. This can be "
            "a sign of overfitting, but should be interpreted alongside the "
            "full curve and held-out test metrics."
        )
    elif final_gap > 0:
        st.info(
            "Validation error is somewhat higher than training error. "
            "Check the full curve and cross-validation variability before "
            "drawing conclusions."
        )
    else:
        st.info(
            "The final validation RMSE is not higher than training RMSE. "
            "This alone does not prove good generalization; use the full "
            "curve and held-out test metrics as well."
        )

    st.caption(
        "A large training/validation gap can suggest overfitting. If both "
        "errors remain high and close, underfitting may be possible. "
        "These are diagnostic clues, not definitive diagnoses."
    )


def display_regression_residual_analysis(baseline_result, y_test):
    """Display held-out residual diagnostics only when selected."""

    show_residuals = st.checkbox(
        "Show Residual Analysis",
        value=False,
        key="show_regression_residual_analysis"
    )

    if not show_residuals:
        return

    if not isinstance(baseline_result, dict):
        st.info("Baseline Random Forest results are unavailable.")
        return

    predictions = baseline_result.get("predictions")

    if predictions is None or y_test is None:
        st.warning("Predictions or test targets are unavailable for residual analysis.")
        return

    actual = np.asarray(y_test, dtype=float).reshape(-1)
    predicted = np.asarray(predictions, dtype=float).reshape(-1)

    if len(actual) != len(predicted) or len(actual) == 0:
        st.warning("Actual values and predictions do not have matching lengths.")
        return

    residuals = actual - predicted
    residual_data = pd.DataFrame({
        "Predicted value": predicted,
        "Residual": residuals,
        "Actual value": actual
    })

    st.subheader("Random Forest Residual Analysis")
    st.caption(
        "Residual = actual value − predicted value. These diagnostics use "
        "the held-out test observations from the baseline Random Forest."
    )

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Mean residual", f"{np.mean(residuals):.4f}")
    metric_two.metric("Residual standard deviation", f"{np.std(residuals):.4f}")
    metric_three.metric("Largest absolute residual", f"{np.max(np.abs(residuals)):.4f}")

    scatter = px.scatter(
        residual_data,
        x="Predicted value",
        y="Residual",
        title="Residuals vs Predicted Values",
        hover_data=["Actual value"]
    )
    scatter.add_hline(y=0, line_dash="dash")
    st.plotly_chart(scatter, use_container_width=True)

    histogram = px.histogram(
        residual_data,
        x="Residual",
        nbins=30,
        title="Distribution of Prediction Errors"
    )
    st.plotly_chart(histogram, use_container_width=True)

    st.caption(
        "A useful residual plot has points scattered around zero without a "
        "clear curve or funnel shape. A strong pattern can indicate missed "
        "relationships or changing error variance. Residuals are diagnostic "
        "signals, not proof that a model is valid."
    )


def run_regression(
        dataframe,
        target_column,
        test_size,
        epochs
):
    st.header("Regression")

    if not display_compatibility(dataframe, target_column, "Regression"):
        return

    try:
        dataset_hash = int(
            pd.util.hash_pandas_object(dataframe, index=True).sum()
        )
    except Exception:
        dataset_hash = (
            f"{len(dataframe)}|{len(dataframe.columns)}|"
            f"{'|'.join(map(str, dataframe.columns))}"
        )

    dataset_signature = (
        f"{len(dataframe)}|{len(dataframe.columns)}|{dataset_hash}"
    )
    cache_key = (
        f"ml_regression_cache::{target_column}::"
        f"{test_size}::{epochs}::{dataset_signature}"
    )

    train_clicked = st.button(
        "Train Regression Models",
        key="train_regression"
    )

    if train_clicked:
        with st.spinner(
            "Training regression models and tuning the Random Forest..."
        ):
            try:
                output = train_regression_models(
                    dataframe,
                    target_column,
                    test_size=test_size,
                    random_state=42
                )
                st.success("Scikit-learn regression models trained.")

                st.subheader("PyTorch Neural Network")
                with st.spinner("Training PyTorch neural network..."):
                    pytorch_output = train_pytorch_regression(
                        dataframe,
                        target_column,
                        split_data=output,
                        epochs=epochs
                    )
                st.success("PyTorch regression model trained.")

                st.session_state[cache_key] = {
                    "output": output,
                    "pytorch_output": pytorch_output
                }

            except Exception as error:
                st.error(f"Regression training failed: {error}")
                return

    cached_results = st.session_state.get(cache_key)
    if cached_results is None:
        st.info(
            "Choose the target column and click "
            "'Train Regression Models' to begin."
        )
        return

    output = cached_results["output"]
    pytorch_output = cached_results["pytorch_output"]
    sklearn_results = output["results"]

    # Combine and display all model results.
    comparison = create_comparison_dataframe(
        sklearn_results,
        "Regression"
    )
    comparison = add_pytorch_to_comparison(
        comparison,
        pytorch_output.get("results", {}),
        "Regression"
    )
    display_model_comparison(comparison, "Regression")
    display_comparison_interpretation(comparison, "Regression")

    # Training-only cross-validation stability diagnostics.
    display_regression_cv_summary(sklearn_results)

    # Random Forest feature importance.
    random_forest_output = sklearn_results.get(
        "Random Forest Regressor", {}
    )
    display_regression_feature_importance(
        random_forest_output.get("feature_importance")
    )

    # Identifier experiment.
    display_identifier_experiment(
        output.get("identifier_experiment")
    )

    # Optional diagnostics: each panel is independently hidden by default.
    baseline_rf_output = sklearn_results.get(
        "Random Forest Regressor", {}
    )
    display_regression_learning_curve(
        baseline_rf_output,
        output.get("X_train"),
        output.get("y_train"),
        cache_key
    )
    display_regression_residual_analysis(
        baseline_rf_output,
        output.get("y_test")
    )

    # Feature selection and hyperparameter tuning.
    tuned_output = sklearn_results.get(
        "Tuned Random Forest (Feature Selection)", {}
    )
    display_regression_tuning(tuned_output)
    display_tuned_random_forest_comparison(sklearn_results)

    # PyTorch interpretation and training history.
    display_pytorch_interpretation(
        pytorch_output.get("interpretation", {})
    )
    display_training_history(
        pytorch_output.get("history", [])
    )
    display_training_interpretation(
        pytorch_output.get("training_interpretation", {})
    )

    st.success(
        "Scikit-learn and PyTorch used the same "
        "outer training and testing observations."
    )


# ============================================================

# 14. CLUSTERING

# ============================================================


def run_clustering(dataframe):
    st.header("Clustering")

    st.write(

        "Clustering groups similar observations without requiring "

        "a target column."

    )

    n_clusters = st.slider(

        "Number of Clusters",

        min_value=2,

        max_value=10,

        value=3,

        key="clustering_number"

    )

    if not st.button(

            "Run Clustering",

            key="run_clustering"

    ):
        return

    with st.spinner("Performing clustering..."):

        try:

            output = perform_clustering(

                dataframe,

                n_clusters=n_clusters,

                random_state=42

            )

            clustered_data = output["clustered_data"]

            st.success(

                "Clustering completed successfully."

            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(

                    "Number of Clusters",

                    output.get(

                        "actual_cluster_count",

                        n_clusters

                    )

                )

            with col2:

                silhouette = output.get(

                    "silhouette_score"

                )

                st.metric(

                    "Silhouette Score",

                    (

                        f"{silhouette:.4f}"

                        if silhouette is not None

                        else "N/A"

                    )

                )

            st.subheader("Clustered Dataset")

            st.dataframe(

                clustered_data.head(100),

                use_container_width=True

            )

            # ------------------------------------------------

            # CLUSTER DISTRIBUTION

            # ------------------------------------------------

            if "Cluster" in clustered_data.columns:
                cluster_counts = (

                    clustered_data["Cluster"]

                    .value_counts()

                    .sort_index()

                    .reset_index()

                )

                cluster_counts.columns = [

                    "Cluster",

                    "Number of Observations"

                ]

                figure = px.bar(

                    cluster_counts,

                    x="Cluster",

                    y="Number of Observations",

                    title="Observations in Each Cluster",

                    text="Number of Observations"

                )

                st.plotly_chart(

                    figure,

                    use_container_width=True

                )

            # ------------------------------------------------

            # DOWNLOAD

            # ------------------------------------------------

            csv_data = clustered_data.to_csv(

                index=False

            ).encode("utf-8")

            st.download_button(

                "Download Clustered Dataset",

                data=csv_data,

                file_name="clustered_dataset.csv",

                mime="text/csv",

                key="download_clustered_data"

            )



        except Exception as error:

            st.error(

                f"Clustering failed: {error}"

            )


# ============================================================

# 15. ANOMALY DETECTION

# ============================================================


def run_anomaly_detection(dataframe):
    st.header("Anomaly Detection")

    st.write(

        "Isolation Forest identifies observations that differ "

        "significantly from the general data pattern."

    )

    contamination = st.slider(

        "Expected Anomaly Proportion",

        min_value=0.01,

        max_value=0.20,

        value=0.05,

        step=0.01,

        key="anomaly_contamination"

    )

    if not st.button(

            "Detect Anomalies",

            key="detect_anomalies"

    ):
        return

    with st.spinner("Detecting anomalies..."):

        try:

            output = detect_anomalies(

                dataframe,

                contamination=contamination,

                random_state=42

            )

            anomaly_data = output["results"]

            st.success(

                "Anomaly detection completed."

            )

            # ------------------------------------------------

            # SUMMARY METRICS

            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(

                    "Total Observations",

                    len(anomaly_data)

                )

            with col2:

                st.metric(

                    "Detected Anomalies",

                    output.get(

                        "anomaly_count",

                        "N/A"

                    )

                )

            with col3:

                percentage = output.get(

                    "anomaly_percentage",

                    0

                )

                st.metric(

                    "Anomaly Percentage",

                    f"{percentage:.2f}%"

                )

            # ------------------------------------------------

            # RESULTS TABLE

            # ------------------------------------------------

            st.subheader("Anomaly Detection Results")

            st.dataframe(

                anomaly_data.head(100),

                use_container_width=True

            )

            # ------------------------------------------------

            # ANOMALY DISTRIBUTION

            # ------------------------------------------------

            if "Anomaly Status" in anomaly_data.columns:
                counts = (

                    anomaly_data["Anomaly Status"]

                    .value_counts()

                    .reset_index()

                )

                counts.columns = [

                    "Status",

                    "Count"

                ]

                figure = px.pie(

                    counts,

                    names="Status",

                    values="Count",

                    title="Normal Observations vs Anomalies"

                )

                st.plotly_chart(

                    figure,

                    use_container_width=True

                )

            # ------------------------------------------------

            # DOWNLOAD

            # ------------------------------------------------

            csv_data = anomaly_data.to_csv(

                index=False

            ).encode("utf-8")

            st.download_button(

                "Download Anomaly Detection Results",

                data=csv_data,

                file_name="anomaly_detection_results.csv",

                mime="text/csv",

                key="download_anomaly_data"

            )



        except Exception as error:

            st.error(

                f"Anomaly detection failed: {error}"

            )


# ============================================================

# 16. MAIN MACHINE LEARNING DASHBOARD

# ============================================================


def render_ml_dashboard(dataframe):
    st.title("Machine Learning Studio")

    st.write(

        "Train machine learning models, compare their performance, "

        "and explore patterns in your dataset."

    )

    if dataframe is None or dataframe.empty:
        st.warning(

            "Please upload a valid dataset first."

        )

        return

    task = st.selectbox(

        "Select Machine Learning Task",

        [

            "Classification",

            "Regression",

            "Clustering",

            "Anomaly Detection"

        ],

        key="ml_task"

    )

    st.divider()

    # ========================================================

    # CLASSIFICATION AND REGRESSION

    # ========================================================

    if task in [

        "Classification",

        "Regression"

    ]:

        target_column = st.selectbox(

            "Select Target Column",

            dataframe.columns.tolist(),

            key=f"{task.lower()}_target"

        )

        col1, col2 = st.columns(2)

        with col1:

            test_size = st.slider(

                "Testing Data Percentage",

                min_value=10,

                max_value=40,

                value=20,

                step=5,

                key="ml_test_size"

            )

        with col2:

            epochs = st.slider(

                "PyTorch Training Epochs",

                min_value=10,

                max_value=100,

                value=50,

                step=10,

                key="pytorch_epochs"

            )

        if task == "Classification":

            run_classification(

                dataframe,

                target_column,

                test_size / 100,

                epochs

            )



        else:

            run_regression(

                dataframe,

                target_column,

                test_size / 100,

                epochs

            )



    # ========================================================

    # CLUSTERING

    # ========================================================

    elif task == "Clustering":

        run_clustering(dataframe)



    # ========================================================

    # ANOMALY DETECTION

    # ========================================================

    elif task == "Anomaly Detection":

        run_anomaly_detection(dataframe)

