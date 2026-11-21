import re
import pandas as pd


ID_NAMES = {
    "id",
    "index",
    "passengerid",
    "customerid",
    "userid",
    "flightid",
    "bookingid",
    "ticketid"
}


def inspect_target(dataframe, target_column):
    """
    Inspect a target column and recommend suitable ML tasks.

    Returns:
        Target statistics
        Classification compatibility
        Regression compatibility
        Human-readable explanations
        Warnings
    """

    if target_column not in dataframe.columns:
        raise ValueError(
            "The selected target column does not exist."
        )

    target = dataframe[target_column]

    total_rows = len(target)

    missing_count = int(target.isna().sum())

    usable_target = target.dropna()

    usable_rows = len(usable_target)

    unique_count = int(usable_target.nunique())

    unique_ratio = (
        unique_count / usable_rows
        if usable_rows > 0
        else 0
    )

    data_type = str(target.dtype)

    name = str(target_column).strip().lower()

    # ==========================================
    # IDENTIFIER DETECTION
    # ==========================================

    is_id_name = (
        name in ID_NAMES
        or bool(
            re.fullmatch(
                r"unnamed:\s*\d+",
                name
            )
        )
    )

    # ==========================================
    # DATA TYPE DETECTION
    # ==========================================

    is_boolean = pd.api.types.is_bool_dtype(target)

    is_numeric = (
        pd.api.types.is_numeric_dtype(target)
        and not is_boolean
    )

    is_categorical = (
        not is_numeric
        or is_boolean
        or isinstance(
            target.dtype,
            pd.CategoricalDtype
        )
    )

    is_binary_numeric = (
        is_numeric
        and unique_count == 2
        and set(usable_target.unique()).issubset({0, 1})
    )

    # ==========================================
    # COMPATIBILITY INITIALIZATION
    # ==========================================

    classification_supported = False

    regression_supported = False

    classification_reasons = []

    regression_reasons = []

    warnings = []

    # ==========================================
    # DATA QUALITY CHECKS
    # ==========================================

    if usable_rows < 10:

        warnings.append(
            "Fewer than 10 usable target rows are available. "
            "Training may be unreliable."
        )

    if missing_count > 0:

        warnings.append(
            f"{missing_count} target values are missing. "
            "These rows must be excluded during training."
        )

    if unique_count < 2:

        warnings.append(
            "The target contains fewer than two distinct values. "
            "A predictive model cannot learn a meaningful target."
        )

    # ==========================================
    # IDENTIFIER CHECK
    # ==========================================

    if is_id_name:

        warnings.append(
            "The column name looks like an identifier. "
            "It may identify records rather than represent "
            "something meaningful to predict."
        )

    # ==========================================
    # CLASSIFICATION CHECK
    # ==========================================

    elif unique_count >= 2:

        if is_boolean:

            classification_supported = True

            classification_reasons.append(
                "The target contains Boolean values "
                "and is suitable for binary classification."
            )

        elif is_binary_numeric:

            classification_supported = True

            classification_reasons.append(
                "The numerical target contains exactly two "
                "values, 0 and 1, making it suitable for "
                "binary classification."
            )

        elif is_categorical:

            if unique_count <= 20:

                classification_supported = True

                classification_reasons.append(
                    f"The target contains {unique_count} "
                    "distinct categories."
                )

            else:

                warnings.append(
                    "The categorical target has more than "
                    "20 distinct values. Check whether it "
                    "represents an identifier."
                )

        elif is_numeric and unique_count <= 20:

            classification_supported = True

            classification_reasons.append(
                "The numerical target has relatively few "
                "distinct values. Classification may be "
                "appropriate if these values represent categories."
            )

    # ==========================================
    # REGRESSION CHECK
    # ==========================================

    if (
        is_numeric
        and not is_boolean
        and unique_count >= 2
        and not is_id_name
    ):

        regression_supported = True

        regression_reasons.append(
            "The target is numerical and contains varying values."
        )

    # ==========================================
    # MINIMUM DATA CHECK
    # ==========================================

    if usable_rows < 10 or unique_count < 2:

        classification_supported = False

        regression_supported = False

    # ==========================================
    # FINAL EXPLANATIONS
    # ==========================================

    if classification_supported:

        classification_reasons.append(
            "Classification algorithms can be considered."
        )

    if regression_supported:

        regression_reasons.append(
            "Regression algorithms can be considered."
        )

    # ==========================================
    # TASK RECOMMENDATION
    # ==========================================

    if not classification_supported and not regression_supported:

        recommendation = "Not recommended"

    elif classification_supported and regression_supported:

        recommendation = (
            "Classification or Regression — verify target meaning"
        )

    elif classification_supported:

        recommendation = "Classification"

    else:

        recommendation = "Regression"

    # ==========================================
    # RETURN RESULTS
    # ==========================================

    return {

        "target_column": target_column,

        "data_type": data_type,

        # Compatibility alias used by the dashboard
        "dtype": data_type,

        "total_rows": total_rows,

        "usable_rows": usable_rows,

        "missing_count": missing_count,

        "unique_count": unique_count,

        "unique_ratio": unique_ratio,

        "is_numeric": is_numeric,

        "is_categorical": is_categorical,

        "is_identifier_like": is_id_name,

        "classification_supported": classification_supported,

        "regression_supported": regression_supported,

        "classification_reasons": classification_reasons,

        "regression_reasons": regression_reasons,

        # Combined explanation for dashboard compatibility
        "reasons": (
            classification_reasons
            + regression_reasons
        ),

        "warnings": warnings,

        "recommendation": recommendation
    }