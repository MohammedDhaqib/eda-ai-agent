import pandas as pd

from modules.pytorch_model import (
    train_pytorch_classification,
    train_pytorch_regression
)


# Small sample classification dataset
classification_data = pd.DataFrame({
    "hours_studied": [
        1, 2, 3, 4, 5, 6, 7, 8,
        2, 3, 4, 6, 7, 8, 9, 10,
        1, 3, 5, 7, 9, 10, 4, 6
    ],
    "attendance": [
        50, 55, 60, 65, 70, 75, 80, 85,
        55, 60, 65, 75, 80, 85, 90, 95,
        45, 58, 72, 82, 92, 98, 68, 78
    ],
    "result": [
        "Fail", "Fail", "Fail", "Fail",
        "Pass", "Pass", "Pass", "Pass",
        "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Pass",
        "Fail", "Fail", "Pass", "Pass",
        "Pass", "Pass", "Fail", "Pass"
    ]
})


print("\nTRAINING PYTORCH CLASSIFICATION MODEL")

classification_output = train_pytorch_classification(
    classification_data,
    target_column="result",
    epochs=20
)

print("\nClassification Results:")
print(classification_output["results"])


# Small sample regression dataset
regression_data = pd.DataFrame({
    "hours_studied": [
        1, 2, 3, 4, 5, 6, 7, 8,
        2, 3, 4, 6, 7, 8, 9, 10,
        1, 3, 5, 7, 9, 10, 4, 6
    ],
    "attendance": [
        50, 55, 60, 65, 70, 75, 80, 85,
        55, 60, 65, 75, 80, 85, 90, 95,
        45, 58, 72, 82, 92, 98, 68, 78
    ],
    "marks": [
        30, 35, 40, 45, 50, 55, 60, 65,
        34, 39, 44, 56, 61, 66, 71, 76,
        28, 41, 52, 63, 74, 80, 46, 57
    ]
})


print("\nTRAINING PYTORCH REGRESSION MODEL")

regression_output = train_pytorch_regression(
    regression_data,
    target_column="marks",
    epochs=20
)

print("\nRegression Results:")
print(regression_output["results"])

print("\nBoth PyTorch training tests completed!")