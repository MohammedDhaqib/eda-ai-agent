from modules.llm_engine import (
    generate_response,
    generate_dataset_summary,
    ask_dataset_question
)


print("\nTEST 1: GENERAL LLM RESPONSE")

response = generate_response(
    "Explain the difference between mean and median."
)

print(response)


print("\nTEST 2: DATASET SUMMARY")

dataset_info = """
Dataset: Students

Rows: 1000
Columns: 5

Columns:
Age: Numerical
Gender: Categorical
Marks: Numerical
Attendance: Numerical
Department: Categorical

Missing values: 25
Duplicate rows: 10
"""

summary = generate_dataset_summary(dataset_info)

print(summary)


print("\nTEST 3: DATASET QUESTION ANSWERING")

answer = ask_dataset_question(
    dataset_info,
    "How many missing values are present?"
)

print(answer)


print("\nALL TESTS COMPLETED!")