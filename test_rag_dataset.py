import pandas as pd

from modules.rag_dataset_builder import index_dataset_insights
from modules.rag_engine import EDARetrievalEngine


# Step 1: Load the actual dataset
dataframe = pd.read_csv("datasets/students.csv")

print("Dataset loaded:", dataframe.shape)


# Step 2: Generate and index actual dataset insights
indexing_result = index_dataset_insights(
    dataframe,
    collection_name="airline_dataset_knowledge"
)

print("\nINDEXING RESULT:")
print(indexing_result)


# Step 3: Connect to the indexed collection
rag = EDARetrievalEngine()

rag.collection = rag.client.get_collection(
    name="airline_dataset_knowledge"
)


# Step 4: Ask questions about the dataset
questions = [
    "Which department has the most students?",
    "What is the average marks?",
    "What is the age distribution?",
    "How many duplicate rows are present?",
    "What is the relationship between age and marks?"
]


# Step 5: Retrieve relevant findings
for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)

    results = rag.retrieve_documents(
        question,
        top_k=3
    )

    print("\nRETRIEVED FINDINGS:")

    for result in results:
        print("-", result)