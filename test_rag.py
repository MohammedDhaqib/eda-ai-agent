from modules.rag_engine import EDARetrievalEngine


# Create the retrieval engine
rag = EDARetrievalEngine()

# Create a collection
rag.create_collection("eda_test")

# Sample dataset findings
documents = [
    "The dataset contains 25976 rows and 25 columns.",
    "The Arrival Delay in Minutes column contains 83 missing values.",
    "The dataset contains zero duplicate rows.",
    "The satisfaction column contains satisfied and neutral or dissatisfied categories.",
    "The Pearson correlation between departure delay and arrival delay is approximately 0.965."
]

# Store findings
count = rag.add_documents(documents)

print(f"\nSuccessfully stored {count} documents!")

# Ask a question
question = "Which column contains missing values?"

# Retrieve relevant findings
results = rag.retrieve_documents(question, top_k=3)

print("\nQuestion:", question)
print("\nRetrieved findings:")

for result in results:
    print("-", result)