from modules.rag_qa import answer_dataset_question


# Ask questions about the students dataset
questions = [
    "What is the average marks?",
    "What is the age distribution?",
    "What is the relationship between age and marks?",
    "How many duplicate rows are present?"
]

for question in questions:

    print("\n" + "=" * 65)
    print("QUESTION:", question)

    result = answer_dataset_question(
        question=question,
        collection_name="airline_dataset_knowledge",
        top_k=3
    )

    print("\nRETRIEVED EVIDENCE:")

    for evidence in result["retrieved_evidence"]:
        print("-", evidence)

    print("\nAI-GENERATED ANSWER:")
    print(result["answer"])