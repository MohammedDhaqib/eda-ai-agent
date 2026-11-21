"""
LangChain integration for the EDA AI Agent.

Uses:
- Ollama for natural-language answers
- Hugging Face for embeddings
- ChromaDB for document retrieval
"""

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class EDALangChainEngine:

    def __init__(
        self,
        model_name="llama3.2:1b",
        collection_name="eda_uploaded_dataset_knowledge",
        persist_directory="./chroma_db"
    ):

        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def retrieve_documents(self, question, top_k=3):

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Please provide a valid question.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        return self.vector_store.similarity_search(
            query=question,
            k=top_k
        )

    @staticmethod
    def check_categorical_tie(documents, question):

        question_lower = question.lower()

        if not any(
            word in question_lower
            for word in ["most", "highest", "maximum", "largest", "top"]
        ):
            return None

        for document in documents:

            content = document.page_content

            if "Category counts and percentages:" not in content:
                continue

            counts_section = content.split(
                "Category counts and percentages:", 1
            )[1]

            import re

            matches = re.findall(
                r"'([^']+)'\s*=\s*(\d+)\s*\(([\d.]+)%\)",
                counts_section
            )

            if len(matches) < 2:
                continue

            counts = [
                (category, int(count), percentage)
                for category, count, percentage in matches
            ]

            highest_count = max(count for _, count, _ in counts)

            winners = [
                (category, count, percentage)
                for category, count, percentage in counts
                if count == highest_count
            ]

            if len(winners) > 1:

                winner_text = " and ".join(
                    f"{category} ({count} students, {percentage}%)"
                    for category, count, percentage in winners
                )

                return (
                    f"There is a tie for the highest count: "
                    f"{winner_text}."
                )

            category, count, percentage = winners[0]

            return (
                f"The category with the highest count is "
                f"{category}, with {count} students ({percentage}%)."
            )

        return None

    def answer_question(self, question, top_k=3):

        documents = self.retrieve_documents(question, top_k)

        if not documents:
            return {
                "status": "no_evidence",
                "question": question,
                "answer": "No relevant dataset evidence was found.",
                "retrieved_documents": []
            }

        # Verify categorical maximum and ties using Python.
        verified_answer = self.check_categorical_tie(
            documents,
            question
        )

        if verified_answer is not None:
            return {
                "status": "success",
                "question": question,
                "answer": verified_answer,
                "retrieved_documents": [
                    document.page_content
                    for document in documents
                ],
                "framework": "LangChain",
                "verification": "Python"
            }

        context = "\n\n".join(
            document.page_content for document in documents
        )

        prompt = f"""
You are the AI assistant of an Exploratory Data Analysis Agent.

Answer using ONLY the retrieved dataset evidence.

QUESTION:
{question}

DATASET EVIDENCE:
{context}

RULES:
1. Never invent statistics.
2. Do not combine unrelated column information.
3. Explain findings clearly.
4. Correlation does not establish causation.
5. If evidence is insufficient, say so.
"""

        response = self.llm.invoke(prompt)

        return {
            "status": "success",
            "question": question,
            "answer": response.content,
            "retrieved_documents": [
                document.page_content for document in documents
            ],
            "llm_model": "llama3.2:1b",
            "framework": "LangChain"
        }