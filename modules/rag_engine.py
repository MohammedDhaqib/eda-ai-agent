"""
Semantic retrieval engine for the EDA AI Agent.

Stores Python-generated dataset findings in ChromaDB
and rejects semantically irrelevant retrieval results.
"""

import chromadb
import numpy as np

from sentence_transformers import SentenceTransformer


class EDARetrievalEngine:

    # Minimum cosine similarity required to accept a finding.
    MIN_SIMILARITY = 0.40

    def __init__(self):

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = None

    def create_collection(self, collection_name):
        """Create or access a dataset-specific collection."""

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "EDA dataset knowledge"}
        )

        return self.collection.name

    def add_documents(self, documents):
        """Embed and store analysis documents."""

        if self.collection is None:
            raise ValueError(
                "Create a collection before adding documents."
            )

        if not documents:
            raise ValueError("No documents were provided.")

        ids = [
            f"document_{i}"
            for i in range(len(documents))
        ]

        embeddings = self.embedding_model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )

        return len(documents)

    def retrieve_documents(self, question, top_k=3):
        """
        Retrieve relevant findings using cosine similarity.

        Findings below the relevance threshold are rejected.
        """

        if self.collection is None:
            raise ValueError(
                "No collection has been selected."
            )

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Please provide a valid question.")

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        document_count = self.collection.count()

        if document_count == 0:
            return []

        query_embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, document_count),
            include=["documents", "embeddings"]
        )

        documents = results.get("documents", [[]])[0]
        embeddings = results.get("embeddings")

        if not documents or embeddings is None:
            return []

        retrieved_embeddings = embeddings[0]

        relevant_documents = []

        for document, stored_embedding in zip(
            documents,
            retrieved_embeddings
        ):

            stored_embedding = np.asarray(
                stored_embedding,
                dtype=float
            )

            query_vector = np.asarray(
                query_embedding,
                dtype=float
            )

            denominator = (
                np.linalg.norm(query_vector)
                * np.linalg.norm(stored_embedding)
            )

            if denominator == 0:
                continue

            similarity = float(
                np.dot(query_vector, stored_embedding)
                / denominator
            )

            if similarity >= self.MIN_SIMILARITY:
                relevant_documents.append(document)

        return relevant_documents