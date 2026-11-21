import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from modules.rag_engine import EDARetrievalEngine


class TestRAGRetrievalEngine(unittest.TestCase):

    def setUp(self):
        # Mock the embedding model so tests do not download model weights.
        self.embedding_patcher = patch("modules.rag_engine.SentenceTransformer")
        self.mock_embedding_class = self.embedding_patcher.start()
        self.addCleanup(self.embedding_patcher.stop)
        self.mock_embedding = self.mock_embedding_class.return_value

        # Mock ChromaDB so tests never touch the real vector database.
        self.client_patcher = patch(
            "modules.rag_engine.chromadb.PersistentClient"
        )
        self.mock_client_class = self.client_patcher.start()
        self.addCleanup(self.client_patcher.stop)
        self.mock_client = self.mock_client_class.return_value

        self.engine = EDARetrievalEngine()
        self.mock_collection = MagicMock()
        self.mock_collection.name = "test_collection"
        self.engine.collection = self.mock_collection

    def test_initialization_creates_model_and_persistent_client(self):
        self.mock_embedding_class.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.mock_client_class.assert_called_once_with(path="./chroma_db")

        fresh_engine = EDARetrievalEngine()
        self.assertIsNone(fresh_engine.collection)

    def test_create_collection_uses_chromadb(self):
        self.mock_client.get_or_create_collection.return_value = (
            self.mock_collection
        )

        result = self.engine.create_collection("test_collection")

        self.mock_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            metadata={"description": "EDA dataset knowledge"},
        )
        self.assertEqual(result, "test_collection")
        self.assertIs(self.engine.collection, self.mock_collection)

    def test_add_documents_upserts_embeddings(self):
        documents = ["Dataset has 100 rows.", "There are no duplicate rows."]
        self.mock_embedding.encode.return_value = np.array(
            [[1.0, 0.0], [0.0, 1.0]]
        )

        result = self.engine.add_documents(documents)

        self.mock_embedding.encode.assert_called_once_with(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.mock_collection.upsert.assert_called_once_with(
            ids=["document_0", "document_1"],
            documents=documents,
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertEqual(result, 2)

    def test_add_documents_without_collection_is_rejected(self):
        self.engine.collection = None
        with self.assertRaisesRegex(
            ValueError, "Create a collection before adding documents"
        ):
            self.engine.add_documents(["A finding."])

    def test_add_empty_documents_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No documents were provided"):
            self.engine.add_documents([])

    def test_relevant_document_is_retrieved(self):
        self.mock_collection.count.return_value = 1
        self.mock_embedding.encode.return_value = np.array([1.0, 0.0])
        self.mock_collection.query.return_value = {
            "documents": [["Age has a mean of 22."]],
            "embeddings": [[[0.9, 0.1]]],
        }

        result = self.engine.retrieve_documents(
            question="What is the average age?", top_k=3
        )

        self.assertEqual(result, ["Age has a mean of 22."])
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[[1.0, 0.0]],
            n_results=1,
            include=["documents", "embeddings"],
        )

    def test_irrelevant_document_is_rejected(self):
        self.mock_collection.count.return_value = 1
        self.mock_embedding.encode.return_value = np.array([1.0, 0.0])
        self.mock_collection.query.return_value = {
            "documents": [["The department is Statistics."]],
            "embeddings": [[[0.0, 1.0]]],
        }

        result = self.engine.retrieve_documents(
            question="What is the average age?"
        )

        self.assertEqual(result, [])

    def test_empty_collection_returns_no_results(self):
        self.mock_collection.count.return_value = 0

        result = self.engine.retrieve_documents(
            question="What is the average age?"
        )

        self.assertEqual(result, [])
        self.mock_embedding.encode.assert_not_called()
        self.mock_collection.query.assert_not_called()

    def test_retrieval_without_collection_is_rejected(self):
        self.engine.collection = None
        with self.assertRaisesRegex(
            ValueError, "No collection has been selected"
        ):
            self.engine.retrieve_documents("What is the average age?")

    def test_empty_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid question"):
            self.engine.retrieve_documents(" ")

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "top_k must be a positive integer"
        ):
            self.engine.retrieve_documents(
                question="What is the average age?", top_k=0
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
