
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

from modules.langchain_engine import EDALangChainEngine


class TestEDALangChainEngine(unittest.TestCase):

    def setUp(self):
        # Create an engine without loading real AI models.
        self.engine = EDALangChainEngine.__new__(
            EDALangChainEngine
        )

        self.engine.vector_store = Mock()
        self.engine.llm = Mock()

    def test_successful_document_retrieval(self):

        expected_documents = [
            SimpleNamespace(page_content="Test dataset evidence")
        ]

        self.engine.vector_store.similarity_search.return_value = (
            expected_documents
        )

        result = self.engine.retrieve_documents(
            "What is in the dataset?",
            top_k=3
        )

        self.assertEqual(result, expected_documents)

        self.engine.vector_store.similarity_search.assert_called_once_with(
            query="What is in the dataset?",
            k=3
        )

    def test_empty_question_is_rejected(self):

        with self.assertRaises(ValueError):
            self.engine.retrieve_documents("")

    def test_invalid_top_k_is_rejected(self):

        with self.assertRaises(ValueError):
            self.engine.retrieve_documents(
                "Test question",
                top_k=0
            )

    def test_categorical_tie_is_detected(self):

        document = SimpleNamespace(
            page_content=(
                "Categorical distribution for column 'Department': "
                "3 observed categories among 5 non-missing rows. "
                "Category counts and percentages: "
                "'Statistics' = 2 (40.00%); "
                "'Computer Science' = 2 (40.00%); "
                "'Mathematics' = 1 (20.00%)."
            )
        )

        result = self.engine.check_categorical_tie(
            [document],
            "Which department has the most students?"
        )

        self.assertIn("There is a tie", result)
        self.assertIn("Statistics", result)
        self.assertIn("Computer Science", result)

    def test_highest_category_is_detected(self):

        document = SimpleNamespace(
            page_content=(
                "Category counts and percentages: "
                "'Statistics' = 3 (60.00%); "
                "'Mathematics' = 2 (40.00%)."
            )
        )

        result = self.engine.check_categorical_tie(
            [document],
            "Which department has the highest count?"
        )

        self.assertIn("Statistics", result)
        self.assertIn("3 students", result)

    def test_no_evidence_returns_safe_answer(self):

        self.engine.retrieve_documents = Mock(return_value=[])

        result = self.engine.answer_question(
            "What is the dataset about?"
        )

        self.assertEqual(result["status"], "no_evidence")

        self.assertEqual(
            result["answer"],
            "No relevant dataset evidence was found."
        )

        self.engine.llm.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
