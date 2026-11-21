import unittest
from unittest.mock import patch, MagicMock

from modules.rag_qa import answer_dataset_question


class TestRAGQuestionAnswering(unittest.TestCase):

    def setUp(self):
        self.engine_patcher = patch(
            "modules.rag_qa.EDARetrievalEngine"
        )

        self.mock_engine_class = self.engine_patcher.start()
        self.addCleanup(self.engine_patcher.stop)

        self.mock_engine = self.mock_engine_class.return_value

        self.mock_engine.client.get_collection.return_value = MagicMock()

    def test_categorical_question_gets_direct_answer(self):

        self.mock_engine.retrieve_documents.return_value = [
            "Categorical distribution for column 'Gender': "
            "'Female': 50 (50%), 'Male': 50 (50%)."
        ]

        result = answer_dataset_question(
            question="What is the distribution of gender?"
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("Gender", result["answer"])
        self.assertIsNone(result["llm_model"])

    @patch("modules.rag_qa.generate_response")
    def test_other_questions_use_llm(self, mock_generate):

        mock_generate.return_value = "The average age is 25."

        self.mock_engine.retrieve_documents.return_value = [
            "Age: mean=25.00, median=24.00."
        ]

        result = answer_dataset_question(
            question="What is the average age?"
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["answer"],
            "The average age is 25."
        )

        mock_generate.assert_called_once()

    @patch("modules.rag_qa.generate_response")
    def test_no_evidence_returns_safe_answer(self, mock_generate):

        self.mock_engine.retrieve_documents.return_value = []

        result = answer_dataset_question(
            question="What is the average salary?"
        )

        self.assertEqual(result["status"], "no_evidence")
        self.assertEqual(result["retrieved_evidence"], [])
        self.assertIn("could not find relevant evidence", result["answer"])

        mock_generate.assert_not_called()

    def test_empty_question_is_rejected(self):

        with self.assertRaises(ValueError):
            answer_dataset_question("")

    def test_invalid_top_k_is_rejected(self):

        with self.assertRaises(ValueError):
            answer_dataset_question(
                question="What is the average age?",
                top_k=0
            )

    def test_missing_chromadb_collection_is_reported(self):

        self.mock_engine.client.get_collection.side_effect = Exception(
            "Collection does not exist"
        )

        with self.assertRaises(ValueError):
            answer_dataset_question(
                question="What is the average age?"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)