from django.test import TestCase  # type: ignore
import pandas as pd
from chat.models import BERT
from core.settings import PLOTS_DATA


class TestBERTRecommendation(TestCase):
    """Unit tests for the BERT-based movie recommendation model."""

    def setUp(self) -> None:
        """Set up the BERT model with test data before each test."""
        plots = pd.read_csv(PLOTS_DATA)
        self.bert_model = BERT(plots)

    def test_recommend_basic(self) -> None:
        """
        Test the basic functionality of the BERT model's recommend method.

        The recommend method should return a list of 5 movie IDs.
        """
        query = "A movie about toys"
        result = self.bert_model.recommend(query)
        self.assertIsInstance(
            result, list, "The result should be a list of movie IDs.")
        self.assertEqual(
            len(result), 5, "The result should contain 5 movie IDs.")

    def tearDown(self) -> None:
        """Clean up any resources allocated in setUp."""
        pass
