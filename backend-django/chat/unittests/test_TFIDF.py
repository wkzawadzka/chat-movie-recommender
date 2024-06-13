from django.test import TestCase  # type: ignore
import pandas as pd
from chat.models import TFIDF
from core.settings import PLOTS_DATA


class TestTFIDFRecommendation(TestCase):
    """Unit tests for the TFIDF-based movie recommendation model."""

    def setUp(self) -> None:
        """Set up the TFIDF model with test data before each test."""
        plots = pd.read_csv(PLOTS_DATA)
        self.tfidf_model = TFIDF(plots)

    def test_recommend(self) -> None:
        """
        Test the basic functionality of the TFIDF model's recommend method.

        The method should return a list of 5 movie IDs.
        """
        query = "A movie about toys"
        result = self.tfidf_model.recommend(query)
        self.assertIsInstance(
            result, list, "The result should be a list of movie IDs.")
        self.assertEqual(
            len(result), 5, "The result should contain 5 movie IDs.")

    def tearDown(self) -> None:
        """Clean up any resources allocated in setUp."""
        pass
