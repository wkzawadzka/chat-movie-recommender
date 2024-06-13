from django.test import TestCase  # type: ignore
import pandas as pd
from chat.models import Word2VecModel
from core.settings import PLOTS_DATA


class TestWord2VecModelRecommendation(TestCase):
    """Unit tests for the Word2Vec-based movie recommendation model."""

    def setUp(self) -> None:
        """Set up the Word2Vec model with test data before each test."""
        plots = pd.read_csv(PLOTS_DATA)
        self.word2vec_model = Word2VecModel(plots)

    def test_recommend(self) -> None:
        """
        Test the basic functionality of the Word2Vec model's recommend method.

        The method should return a list of 5 movie IDs.
        """
        query = "A movie about toys"
        result = self.word2vec_model.recommend(query)
        self.assertIsInstance(
            result, list, "The result should be a list of movie IDs.")
        self.assertEqual(
            len(result), 5, "The result should contain 5 movie IDs.")

    def tearDown(self) -> None:
        """Clean up any resources allocated in setUp."""
        pass
