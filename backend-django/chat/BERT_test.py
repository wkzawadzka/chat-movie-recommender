import unittest
import pandas as pd
from models import BERT

test_movies = pd.DataFrame({
    'movieID': [1, 2, 3],
    'overview': ['This is movie 1', 'Overview of movie 2', 'Movie 3 description']
})

class TestBERTRecommendation(unittest.TestCase):

    def setUp(self):
        self.bert_model = BERT(test_movies)

    def test_recommend_basic(self):
        query = "A movie about toys"
        result = self.bert_model.recommend(query)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)

    def test_recommend_empty_query(self):
        query = ""
        result = self.bert_model.recommend(query)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_recommend_short_overview(self):
        query = "Action movie"
        result = self.bert_model.recommend(query)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)

    def test_recommend_no_matches(self):
        query = "This is a test query with no matches"
        result = self.bert_model.recommend(query)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()