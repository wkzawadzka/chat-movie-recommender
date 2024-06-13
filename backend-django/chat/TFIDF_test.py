import unittest
import pandas as pd
from models import TFIDF

# Example test data
test_movies = pd.DataFrame({
    'movieID': [1, 2, 3],
    'overview': ['This is movie 1', 'Overview of movie 2', 'Movie 3 description']
})

class TestTFIDFRecommendation(unittest.TestCase):

    def setUp(self):
        self.tfidf_model = TFIDF(test_movies)

    def test_recommend(self):
        query = "A movie about toys"
        result = self.tfidf_model.recommend(query)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)

    def test_recommend_empty_query(self):
        query = ""
        result = self.tfidf_model.recommend(query)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)
    
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
