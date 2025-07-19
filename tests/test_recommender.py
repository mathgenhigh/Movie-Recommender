# Note: all tests passed successfully

import unittest
import os
from src.data_preprocessing import load_data, get_tfidf_matrix
from src.recommender import recommend_movies

class TestRecommender(unittest.TestCase):
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_dir = os.path.join(self.project_root, "data")
        
        # Load datasets once for tests
        self.movies_df = load_data(self.data_dir, dataset="movies")
        self.movies_matrix = get_tfidf_matrix(self.movies_df, dataset="movies")
        
        self.tags_df = load_data(self.data_dir, dataset="tags")
        self.tags_matrix = get_tfidf_matrix(self.tags_df, dataset="tags")

    def test_recommend_movies_valid_title(self):
        title = self.movies_df['title'].iloc[0]
        recommendations = recommend_movies(title, self.movies_df, self.movies_matrix, dataset="movies", top_n=5)
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) <= 5)
        self.assertIn(str, [type(r) for r in recommendations])
    
    def test_recommend_movies_invalid_title(self):
        with self.assertRaises(ValueError):
            recommend_movies("Nonexistent Movie Title", self.movies_df, self.movies_matrix, dataset="movies")
    
    def test_recommend_tags_returns_movie_ids(self):
        tag = self.tags_df['tag'].iloc[0]
        movie_ids = recommend_movies(tag, self.tags_df, self.tags_matrix, dataset="tags")
        self.assertIsInstance(movie_ids, list)
        self.assertTrue(all(isinstance(i, (int, float)) for i in movie_ids))  # movieId can be int or float
    
    def test_recommend_tags_invalid_tag(self):
        with self.assertRaises(ValueError):
            recommend_movies("nonexistenttag", self.tags_df, self.tags_matrix, dataset="tags")

if __name__ == "__main__":
    unittest.main()