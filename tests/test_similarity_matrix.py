import unittest 
import pandas as pd 
import numpy as np
from src.recommender import build_ratings_similarity_matrix

class TestRatingsSimiliratyMatrix(unittest.TestCase):
    def setUp(self):
        # Sample ratings
        self.ratings_df = pd.DataFrame({
            "userId": [1, 1, 2, 2, 3],
            "movieId": [101, 102, 101, 103, 102],
            "rating": [5.0, 4.0, 4.0, 3.0, 4.0],
        })

        # Sample movies
        self.movies_df = pd.DataFrame({
            "movieId": [101, 102, 103],
            "title": ["Movie A", "Movie B", "Movie C"]
        })
        
    def test_similarity_matrix_structure(self):
        similarity_df = build_ratings_similarity_matrix(self.ratings_df, self.movies_df)
        
        # Check if output is DataFrame
        self.assertIsInstance(similarity_df, pd.DataFrame)
        
        # Check dimensions
        self.assertEqual(similarity_df.shape, (3, 3))
        
        # Check index and columns match movie titles
        expected_titles = ["Movie A", "Movie B", "Movie C"]
        self.assertListEqual(list(similarity_df.index), expected_titles)
        self.assertListEqual(list(similarity_df.columns), expected_titles)
        
    def test_diagonal_values_are_one(self):
        similarity_df = build_ratings_similarity_matrix(self.ratings_df, self.movies_df)
        
        # Diagonal should 1.0
        for i in range(similarity_df.shape[0]):
            self.assertAlmostEqual(similarity_df.iloc[i, i], 1.0, places=6)
            
    def test_similarity_matrix_is_symmetric(self):
        similarity_df = build_ratings_similarity_matrix(self.ratings_df, self.movies_df)
        
        # Check symmetry
        np.testing.assert_allclose(similarity_df.values, similarity_df.values.T, rtol=1e-5, atol=1e-8)
        
if __name__ == '__main__':
    unittest.main()