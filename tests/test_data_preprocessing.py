# Note: all tests passed successfully

import unittest
import os
import pandas as pd
from src.data_preprocessing import load_data, get_tfidf_matrix

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_dir = os.path.join(self.project_root, "data")
    
    def test_load_movies_data(self):
        df = load_data(self.data_dir, dataset="movies")
        self.assertIn("description", df.columns)
        self.assertIn("title", df.columns)
        self.assertTrue(len(df) > 0)
    
    def test_load_tags_data(self):
        df = load_data(self.data_dir, dataset="tags")
        self.assertIn("tag", df.columns)
        self.assertIn("movieId", df.columns)
        self.assertTrue(len(df) > 0)
    
    def test_get_tfidf_matrix_movies(self):
        df = load_data(self.data_dir, dataset="movies")
        matrix = get_tfidf_matrix(df, dataset="movies")
        self.assertEqual(matrix.shape[0], len(df))
    
    def test_get_tfidf_matrix_tags(self):
        df = load_data(self.data_dir, dataset="tags")
        matrix = get_tfidf_matrix(df, dataset="tags")
        self.assertEqual(matrix.shape[0], len(df))

if __name__ == "__main__":
    unittest.main()