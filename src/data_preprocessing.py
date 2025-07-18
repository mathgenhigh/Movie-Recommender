import os
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(data_dir, dataset="movies"):
    """
    Load one dataset from the given directory.
    
    Parameters:
        data_dir (str): Path to the directory containing CSV files.
        dataset (str): Which dataset to load.
        
    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    
    filenames = {
        "movies": "movies.csv",
        "ratings": "ratings.csv",
        "tags": "tags.csv"
    }
    
    if dataset not in filenames:
        raise ValueError(f"Dataset must be one of {list(filenames.keys())}")
    
    file_path = os.path.join(data_dir, filenames[dataset])
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{filenames[dataset]} not found in {data_dir}")
    
    df = pd.read_csv(file_path)
    
    if dataset == "movies":
        if 'description' not in df.columns:
            if 'genres' in df.columns:
                df['description'] = df['genres'].str.replace('|', ' ', regex=False)
            else:
                raise KeyError("Expected 'genres' column not found in movies dataset")
        df['description'].replace('', pd.NA, inplace=True)
        df.dropna(subset=['description'], inplace=True)

    return df 

def get_tfidf_matrix(df, dataset="movies"):
    """
    Create TF-IDF matrix for the given dataset DataFrame.
   
    Parameters:
        df: pandas DataFrame containing the data.
        dataset: str

    Returns:
        tfidf_matrix: sparse matrix of TF-IDF features
    """
    
    if dataset == "movies":
        if 'description' not in df.columns:
            raise ValueError("Movies dataset must contain 'description' columns")
        text_data = df['description']
    
    elif dataset == "tags":
        if 'tag' not in df.columns:
            raise ValueError("Tags dataset must contain 'tag' column")
        text_data = df['tag']
        
    elif dataset == "ratings":
        raise ValueError("Ratings dataset does not contain textual data for TF-IDF")
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix
