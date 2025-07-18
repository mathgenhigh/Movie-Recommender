import pandas as pd 
from typing import List
from scipy.sparse import csc_matrix 
from sklearn.metrics.pairwise import cosine_similarity

def build_ratings_similarity_matrix(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build movie-to-movie similarity matrix from ratings data.
    
    Parameters:
        ratings_df: Dataframe with columns['userId', 'movieId', 'rating']
        movies_df: DataFrame with columns ['movieId', 'title']
        
    Returns:
        similarity_df: DataFrame indexed by movie titles with cosine similarity values.
    """
    
    movie_user_matrix = ratings_df.pivot(index="movieId", columns="userId", values='rating').fillna(0)
    
    similarity = cosine_similarity(movie_user_matrix)
    
    similarity_df = pd.DataFrame(similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)
    
    similarity_df.index = similarity_df.index.map(movies_df.set_index('movieId')['title'])
    similarity_df.columns = similarity_df.columns.map(movies_df.set_index('movieId')['title'])
    
    return similarity_df

def recommend_movies(
    title: str,
    df: pd.DataFrame,
    tfidf_matrix: csc_matrix,
    dataset: str = "movies",
    top_n: int = 10
) -> List[str]:
    """
    Recommend similar movies or tags based on the dataset type using cosine similarity.
    
    Parameters:
        title (str):
            The movie title or tag to base recommendations on.
        df (pd.DataFrame):
            The DataFrame containing the dataset.
        tfidf_matrix (csr_matrix):
            The TF-IDF matrix computed from the dataset.
        dataset (str, default "movies"):
            Dataset type.
        top_n (int, default 10):
            Number of recommendations to return.
            
    Returns List[str]:
        List of recommended movie titles or tags.
    """
    
    if dataset == "movies":
        if title not in df['title'].values:
            raise ValueError(f"Title '{title}' not found in movies dataset.")
        idx = df.index[df['title'] == title][0]
        
        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
        
        return df['title'].iloc[similar_indices].tolist()
    
    elif dataset == "tags":
        matching_movie_ids = df.loc[df['tag'].str.lower() == title.lower(), 'movieId'].unique()
        if len(matching_movie_ids) == 0:
            raise ValueError(f"Tag: '{title}' not found in tags dataset.")
        return matching_movie_ids.tolist()
    
    elif dataset == "ratings":
        similarity_df = tfidf_matrix
        
        if title not in similarity_df.index:
            raise ValueError(f"TItle '{title}' not found in ratings similarity matrix.")

        similarity_scores = similarity_df.loc[title]
        similar_titles = similarity_scores.sort_values(ascending=False).iloc[1:top_n+1].index.tolist()
        
        return similar_titles

    else:
        raise ValueError(f"Unknown dataset type: {dataset}")