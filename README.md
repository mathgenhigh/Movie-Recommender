# ML-Powered Movie Recommendation System
--- 

## About the Project

This project implements a **content-based recommendation system** that suggests movies similar to a given movie title or tag using TF-IDF vectorization and cosine similarity.
Additionally, it includes a ratings-based recommendation approach leveraging user ratings similarity.

### Features 

- Preprocessing of movie, tags, and ratings data.
- TF-IDF vectorization of textual features for similarity computation.
- Cosine similarity-based recommendations for:
    - Movie titles (based on genres/descriptions)
    - Movie tags (returning relevenat movie IDs)
    - User ratinfs (collaborative filtering style)
- Modular, reusable code with clear function annotations.
- Unit tests to ensure reliability.

--- 

## Built With

- Python 3.x
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pytest](https://docs.pytest.org/)

--- 

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/mathgenhigh/Movie-Recommender.git
   cd Movie-Recommender
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate    # On Linus/MacOS: source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

You can run the Jupyter notebook for interactive exploration:

    <pre><code>```bash jupyter notebook notebook/Movie_Recommender.ipynb```</code></pre>

Or import and use the modules directly in your Python scripts on REPL.
    
