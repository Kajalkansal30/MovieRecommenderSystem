# Movie Recommender System

A content-based movie recommendation system that suggests similar movies based on their content features like genres, keywords, cast, and crew.

## Overview

This system uses natural language processing and machine learning to recommend movies similar to a given movie. It analyzes various movie attributes including:
- Movie overview/description
- Genres
- Keywords
- Cast members
- Crew (directors)

## Features

- **Content-based filtering**: Recommends movies based on similarity of content features
- **Multi-attribute analysis**: Considers genres, keywords, cast, and crew for recommendations
- **Pre-trained model**: Uses pre-computed similarity matrix for fast recommendations
- **Scalable**: Built with vectorization and cosine similarity for efficient processing

## Dataset

The system uses the TMDB 5000 Movies dataset containing:
- **tmdb_5000_movies.csv**: Movie metadata including genres, overview, etc.
- **tmdb_5000_credits.csv**: Cast and crew information

**Total movies**: 4,803 after preprocessing

## How It Works

1. **Data Preprocessing**:
   - Merges movies and credits datasets
   - Extracts relevant features (genres, keywords, cast, crew)
   - Cleans and tokenizes text data

2. **Feature Engineering**:
   - Creates a combined "tags" feature from all relevant attributes
   - Removes spaces from multi-word features
   - Joins all features into a single text string

3. **Vectorization**:
   - Uses CountVectorizer with 5000 max features
   - Removes English stopwords
   - Creates a sparse matrix representation

4. **Similarity Calculation**:
   - Computes cosine similarity between all movie vectors
   - Stores similarity matrix for fast retrieval

5. **Recommendation**:
   - Finds the most similar movies based on cosine similarity scores
   - Returns top 5 recommendations excluding the input movie

## Usage

### Basic Recommendation

```python
import pickle

# Load the pre-trained model
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie_title):
    """Recommend similar movies"""
    # Implementation available in Recommender.ipynb
    pass

# Example usage
recommend('The Dark Knight')
```

### Available Functions

- `recommend(movie_name)`: Returns 5 similar movie recommendations
- Pre-loaded data: `movie_list.pkl` (movie dataframe) and `similarity.pkl` (similarity matrix)

## File Structure

```
MovieRecommenderSystemPython/
├── Recommender.ipynb      # Main implementation notebook
├── movie_list.pkl         # Preprocessed movie dataset
├── similarity.pkl         # Pre-computed similarity matrix
├── tmdb_5000_movies.csv   # Original movie metadata
├── tmdb_5000_credits.csv  # Original credits data
└── README.md             # This file
```

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- pickle (standard library)

## Installation

1. Clone the repository
2. Ensure all required packages are installed:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. The system is ready to use with the pre-trained model files

## Example Recommendations

| Input Movie | Recommended Movies |
|-------------|-------------------|
| The Dark Knight | The Dark Knight Rises, Batman Begins, Iron Man, Inception, The Avengers |
| Titanic | Pearl Harbor, The Notebook, Atonement, Revolutionary Road, The Great Gatsby |

## Technical Details

- **Vectorization**: CountVectorizer with 5000 features
- **Similarity Metric**: Cosine similarity
- **Processing Time**: ~2-3 seconds for initial setup
- **Memory Usage**: ~50MB for similarity matrix

## Future Improvements

- Add collaborative filtering for personalized recommendations
- Implement real-time recommendation API
- Add movie posters and additional metadata
- Create web interface for user interaction
- Add rating-based filtering

## License

This project uses the TMDB 5000 Movies dataset which is publicly available for educational and research purposes.
