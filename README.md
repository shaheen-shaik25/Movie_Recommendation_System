# 🎬 Movie Recommender System

> **An intelligent content-based movie recommendation system built with Python, Scikit-learn, Pandas, NLTK and Streamlit — featuring personalized recommendations, natural-language search, taste profiles, explainable recommendations, movie discovery, TMDB posters/trailers, watchlists, and offline evaluation.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-purple)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![TMDB](https://img.shields.io/badge/TMDB-API-blue)
![Tests](https://img.shields.io/badge/Tests-pytest-yellow)

---

## 🎥 Overview

**Movie Recommender System** is a machine-learning-powered movie discovery application built using the **TMDB 5000 Movie Dataset**.

The system uses **content-based recommendation** to identify movies with similar characteristics by analyzing:

- 🎞️ Movie overview
- 🎭 Genres
- 🏷️ Keywords
- 🎬 Director
- ⭐ Cast
- 📊 Ratings
- 🔥 Popularity
- 🌍 Language

Users can discover movies in several ways:

- Select a movie and find similar movies
- Describe the movie they want using natural language
- Combine multiple favourite movies into a personalized taste profile
- Browse movies by rating, popularity, release year, or hidden gems
- Filter and rank recommendations
- Save movies to a watchlist
- View posters and trailers through TMDB
- Understand **why** a movie was recommended

The project also includes an **offline evaluation framework** that compares the recommender against random and popularity-based baselines.

---

# ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Similar Movies** | Find movies with similar content using cosine similarity |
| 🔎 **Natural-Language Search** | Describe a movie, plot, actor, director, mood, or theme |
| 🧬 **Taste Profile** | Combine multiple favourite movies into one preference profile |
| 🎲 **Surprise Me** | Randomly select a movie and discover similar titles |
| 🧭 **Browse** | Explore Top Rated, Most Popular, Hidden Gems, Newest, and Random movies |
| 🎚️ **Advanced Filters** | Filter by genre, rating, year, and language |
| 🏆 **Multiple Ranking Modes** | Best Match, Highest Rated, Most Popular, Balanced |
| 💡 **Explainable Recommendations** | Shows shared director, cast, genres, and themes |
| 📌 **Watchlist** | Save movies and export your watchlist |
| 🖼️ **TMDB Posters** | Fetch real movie posters through the TMDB API |
| 🎬 **Trailers** | Provide trailer links through TMDB/YouTube |
| 📴 **Offline Fallback** | Generate placeholder posters when TMDB is unavailable |
| 📊 **Dataset Explorer** | Explore genres, ratings, directors, actors, and trends |
| 📈 **Offline Evaluation** | Evaluate recommendation quality against baseline systems |
| ⚡ **Sparse ML Model** | Computes similarity on demand using sparse matrices |
| 🎨 **Cinema UI** | Dark velvet-and-gold responsive interface |
| 🧪 **Automated Tests** | pytest test suite for the recommendation engine |
| 🐳 **Docker Support** | Containerized deployment support |

---

# 🖥️ Application

The application provides several movie discovery workflows.

## 🎯 1. Similar Movies

Select a movie and generate recommendations based on its content.

For example:

```text
Avatar
```

The system analyzes the movie's:

```text
Overview
Genres
Keywords
Director
Top Cast
```

and calculates similarity with the rest of the catalogue.

Each result can include:

- Poster
- Rating
- Match percentage
- Release year
- Runtime
- Genres
- Director
- Cast
- Explanation
- TMDB link
- Trailer

---

# 🔎 2. Describe a Movie

Instead of selecting a movie, users can describe what they want.

Example queries:

```text
astronauts stranded in space
```

```text
heist with a twist
```

```text
Tom Hanks
```

```text
superhero movie with time travel
```

The query is transformed using the same text-processing pipeline and compared with the movie catalogue using TF-IDF similarity.

---

# 🧬 3. Taste Profile

Users can select multiple favourite movies.

Example:

```text
Toy Story
Finding Nemo
The Incredibles
```

The system creates a combined taste profile by averaging the feature vectors of the selected movies.

```text
Movie A ─┐
Movie B ─┼──► Taste Profile ──► Similarity ──► Recommendations
Movie C ─┘
```

The watchlist can also be used as the source for a taste profile.

---

# 🎲 4. Surprise Me

Don't know what to watch?

The application can randomly select a movie and immediately generate recommendations.

---

# 🧭 5. Browse

Movies can be explored using:

- ⭐ Top Rated
- 🔥 Most Popular
- 💎 Hidden Gems
- 🆕 Newest
- 🎲 Random

The **Top Rated** view uses the Bayesian weighted rating rather than simply sorting by raw average rating.

---

# 🎚️ 6. Filters & Ranking

Recommendations can be filtered by:

- Genre
- Minimum rating
- Release year
- Language

Results can be ranked by:

### Best Match

Ranks according to content similarity.

### Highest Rated

Ranks according to Bayesian weighted rating.

### Most Popular

Ranks according to popularity.

### Balanced

Combines similarity and rating:

\[
Score =
0.7 \times Similarity
+
0.3 \times \frac{WeightedRating}{10}
\]

---

# 💡 7. Explainable Recommendations

The system provides a human-readable explanation for recommendations.

Possible explanations include:

```text
same director
```

```text
shared cast: Actor A, Actor B
```

```text
genres: Action, Adventure, Science Fiction
```

```text
themes: space, future, alien
```

If no strong metadata overlap is found, the system falls back to:

```text
similar plot and style
```

This makes the recommendation process more transparent instead of returning unexplained movie titles.

---

# 🧠 Machine Learning Pipeline

```text
              TMDB 5000 Dataset
                       │
                       ▼
              Merge Movie + Credits
                       │
                       ▼
               Data Cleaning
                       │
                       ▼
              Feature Engineering
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    Overview        Genres        Keywords
        │              │              │
        └──────────────┼──────────────┘
                       │
                  Top 3 Cast
                       │
                    Director
                       │
                       ▼
                 Combined Tags
                       │
                       ▼
                 Porter Stemming
                       │
                       ▼
             Bag-of-Words Matrix
                       │
                       ▼
                L2 Normalization
                       │
                       ▼
              Cosine Similarity
                       │
                       ▼
              Filtering + Ranking
                       │
                       ▼
               Recommendations
```

---

# 🔬 Feature Engineering

For every movie, the system constructs a combined `tags` representation from:

```text
Overview
+
Genres
+
Keywords
+
Top 3 Cast Members
+
Director
```

Multi-word metadata such as:

```text
Science Fiction
```

is normalized so that it remains a single feature token.

The text is then processed using **Porter stemming**.

---

# 📐 Cosine Similarity

The recommendation engine represents movies as numerical vectors.

For two vectors \(A\) and \(B\):

\[
CosineSimilarity(A,B)
=
\frac{A \cdot B}
{\|A\|\|B\|}
\]

A higher value indicates greater similarity in the selected feature space.

Because the feature matrix is normalized and sparse, similarity can be computed efficiently using sparse matrix multiplication.

---

# 🔎 TF-IDF Search

Natural-language search uses a TF-IDF representation.

The workflow is:

```text
User Query
    │
    ▼
Text preprocessing
    │
    ▼
TF-IDF vector
    │
    ▼
Compare with movie vectors
    │
    ▼
Cosine similarity
    │
    ▼
Rank results
```

This allows users to search for movies using concepts rather than exact titles.

---

# 🧬 Taste Profile Mathematics

If the user selects several favourite movies:

\[
A,B,C,\ldots
\]

the system creates a profile:

\[
Profile =
\frac{A+B+C+\cdots}{n}
\]

The resulting vector represents the user's combined content preferences.

The recommender then calculates similarity between this profile and every movie.

---

# ⭐ Bayesian Weighted Rating

Raw ratings can be misleading when movies have very different numbers of votes.

The system therefore calculates a Bayesian weighted rating:

\[
WR =
\frac{v}{v+m}R+
\frac{m}{v+m}C
\]

Where:

- \(R\) = movie's average rating
- \(v\) = number of votes
- \(m\) = minimum vote threshold
- \(C\) = average rating across the catalogue

This prevents movies with only a few votes from dominating the ranking.

---

# 📈 Evaluation

The TMDB 5000 dataset does **not contain real user preference histories**, so conventional recommender metrics such as user-level Precision@K or Recall@K cannot directly measure whether a recommendation matches an individual's taste.

Instead, this project includes an **offline evaluation framework** using proxy ground truth and two baseline systems:

- Random recommendation
- Most-popular recommendation

Evaluation is implemented in:

```text
recommender/evaluation.py
evaluate.py
```

---

## 📊 Evaluation Results

Evaluation was performed across **4,803 movies** at \(k=10\).

| Metric @ 10 | Content-Based | Random | Most Popular |
|---|---:|---:|---:|
| Genre Precision | **0.889** | 0.502 | 0.521 |
| Genre Jaccard | **0.484** | 0.170 | 0.149 |
| Franchise Hit | **0.440** | 0.000 | 0.012 |
| Franchise Recall | **0.393** | 0.000 | 0.004 |
| Diversity | 0.771 | **0.947** | 0.893 |
| Catalogue Coverage | **0.945** | 1.000 | 0.002 |
| Average Rating | 5.99 | 6.09 | **7.27** |

### Interpretation

The content-based recommender achieves high genre overlap because genre information is explicitly included in its features.

The **franchise metrics provide a more independent signal**, because the franchise grouping is generated separately from the recommendation features.

The system also reaches approximately **94.5% catalogue coverage**, indicating that recommendations are not concentrated almost entirely on the most popular movies.

The lower diversity compared with random recommendations is expected because the system is specifically designed to recommend similar movies.

The most-popular baseline produces a higher average rating, illustrating a trade-off between popularity/rating and content similarity.

> **Important:** These are proxy offline metrics, not measurements of actual user satisfaction or personalized recommendation accuracy.

---

## 🔬 Evaluation Methodology

The evaluation uses:

### Genre Precision@K

Percentage of recommendations sharing at least one genre with the query movie.

### Genre Jaccard@K

Measures the amount of genre overlap:

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

### Franchise Hit@K

Checks whether at least one related movie from the same detected title group appears in the top-K recommendations.

### Franchise Recall@K

Measures how many relevant franchise movies are retrieved.

### Diversity@K

Measures how different recommendations are from one another based on content similarity.

### Catalogue Coverage

Percentage of the complete movie catalogue that appears in recommendations across all evaluated queries.

---

# ⚠️ Evaluation Limitations

The evaluation should be interpreted carefully.

The TMDB dataset does not contain individual user-rating histories, meaning it cannot establish:

```text
"What movies would this specific user actually like?"
```

The franchise ground truth is also a proxy based on title patterns rather than official franchise metadata.

Therefore, these results demonstrate **offline recommendation behavior**, not real-world user satisfaction.

For true recommender-system evaluation, the next step would be to use a dataset such as **MovieLens**, where actual user ratings are available.

---

# ⚡ Model & Performance Design

The project avoids storing a large dense movie-to-movie similarity matrix.

Instead, it stores sparse feature representations and calculates similarity when needed.

### Earlier approach

```text
Movie Features
      ↓
Huge Similarity Matrix
      ↓
Large Storage Requirement
```

### Current approach

```text
Movie Features
      ↓
Sparse Matrix
      ↓
Similarity calculated on demand
```

### Benefits

- Smaller model artifact
- Lower storage requirements
- No large dense similarity matrix
- Efficient sparse operations
- Easier deployment
- Automatic model rebuilding when required

The project therefore no longer requires the old large `similarity.pkl`.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   TMDB 5000 CSVs   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Processing     │
                    │ data.py             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ Tags + Stemming     │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
              ┌──────────────┐    ┌──────────────┐
              │ Bag-of-Words │    │ TF-IDF       │
              │ Similarity   │    │ Search       │
              └──────┬───────┘    └──────┬───────┘
                     │                   │
                     └─────────┬─────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Recommendation      │
                    │ Engine               │
                    │ engine.py            │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
       Similar Movies     Taste Profile     Text Search
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Filters & Ranking   │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Frontend  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ TMDB API            │
                    │ Posters / Trailers  │
                    └─────────────────────┘
```

---

# 📁 Project Structure

```text
movie_recommender/
│
├── app.py
├── build_model.py
├── evaluate.py
│
├── recommender/
│   ├── __init__.py
│   ├── data.py
│   ├── engine.py
│   ├── evaluation.py
│   └── tmdb.py
│
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── artifacts/
│   └── model.joblib
│
├── tests/
│   └── test_engine.py
│
├── docs/
│   ├── evaluation_k5.csv
│   ├── evaluation_k10.csv
│   └── original_project_notes.txt
│
├── .streamlit/
│   └── config.toml
│
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── Procfile
├── setup.sh
├── run.sh
├── run.bat
└── README.md
```

---

# 🛠️ Tech Stack

## Programming

- Python 3.9+

## Machine Learning

- Scikit-learn
- NumPy
- SciPy
- Joblib

## Data Processing

- Pandas
- Python AST parsing

## NLP

- NLTK
- Porter Stemmer
- TF-IDF

## Frontend

- Streamlit
- HTML
- CSS

## External API

- TMDB API
- Requests

## Testing

- Pytest

## Deployment

- Streamlit Community Cloud
- Docker
- Heroku-style deployment configuration

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure TMDB

Create a local `.env` file using `.env.example`:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

The `.env` file must **not** be committed to GitHub.

Without a TMDB API key, the recommendation engine still works, but real posters and trailer information will not be available.

---

# ▶️ Run the Application

### Windows

```bash
run.bat
```

or:

```bash
streamlit run app.py
```

### macOS / Linux

```bash
./run.sh
```

or:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 🧠 Build the Model

The model can be rebuilt manually:

```bash
python build_model.py
```

The resulting artifact is stored in:

```text
artifacts/model.joblib
```

The application can also rebuild the cached model when required by the installed library versions.

---

# 🧪 Run Tests

Run the test suite:

```bash
pytest -q
```

The tests cover core recommendation-engine behavior such as:

- Movie retrieval
- Similarity recommendations
- Recommendation count
- Self-recommendation exclusion
- Genre filtering
- Rating filtering
- Year filtering
- Ranking modes
- Taste profiles
- Search
- Browse functionality
- Recommendation explanations

---

# 📊 Run Evaluation

Evaluate the complete catalogue:

```bash
python evaluate.py --k 10
```

For a faster sample:

```bash
python evaluate.py --k 10 --sample 500
```

Evaluation results are saved to:

```text
docs/evaluation_k10.csv
```

For \(k=5\):

```bash
python evaluate.py --k 5
```

Results:

```text
docs/evaluation_k5.csv
```

---

# 🔐 Security

Never store API keys directly inside source code.

Use:

```text
.env
```

for local development and environment variables/Streamlit Secrets for deployment.

The repository contains:

```text
.env.example
```

but should **not** contain the actual `.env`.

If an API key has ever been committed publicly:

1. Revoke the exposed key.
2. Generate a new key.
3. Remove the old key from the repository.
4. Check Git history for previous exposure.

---

# ☁️ Deployment

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a Streamlit Community Cloud application.
3. Select `app.py`.
4. Add the TMDB key under **Secrets**:

```toml
TMDB_API_KEY = "your_api_key"
```

5. Deploy.

---

## 🐳 Docker

Build:

```bash
docker build -t movie-recommender .
```

Run:

```bash
docker run -p 8501:8501 \
  -e TMDB_API_KEY=your_api_key \
  movie-recommender
```

Open:

```text
http://localhost:8501
```


---

# ⚠️ Limitations

The current system is primarily a **content-based recommender**.

Therefore:

- It does not learn individual user behavior.
- It does not use collaborative filtering.
- Recommendations depend on available movie metadata.
- Similarity is based primarily on textual/content features.
- Real user satisfaction cannot be measured using the TMDB dataset alone.
- TMDB posters and trailers require API access.

---

# 🚀 Future Improvements

## 🤝 Collaborative Filtering

Integrate actual user ratings using datasets such as MovieLens.

Potential approaches:

- Matrix Factorization
- SVD
- ALS
- Neural Collaborative Filtering

---

## 🧠 Semantic Embeddings

Replace or complement TF-IDF with transformer-based sentence embeddings.

```text
Movie Overview
      ↓
Sentence Embedding
      ↓
Vector Database
      ↓
Semantic Similarity
      ↓
Recommendations
```

This could improve recommendations for semantically similar descriptions that use different vocabulary.

---

## 👤 User Accounts

Add authentication and persistent profiles containing:

```text
Favourite Movies
Watch History
Ratings
Watchlist
Preferred Genres
```

---

## 🌐 REST API

Expose the recommendation engine through FastAPI.

Example architecture:

```text
Frontend
   │
   ▼
FastAPI
   │
   ▼
Recommendation Engine
   │
   ▼
ML Model
```

Potential endpoints:

```text
GET  /movies
GET  /recommend/{movie_id}
POST /recommend/profile
POST /search
POST /ratings
```

---

## 📈 Advanced Evaluation

With real user-rating data, evaluate using:

- Precision@K
- Recall@K
- F1@K
- NDCG@K
- MAP@K
- Hit Rate
- RMSE / MAE for rating prediction

---

# 🎓 What This Project Demonstrates

This project demonstrates practical experience with:

- End-to-end machine learning application development
- Content-based recommendation systems
- Natural Language Processing
- TF-IDF
- Bag-of-Words
- Cosine similarity
- Sparse matrix computation
- Feature engineering
- Bayesian ranking
- Personalized taste profiles
- Explainable recommendations
- Offline recommender evaluation
- Baseline comparison
- Streamlit application development
- API integration
- Environment-based secret management
- Automated testing
- Model serialization
- Docker deployment
- Git/GitHub project organization

---

# 📚 Dataset

This project uses the:

**TMDB 5000 Movie Dataset**

Dataset:

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

The project uses:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

---

# 🙏 Credits

Movie metadata is provided by the **TMDB 5000 Movie Dataset**.

This product uses the TMDB API but is **not endorsed or certified by TMDB**.

---

# 👨‍💻 Author

**Your Name**

Computer Science Undergraduate

### Interests

```text
Machine Learning
Artificial Intelligence
Data Science
Natural Language Processing
Recommendation Systems
ML Applications
```

### Project

🎬 **Movie Recommender System**

Built with:

```text
Python • Pandas • Scikit-learn • NLTK • Streamlit • TMDB API
```

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
