# 🎬 Movie Recommender System

A content-based movie recommender built with **Python, scikit-learn and Streamlit**, using the
[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
Pick a movie (or describe one) and get recommendations with posters, match scores and the reason behind each pick.

## ✨ Features

| Feature | What it does |
|---|---|
| 🎯 **Similar movies** | Choose a title and get the closest matches with a **% match score** and a "Why?" explanation (shared director, cast, genres, themes). |
| 🎲 **Surprise me** | Picks a random movie to start from. |
| 🔎 **Describe a movie** | Free-text search: *"astronauts stranded in space"*, *"heist with a twist"*, *"Tom Hanks"* (TF-IDF). |
| 🧬 **Taste profile** | Select several favourites (or use your watchlist) and get recommendations that blend them all. |
| 🧭 **Browse** | Top rated (weighted), Most popular, Hidden gems, Newest, Random. |
| 📊 **Explore** | Dataset dashboard: genres, ratings per decade, top directors and actors. |
| 🎚️ **Filters & ranking** | Genre, minimum rating, release year, language; rank by best match, rating, popularity, or a balanced score. |
| 📌 **Watchlist** | Save movies while you browse and download them as CSV. |
| 🖼️ **Posters & trailers** | Fetched in parallel from TMDB; generated placeholder posters if offline or no API key. |
| 🎨 **Cinema-style UI** | Dark velvet-and-gold theme, poster-first cards with rating badges, match bars and rank badges, a featured-movie panel, helpful empty states, and a responsive mobile layout. |
| ⚡ **Lightweight model** | Similarity is computed on demand from a sparse matrix, so the old 185 MB `similarity.pkl` is no longer needed. |

## 🚀 Quick start

Requires **Python 3.9+**.

**Easiest (one command)**

```bash
./run.sh        # macOS / Linux
run.bat         # Windows (double-click)
```

**Manual**

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

The first launch may take ~15 seconds if the cached model in `artifacts/` was built with different
library versions; it is then rebuilt automatically. You can also do it manually: `python build_model.py`.

## 🔑 TMDB API key (for real posters)

1. Get a free key at <https://www.themoviedb.org/settings/api>.
2. Copy `.env.example` to `.env` and set `TMDB_API_KEY=your_key`.
   (On Streamlit Cloud, put `TMDB_API_KEY = "your_key"` in the app's *Secrets* instead.)

Without a key everything still works; posters are just generated placeholders.

> ⚠️ **Security note:** your original `app.py` had an API key hard-coded in the source. It now lives in `.env`
> (which is git-ignored). If that code was ever pushed to a public repo, **generate a new key** on TMDB and replace it.

## 🧠 How it works

```
CSV data ─► merge on movie id ─► tags = overview + genres + keywords + top-3 cast + director
        ─► Porter stemming ─► Bag-of-Words (5000 features) ─► L2-normalise
        ─► cosine similarity (sparse dot product) ─► filter ─► rank ─► cards
```

* **Similar movies / taste profile** – cosine similarity over bag-of-words vectors. A taste profile averages the vectors of your favourites.
* **Free-text search** – the query is stemmed the same way and scored against a TF-IDF matrix.
* **Weighted rating** – IMDB's Bayesian formula `WR = v/(v+m)·R + m/(v+m)·C`, so a 9.0 from 3 votes doesn't outrank an 8.3 from 10,000.
* **Balanced ranking** – `0.7 × similarity + 0.3 × weighted rating/10` among the 200 closest matches.

## 📁 Project structure

```
movie_recommender/
├── app.py                 # Streamlit UI
├── build_model.py         # Rebuilds artifacts/model.joblib from the CSVs
├── recommender/
│   ├── data.py            # Loading, cleaning, tag creation
│   ├── engine.py          # Recommender class (similar, profile, search, browse, explain)
│   └── tmdb.py            # Poster/trailer fetching, placeholder posters
├── data/                  # TMDB 5000 CSVs
├── artifacts/model.joblib # Pre-built model (auto-rebuilt if missing/incompatible)
├── tests/test_engine.py   # pytest suite
├── docs/original_project_notes.txt
├── requirements.txt  .env.example  .gitignore
├── Procfile  setup.sh     # Heroku-style deployment (from the original project)
├── Dockerfile
└── run.sh  run.bat        # one-command launchers
```

## 🧪 Tests

```bash
pytest -q
```

## ☁️ Deployment

* **Streamlit Community Cloud** – push to GitHub, select `app.py`, add `TMDB_API_KEY` under *Secrets*.
* **Heroku** – `Procfile` and `setup.sh` are included; set the `TMDB_API_KEY` config var.
* **Docker** – `docker build -t movies . && docker run -p 8501:8501 -e TMDB_API_KEY=your_key movies`

## 🛠️ What changed from the original project

* Fixed `requirements.txt` (`pickle` is built in and broke `pip install`) and added missing dependencies.
* Removed the hard-coded API key and added error handling (the old code crashed when a poster was missing or the API failed).
* Replaced the fixed 5-result layout with configurable results, filters and ranking.
* Replaced the 185 MB similarity matrix and the two pickles with a compact 4 MB model plus on-demand similarity.
* Redesigned the interface (custom theme in `.streamlit/config.toml` and CSS in `app.py`).
* Fixed the wrong selectbox label ("How would you like to be contacted") and the title typo.
* Added tests, Docker support, launchers and this README.

## 🔭 Ideas for next steps

Collaborative filtering with user ratings, sentence-embedding similarity, user accounts, and a REST API (FastAPI) in front of `recommender.engine`.

## 📄 Credits

Data: TMDB 5000 Movie Dataset. This product uses the TMDB API but is not endorsed or certified by TMDB.
