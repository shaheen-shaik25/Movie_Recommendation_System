"""Content-based recommendation engine (bag-of-words cosine similarity + TF-IDF search)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import build_movies, stem

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT = ROOT / "artifacts" / "model.joblib"

SORT_OPTIONS = ["Best match", "Highest rated", "Most popular", "Balanced (match + rating)"]


class Recommender:
    def __init__(self, movies: pd.DataFrame, count_matrix, tfidf_matrix, tfidf_vec):
        self.movies = movies.reset_index(drop=True)
        self.X = count_matrix      # L2-normalised bag-of-words -> dot product == cosine
        self.T = tfidf_matrix      # TF-IDF for free-text search
        self.tfidf_vec = tfidf_vec
        self._pos = {mid: i for i, mid in enumerate(self.movies["movie_id"])}

    # ------------------------------------------------------------------ build / load
    @classmethod
    def build(cls, data_dir: Path = DATA_DIR) -> "Recommender":
        movies = build_movies(data_dir)
        X = normalize(CountVectorizer(max_features=5000, stop_words="english").fit_transform(movies["tags"]))
        tfidf_vec = TfidfVectorizer(max_features=8000, stop_words="english", sublinear_tf=True)
        T = tfidf_vec.fit_transform(movies["tags"])
        return cls(movies, X.tocsr(), T.tocsr(), tfidf_vec)

    def save(self, path: Path = ARTIFACT) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"movies": self.movies, "X": self.X, "T": self.T, "vec": self.tfidf_vec,
             "sklearn": sklearn.__version__, "pandas": pd.__version__},
            path, compress=3,
        )

    @classmethod
    def load_or_build(cls, path: Path = ARTIFACT) -> "Recommender":
        """Load cached model; rebuild automatically if missing or from another library version."""
        if path.exists():
            try:
                blob = joblib.load(path)
                if blob.get("sklearn") == sklearn.__version__ and blob.get("pandas") == pd.__version__:
                    return cls(blob["movies"], blob["X"], blob["T"], blob["vec"])
            except Exception:
                pass
        model = cls.build()
        try:
            model.save(path)
        except OSError:
            pass
        return model

    # ------------------------------------------------------------------ lookups
    def __len__(self) -> int:
        return len(self.movies)

    @property
    def titles(self) -> list[str]:
        return sorted(self.movies["title"].tolist())

    @property
    def all_genres(self) -> list[str]:
        return sorted({g for gs in self.movies["genres_list"] for g in gs})

    @property
    def languages(self) -> list[str]:
        return self.movies["language"].value_counts().index.tolist()

    def get(self, movie_id: int) -> pd.Series:
        return self.movies.iloc[self._pos[movie_id]]

    def id_for_title(self, title: str) -> int:
        return int(self.movies.loc[self.movies["title"] == title, "movie_id"].iloc[0])

    # ------------------------------------------------------------------ filtering / ranking
    def _filter(self, df: pd.DataFrame, genres=None, min_rating=0.0, year_range=None, languages=None):
        if genres:
            df = df[df["genres_list"].apply(lambda g: all(x in g for x in genres))]
        if min_rating:
            df = df[df["vote_average"] >= min_rating]
        if year_range:
            df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
        if languages:
            df = df[df["language"].isin(languages)]
        return df

    def _rank(self, sims: np.ndarray, n: int, exclude_ids=(), sort_by="Best match", **filters) -> pd.DataFrame:
        df = self.movies.assign(similarity=sims)
        df = df[~df["movie_id"].isin(set(exclude_ids)) & (df["similarity"] > 0)]
        df = self._filter(df, **filters).nlargest(200, "similarity")
        if sort_by == "Highest rated":
            df = df.sort_values("weighted_rating", ascending=False)
        elif sort_by == "Most popular":
            df = df.sort_values("popularity", ascending=False)
        elif sort_by == "Balanced (match + rating)":
            df = df.assign(_s=0.7 * df["similarity"] + 0.3 * df["weighted_rating"] / 10)
            df = df.sort_values("_s", ascending=False).drop(columns="_s")
        return df.head(n).reset_index(drop=True)

    # ------------------------------------------------------------------ public features
    def recommend(self, movie_id: int, n: int = 5, sort_by: str = "Best match", **filters) -> pd.DataFrame:
        """Movies similar to one title."""
        i = self._pos[movie_id]
        sims = (self.X @ self.X[i].T).toarray().ravel()
        return self._rank(sims, n, exclude_ids=[movie_id], sort_by=sort_by, **filters)

    def recommend_for_profile(self, movie_ids: list[int], n: int = 10, sort_by: str = "Best match", **filters) -> pd.DataFrame:
        """Recommendations from a *taste profile*: the average of several favourite movies."""
        idx = [self._pos[m] for m in movie_ids if m in self._pos]
        if not idx:
            return self.movies.head(0).assign(similarity=[])
        profile = np.asarray(self.X[idx].mean(axis=0)).ravel()
        sims = np.asarray(self.X @ profile).ravel()
        return self._rank(sims, n, exclude_ids=movie_ids, sort_by=sort_by, **filters)

    def search(self, text: str, n: int = 10, **filters) -> pd.DataFrame:
        """Free-text search: describe a plot, mood, actor or director."""
        q = self.tfidf_vec.transform([stem(text)])
        if q.nnz == 0:
            return self.movies.head(0).assign(similarity=[])
        sims = (self.T @ q.T).toarray().ravel()
        return self._rank(sims, n, **filters)

    def browse(self, mode: str = "Top rated", n: int = 10, seed: int | None = None, **filters) -> pd.DataFrame:
        df = self._filter(self.movies, **filters)
        if mode == "Top rated":
            df = df[df["vote_count"] >= 200].sort_values("weighted_rating", ascending=False)
        elif mode == "Most popular":
            df = df.sort_values("popularity", ascending=False)
        elif mode == "Hidden gems":
            df = df[(df["vote_average"] >= 7.3) & df["vote_count"].between(100, 800)]
            df = df.sort_values("vote_average", ascending=False)
        elif mode == "Newest":
            df = df.sort_values("year", ascending=False)
        elif mode == "Random":
            df = df.sample(frac=1, random_state=seed)
        return df.head(n).assign(similarity=np.nan).reset_index(drop=True)

    def explain(self, source_id: int, other: pd.Series) -> str:
        """Short human-readable reason why `other` was recommended for `source_id`."""
        a = self.get(source_id)
        bits = []
        if a["director"] and a["director"] == other["director"]:
            bits.append(f"same director ({a['director']})")
        cast = [c for c in other["cast_list"] if c in a["cast_list"]]
        if cast:
            bits.append("shared cast: " + ", ".join(cast[:2]))
        genres = [g for g in other["genres_list"] if g in a["genres_list"]]
        if genres:
            bits.append("genres: " + ", ".join(genres[:3]))
        kws = [k for k in other["keywords_list"] if k in a["keywords_list"]]
        if kws:
            bits.append("themes: " + ", ".join(kws[:3]))
        return " · ".join(bits) if bits else "similar plot and style"
