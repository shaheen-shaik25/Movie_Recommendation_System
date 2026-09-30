"""Dataset loading and preprocessing for the TMDB 5000 movie dataset."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from nltk.stem.porter import PorterStemmer

_stemmer = PorterStemmer()


def parse_names(text, limit: int | None = None, key: str = "name") -> list[str]:
    """Parse TMDB's JSON-ish string columns into a list of names."""
    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    names = [i[key] for i in items if key in i]
    return names[:limit] if limit else names


def director_of(text) -> str:
    try:
        for member in ast.literal_eval(text):
            if member.get("job") == "Director":
                return member.get("name", "")
    except (ValueError, SyntaxError):
        pass
    return ""


def stem(text: str) -> str:
    return " ".join(_stemmer.stem(w) for w in text.lower().split())


def _squash(names: list[str]) -> list[str]:
    """'Science Fiction' -> 'sciencefiction' so multi-word names stay one token."""
    return [n.replace(" ", "").lower() for n in names]


def build_movies(data_dir: str | Path) -> pd.DataFrame:
    """Merge the two CSVs and return one tidy DataFrame (with a `tags` column)."""
    data_dir = Path(data_dir)
    movies = pd.read_csv(data_dir / "tmdb_5000_movies.csv")
    credits = pd.read_csv(data_dir / "tmdb_5000_credits.csv")
    df = movies.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id")
    df = df.drop_duplicates("movie_id").reset_index(drop=True)

    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")
    df["genres_list"] = df["genres"].apply(parse_names)
    df["keywords_list"] = df["keywords"].apply(parse_names)
    df["cast_list"] = df["cast"].apply(lambda t: parse_names(t, 5))
    df["director"] = df["crew"].apply(director_of)
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    df["runtime"] = df["runtime"].fillna(0).astype(int)

    # Bayesian weighted rating (IMDB formula): fair ranking that accounts for vote count.
    m = df["vote_count"].quantile(0.80)
    c = df["vote_average"].mean()
    v, r = df["vote_count"], df["vote_average"]
    df["weighted_rating"] = (v / (v + m)) * r + (m / (v + m)) * c

    def make_tags(row) -> str:
        parts = (
            row["overview"].split()
            + _squash(row["genres_list"])
            + _squash(row["keywords_list"])
            + _squash(row["cast_list"][:3])
            + _squash([row["director"]] if row["director"] else [])
        )
        return stem(" ".join(parts))

    df["tags"] = df.apply(make_tags, axis=1)

    keep = [
        "movie_id", "title", "year", "overview", "tagline", "genres_list", "keywords_list",
        "cast_list", "director", "runtime", "original_language", "vote_average",
        "vote_count", "popularity", "weighted_rating", "tags",
    ]
    return df[keep].rename(columns={"original_language": "language"})
