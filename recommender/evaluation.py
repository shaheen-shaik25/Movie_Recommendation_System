"""Offline evaluation of the recommender.

The TMDB dataset has no user ratings, so classic "accuracy" cannot be measured. Instead we use
ranking metrics against proxy ground truth, and compare with two naive baselines
(random and most-popular) so the numbers have context.

Metrics (all @k):
  * Genre precision  - share of recommendations having >=1 genre in common with the query movie
  * Genre Jaccard    - average genre overlap (0-1) between query and each recommendation
  * Franchise hit    - ground truth independent of the features: movies sharing the first two words of
                       the title (e.g. "Toy Story", "Toy Story 2"). Hit = at least one is in the top-k
  * Franchise recall - fraction of the query's franchise found in the top-k
  * Diversity        - average pairwise dissimilarity (1 - cosine) inside a recommendation list
  * Coverage         - share of the catalogue that ever gets recommended
  * Avg rating       - mean vote_average of recommended movies
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .engine import Recommender

_ARTICLES = {"the", "a", "an"}


def franchise_key(title: str) -> str | None:
    words = [w for w in re.findall(r"[a-z0-9]+", title.lower())]
    while words and words[0] in _ARTICLES:
        words = words[1:]
    if len(words) < 2:
        return None
    key = " ".join(words[:2])
    return key if len(key) >= 8 else None


def franchise_groups(movies: pd.DataFrame) -> dict[int, set[int]]:
    """movie_id -> ids of the *other* movies in the same franchise (groups of 2-10 titles)."""
    groups = defaultdict(set)
    for mid, title in zip(movies["movie_id"], movies["title"]):
        key = franchise_key(title)
        if key:
            groups[key].add(mid)
    out = {}
    for members in groups.values():
        if 2 <= len(members) <= 10:
            for mid in members:
                out[mid] = members - {mid}
    return out


def _genre_scores(query_genres: list[str], recs: pd.DataFrame) -> tuple[float, float]:
    q = set(query_genres)
    if not q or recs.empty:
        return np.nan, np.nan
    overlaps = [len(q & set(g)) for g in recs["genres_list"]]
    jaccard = [len(q & set(g)) / len(q | set(g)) for g in recs["genres_list"]]
    return float(np.mean([o > 0 for o in overlaps])), float(np.mean(jaccard))


def _diversity(rec: Recommender, ids: list[int]) -> float:
    if len(ids) < 2:
        return np.nan
    idx = [rec._pos[i] for i in ids]
    sims = (rec.X[idx] @ rec.X[idx].T).toarray()
    n = len(idx)
    return float(1 - (sims.sum() - np.trace(sims)) / (n * (n - 1)))


def evaluate(rec: Recommender, k: int = 10, sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Return a table of metrics for the model and the two baselines."""
    rng = np.random.default_rng(seed)
    movies = rec.movies
    ids = movies["movie_id"].to_numpy()
    query_ids = rng.choice(ids, size=min(sample, len(ids)), replace=False) if sample else ids
    fran = franchise_groups(movies)
    popular = movies.sort_values("popularity", ascending=False)

    systems = {"Content-based (this app)": lambda mid: rec.recommend(int(mid), n=k),
               "Random baseline": lambda mid: movies[movies["movie_id"] != mid].sample(k, random_state=int(rng.integers(1 << 31))),
               "Most-popular baseline": lambda mid: popular[popular["movie_id"] != mid].head(k)}
    rows = []
    for name, fn in systems.items():
        prec, jac, div, rating = [], [], [], []
        hit, recall, seen = [], [], set()
        for mid in query_ids:
            recs = fn(mid)
            rec_ids = recs["movie_id"].tolist()
            seen.update(rec_ids)
            p, j = _genre_scores(rec.get(int(mid))["genres_list"], recs)
            prec.append(p); jac.append(j); div.append(_diversity(rec, rec_ids)); rating.append(recs["vote_average"].mean())
            if int(mid) in fran:
                relevant = fran[int(mid)]
                found = len(relevant & set(rec_ids))
                hit.append(found > 0)
                recall.append(found / min(len(relevant), k))
        rows.append({
            "System": name,
            f"Genre precision@{k}": np.nanmean(prec),
            f"Genre Jaccard@{k}": np.nanmean(jac),
            f"Franchise hit@{k}": np.mean(hit),
            f"Franchise recall@{k}": np.mean(recall),
            f"Diversity@{k}": np.nanmean(div),
            "Coverage": len(seen) / len(movies),
            "Avg rating": np.nanmean(rating),
        })
    return pd.DataFrame(rows).set_index("System")
