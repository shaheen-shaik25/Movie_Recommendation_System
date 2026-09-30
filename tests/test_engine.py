import pytest

from recommender import Recommender


@pytest.fixture(scope="module")
def rec():
    return Recommender.load_or_build()


def test_dataset_size(rec):
    assert len(rec) > 4500


def test_recommend_excludes_self_and_respects_n(rec):
    mid = rec.id_for_title("Avatar")
    out = rec.recommend(mid, n=8)
    assert len(out) == 8
    assert mid not in out["movie_id"].values
    assert out["similarity"].is_monotonic_decreasing


def test_related_movies_make_sense(rec):
    out = rec.recommend(rec.id_for_title("The Dark Knight Rises"), n=5)
    assert any("Batman" in t for t in out["title"])


def test_filters_applied(rec):
    out = rec.recommend(rec.id_for_title("Toy Story"), n=10, genres=["Animation"], min_rating=6.5, year_range=(2000, 2016))
    assert len(out) > 0
    assert all("Animation" in g for g in out["genres_list"])
    assert (out["vote_average"] >= 6.5).all() and out["year"].between(2000, 2016).all()


def test_sort_by_rating(rec):
    out = rec.recommend(rec.id_for_title("Avatar"), n=10, sort_by="Highest rated")
    assert out["weighted_rating"].is_monotonic_decreasing


def test_profile_recommendations(rec):
    ids = [rec.id_for_title("Toy Story"), rec.id_for_title("Finding Nemo")]
    out = rec.recommend_for_profile(ids, n=6)
    assert len(out) == 6 and not set(ids) & set(out["movie_id"])


def test_free_text_search(rec):
    out = rec.search("astronauts stranded in space", n=10)
    assert len(out) > 0
    assert any("Science Fiction" in g for g in out["genres_list"])
    assert rec.search("qwertyzxcv", n=5).empty


@pytest.mark.parametrize("mode", ["Top rated", "Most popular", "Hidden gems", "Newest", "Random"])
def test_browse_modes(rec, mode):
    assert len(rec.browse(mode, n=10, seed=1)) == 10


def test_explain_returns_text(rec):
    a = rec.id_for_title("The Dark Knight Rises")
    other = rec.recommend(a, n=1).iloc[0]
    assert isinstance(rec.explain(a, other), str)


def test_evaluation_beats_baselines(rec):
    from recommender.evaluation import evaluate

    t = evaluate(rec, k=5, sample=150, seed=1)
    m, r = t.loc["Content-based (this app)"], t.loc["Random baseline"]
    assert m["Genre precision@5"] > r["Genre precision@5"] + 0.2
    assert m["Franchise hit@5"] > r["Franchise hit@5"]
