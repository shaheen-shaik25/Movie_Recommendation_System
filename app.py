"""Movie Recommender - Streamlit app.  Run with:  streamlit run app.py"""
import base64
import html
import random
import urllib.parse

import pandas as pd
import streamlit as st

from recommender import SORT_OPTIONS, Recommender, tmdb

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# ----------------------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Young+Serif&display=swap');

:root {
  --ink: #160c15; --velvet: #231320; --velvet-2: #311b2b; --gold: #e8b563;
  --ivory: #f3e9dc; --muted: #b7a3a6; --line: rgba(243,233,220,.13);
  --display: 'Young Serif', Georgia, 'Times New Roman', serif;
  --body: 'Figtree', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input, .stApp textarea,
.stApp [data-testid="stCaptionContainer"] { font-family: var(--body); }
.stApp h1, .stApp h2, .stApp h3 { font-family: var(--display); font-weight: 400; letter-spacing: .005em; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }

/* ---------- hero ---------- */
.hero { position: relative; padding: 2.6rem 2.4rem 3rem; border-radius: 18px; margin-bottom: 1.6rem;
  background: radial-gradient(110% 160% at 0% 0%, #5a2342 0%, rgba(90,35,66,0) 58%), var(--velvet);
  border: 1px solid var(--line); overflow: hidden; animation: rise .7s ease-out both; }
.hero h1 { font-size: clamp(2rem, 4.6vw, 3.4rem); line-height: 1.08; margin: 0 0 .7rem; max-width: 16ch; padding: 0; }
.hero p { color: var(--muted); font-size: 1.08rem; max-width: 52ch; margin: 0; line-height: 1.55; }
.hero .bulbs { position: absolute; left: 0; right: 0; bottom: 0; height: 26px;
  background: radial-gradient(circle, var(--gold) 0 3px, rgba(232,181,99,.25) 4px 7px, transparent 8px) 0 50% / 26px 26px repeat-x; }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: .7rem 1rem; font-size: 1.02rem; font-weight: 500; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--gold); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--gold); height: 3px; }

/* ---------- buttons, inputs, expanders ---------- */
.stApp [data-testid^="stBaseButton"] { border-radius: 999px; font-weight: 600; transition: border-color .15s, background .15s; }
.stApp [data-testid="stBaseButton-secondary"] { background: transparent; border: 1px solid var(--line); color: var(--ivory); }
.stApp [data-testid="stBaseButton-secondary"]:hover { border-color: var(--gold); color: var(--gold); }
.stApp [data-testid="stBaseButton-primary"] { background: var(--gold); color: #2a1426; border: 1px solid var(--gold); }
.stApp [data-testid="stBaseButton-primary"]:hover { background: #f1c887; color: #2a1426; }
.stApp :focus-visible { outline: 2px solid var(--gold); outline-offset: 2px; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; background: transparent; }
[data-testid="stExpander"] summary { padding: .35rem .7rem; font-size: .9rem; color: var(--muted); }
[data-testid="stMetric"] { background: var(--velvet); border: 1px solid var(--line); border-left: 3px solid var(--gold);
  border-radius: 10px; padding: .8rem 1rem; }
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
.brand { font-family: var(--display); font-size: 1.35rem; margin: .2rem 0 .8rem; color: var(--ivory); }
.brand i { color: var(--gold); font-style: normal; }

/* ---------- movie card ---------- */
.card .poster { position: relative; aspect-ratio: 2 / 3; border-radius: 10px; overflow: hidden;
  background: var(--velvet-2); border: 1px solid var(--line); box-shadow: 0 10px 24px -14px rgba(0,0,0,.8); }
.card .poster img { width: 100%; height: 100%; object-fit: cover; display: block; }
.card .badge { position: absolute; top: 8px; padding: 3px 9px; border-radius: 999px; font-size: .78rem; font-weight: 700;
  background: rgba(22,12,21,.82); color: var(--ivory); backdrop-filter: blur(4px); }
.card .badge.rating { left: 8px; } .card .badge.rating b { color: var(--gold); }
.card .badge.rank { right: 8px; background: var(--gold); color: #2a1426; }
.card .title { font-weight: 700; font-size: 1rem; line-height: 1.25; margin-top: .6rem; min-height: 2.5em; color: var(--ivory); }
.card .sub { color: var(--muted); font-size: .85rem; margin-top: .15rem; }
.card .chips { margin-top: .4rem; display: flex; flex-wrap: wrap; gap: 4px; min-height: 24px; }
.chip { font-size: .72rem; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
.match { display: flex; align-items: center; gap: 8px; margin: .55rem 0 .5rem; font-size: .82rem; color: var(--muted); }
.match .bar { flex: 1; height: 5px; border-radius: 3px; background: var(--velvet-2); overflow: hidden; }
.match .bar i { display: block; height: 100%; background: var(--gold); border-radius: 3px; }
.match b { color: var(--gold); min-width: 2.6em; text-align: right; }

/* ---------- featured movie ---------- */
.feature { display: flex; gap: 1.6rem; align-items: flex-start; padding: 1.2rem; border-radius: 14px;
  background: var(--velvet); border: 1px solid var(--line); margin: .4rem 0 1.6rem; }
.feature .poster { width: 150px; flex: none; aspect-ratio: 2 / 3; border-radius: 8px; overflow: hidden; border: 1px solid var(--line); }
.feature .poster img { width: 100%; height: 100%; object-fit: cover; display: block; }
.feature h2 { margin: 0 0 .25rem; font-size: 1.9rem; padding: 0; }
.feature .sub { color: var(--muted); margin-bottom: .6rem; }
.feature p { color: var(--ivory); opacity: .88; line-height: 1.6; max-width: 70ch; margin: .7rem 0; }
.feature .people { color: var(--muted); font-size: .92rem; } .feature .people b { color: var(--ivory); font-weight: 600; }

/* ---------- empty state ---------- */
.empty { text-align: center; padding: 2.6rem 1rem; border: 1px dashed var(--line); border-radius: 14px; color: var(--muted); }
.empty strong { display: block; font-family: var(--display); font-size: 1.25rem; color: var(--ivory); margin-bottom: .3rem; font-weight: 400; }

h2.section { font-size: 1.5rem; margin: .2rem 0 1rem; padding: 0; }

@media (max-width: 640px) {
  .hero { padding: 1.6rem 1.2rem 2.2rem; }
  .feature { flex-direction: column; }
  .feature .poster { width: 120px; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .8rem; }
  [data-testid="stColumn"] { min-width: calc(50% - .8rem) !important; flex: 1 1 calc(50% - .8rem) !important; }
}
@media (prefers-reduced-motion: reduce) { .hero { animation: none; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the recommendation model (first run can take ~15 seconds)...")
def get_engine() -> Recommender:
    return Recommender.load_or_build()


@st.cache_data(show_spinner=False)
def placeholder_uri(title: str) -> str:
    return "data:image/png;base64," + base64.b64encode(tmdb.placeholder_poster(title)).decode()


def api_key() -> str:
    key = tmdb.get_api_key()
    if not key:
        try:
            key = str(st.secrets.get("TMDB_API_KEY", "")).strip()
        except Exception:
            key = ""
    return key


rec = get_engine()
KEY = api_key()
LANG_NAMES = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "ja": "Japanese",
              "zh": "Chinese", "hi": "Hindi", "it": "Italian", "ko": "Korean", "ru": "Russian",
              "pt": "Portuguese", "cn": "Cantonese", "te": "Telugu", "ta": "Tamil"}
GOLD = "#e8b563"

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []


def toggle_watch(movie_id: int) -> None:
    wl = st.session_state.watchlist
    wl.remove(movie_id) if movie_id in wl else wl.append(movie_id)


def poster_src(info: dict, title: str) -> str:
    return info.get("poster") or placeholder_uri(title)


def chips(genres, limit=3) -> str:
    return "".join(f"<span class='chip'>{html.escape(g)}</span>" for g in genres[:limit])


def empty_state(title: str, hint: str) -> None:
    st.markdown(f"<div class='empty'><strong>{html.escape(title)}</strong>{html.escape(hint)}</div>",
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------- rendering
def card_html(m: pd.Series, poster: str, rank: int | None) -> str:
    title = html.escape(m["title"])
    meta = " · ".join(str(x) for x in [m["year"] or "N/A", f"{m['runtime']} min" if m["runtime"] else ""] if x)
    rank_badge = f"<span class='badge rank'>#{rank}</span>" if rank else ""
    match = ""
    if pd.notna(m.get("similarity")):
        pct = round(m["similarity"] * 100)
        match = f"<div class='match'><div class='bar'><i style='width:{pct}%'></i></div><b>{pct}%</b></div>"
    return (
        f"<div class='card'><div class='poster'><img src='{poster}' alt='Poster of {title}' loading='lazy'>"
        f"<span class='badge rating'><b>★</b> {m['vote_average']:.1f}</span>{rank_badge}</div>"
        f"<div class='title'>{title}</div><div class='sub'>{meta}</div>"
        f"<div class='chips'>{chips(m['genres_list'], 2)}</div>{match}</div>"
    )


def render_cards(df: pd.DataFrame, tab: str, source_id: int | None = None, ranked: bool = False) -> None:
    if df.empty:
        empty_state("No movies match these filters", "Lower the minimum rating or widen the year range in the sidebar.")
        return
    infos = {}
    if KEY and show_posters:
        with st.spinner("Fetching posters..."):
            infos = tmdb.fetch_many(df["movie_id"], KEY)
    for start in range(0, len(df), per_row):
        cols = st.columns(per_row)
        for offset, (col, (_, m)) in enumerate(zip(cols, df.iloc[start:start + per_row].iterrows())):
            mid = int(m["movie_id"])
            info = infos.get(mid, {})
            with col:
                st.markdown(card_html(m, poster_src(info, m["title"]), start + offset + 1 if ranked else None),
                            unsafe_allow_html=True)
                with st.expander("Details"):
                    st.write(m["overview"] or "No overview available.")
                    if m["director"]:
                        st.markdown(f"**Director:** {m['director']}")
                    if m["cast_list"]:
                        st.markdown("**Cast:** " + ", ".join(m["cast_list"]))
                    if source_id is not None:
                        st.markdown(f"**Why it matches:** {rec.explain(source_id, m)}")
                    q = urllib.parse.quote_plus(f"{m['title']} {m['year']} trailer")
                    trailer = info.get("trailer") or "https://www.youtube.com/results?search_query=" + q
                    st.markdown(f"[TMDB](https://www.themoviedb.org/movie/{mid}) · "
                                f"[{'Watch trailer' if info.get('trailer') else 'Find trailer'}]({trailer})")
                saved = mid in st.session_state.watchlist
                st.button("✓ Saved" if saved else "＋ Watchlist", key=f"wl_{tab}_{mid}",
                          type="primary" if saved else "secondary", on_click=toggle_watch, args=(mid,),
                          use_container_width=True)


def render_feature(m: pd.Series) -> None:
    poster = poster_src(tmdb.fetch_info(int(m["movie_id"]), KEY) if (KEY and show_posters) else {}, m["title"])
    meta = " · ".join(str(x) for x in [m["year"] or "N/A", f"{m['runtime']} min" if m["runtime"] else "",
                                       f"★ {m['vote_average']:.1f}"] if x)
    people = ""
    if m["director"]:
        people += f"<b>Directed by</b> {html.escape(m['director'])}<br>"
    if m["cast_list"]:
        people += "<b>Starring</b> " + html.escape(", ".join(m["cast_list"][:4]))
    st.markdown(
        f"<div class='feature'><div class='poster'><img src='{poster}' alt='Poster of {html.escape(m['title'])}'></div>"
        f"<div><h2>{html.escape(m['title'])}</h2><div class='sub'>{meta}</div>{chips(m['genres_list'], 4)}"
        f"<p>{html.escape(m['overview'])}</p><div class='people'>{people}</div></div></div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("<div class='brand'>Movie <i>Recommender</i></div>", unsafe_allow_html=True)
    st.subheader("Results")
    n_results = st.slider("How many movies", 5, 20, 10)
    per_row = st.slider("Cards per row", 3, 6, 5)
    sort_by = st.selectbox("Rank by", SORT_OPTIONS)
    show_posters = st.toggle("Show posters", value=True, help="Turn off for a faster page on slow connections.")

    st.subheader("Filters")
    f_genres = st.multiselect("Must include genres", rec.all_genres)
    f_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)
    y_min, y_max = int(rec.movies.loc[rec.movies["year"] > 0, "year"].min()), int(rec.movies["year"].max())
    f_years = st.slider("Release year", y_min, y_max, (y_min, y_max))
    f_langs = st.multiselect("Language", rec.languages[:15], format_func=lambda c: LANG_NAMES.get(c, c))

    st.divider()
    st.subheader(f"Watchlist ({len(st.session_state.watchlist)})")
    for mid in list(st.session_state.watchlist):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.write(rec.get(mid)["title"])
        c2.button("✕", key=f"rm_{mid}", on_click=toggle_watch, args=(mid,), help="Remove")
    if st.session_state.watchlist:
        wl_df = rec.movies[rec.movies["movie_id"].isin(st.session_state.watchlist)]
        st.download_button("Download as CSV", wl_df[["title", "year", "vote_average", "director"]].to_csv(index=False),
                           "watchlist.csv", "text/csv", use_container_width=True)
        st.button("Clear watchlist", on_click=st.session_state.watchlist.clear, use_container_width=True)
    else:
        st.caption("Tap ＋ Watchlist on any movie to save it here.")

    if not KEY:
        st.info("No TMDB API key found, so posters are generated. Add one in `.env` (see the README).")

filters = dict(genres=f_genres, min_rating=f_rating, year_range=f_years, languages=f_langs)

# ----------------------------------------------------------------------------- hero
st.markdown(
    f"<div class='hero'><h1>What should you watch tonight?</h1>"
    f"<p>Pick a film you love, describe a mood, or blend a few favourites. "
    f"{len(rec):,} movies, matched by plot, cast, director and themes.</p><div class='bulbs'></div></div>",
    unsafe_allow_html=True,
)

tab_sim, tab_search, tab_profile, tab_browse, tab_explore = st.tabs(
    ["Similar movies", "Describe it", "Taste profile", "Browse", "Explore"]
)

with tab_sim:
    titles = rec.titles
    if "sim_title" not in st.session_state:
        st.session_state.sim_title = "Avatar" if "Avatar" in titles else titles[0]
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    choice = c1.selectbox("Start with a movie you like (type to search)", titles, key="sim_title")
    c2.button("🎲 Surprise me", on_click=lambda: st.session_state.update(sim_title=random.choice(titles)),
              use_container_width=True)
    mid = rec.id_for_title(choice)
    render_feature(rec.get(mid))
    st.markdown(f"<h2 class='section'>If you liked {html.escape(choice)}</h2>", unsafe_allow_html=True)
    render_cards(rec.recommend(mid, n_results, sort_by, **filters), "sim", source_id=mid)

with tab_search:
    st.markdown("<h2 class='section'>Describe what you feel like watching</h2>", unsafe_allow_html=True)
    st.text_input("Plot, mood, actor or director", key="q", placeholder="e.g. astronauts stranded in space")
    examples = ["heist with a clever twist", "haunted house", "Tom Hanks", "feel-good animated family"]
    for col, ex in zip(st.columns(len(examples)), examples):
        col.button(ex, key=f"ex_{ex}", on_click=lambda e=ex: st.session_state.update(q=e), use_container_width=True)
    q = st.session_state.get("q", "").strip()
    if q:
        results = rec.search(q, n_results, **filters)
        if results.empty:
            empty_state("No matches for that search", "Try fewer or more general words, or relax the sidebar filters.")
        else:
            render_cards(results, "search")
    else:
        empty_state("Nothing searched yet", "Type a few words above, or tap one of the examples.")

with tab_profile:
    st.markdown("<h2 class='section'>Blend your favourites</h2>", unsafe_allow_html=True)
    favs = st.multiselect("Pick two or more movies you love", rec.titles)
    from_wl = st.checkbox("Use my watchlist instead", disabled=not st.session_state.watchlist)
    ids = st.session_state.watchlist if from_wl else [rec.id_for_title(t) for t in favs]
    if ids:
        render_cards(rec.recommend_for_profile(ids, n_results, sort_by, **filters), "profile")
    else:
        empty_state("No favourites yet", "Choose at least one movie above to get blended recommendations.")

with tab_browse:
    mode = st.radio("Show me", ["Top rated", "Most popular", "Hidden gems", "Newest", "Random"], horizontal=True)
    if mode == "Random":
        st.button("🔀 Shuffle", key="shuffle")
    hint = {"Top rated": "Ranked by weighted rating, which accounts for the number of votes.",
            "Hidden gems": "Highly rated films that relatively few people have voted on."}.get(mode)
    if hint:
        st.caption(hint)
    render_cards(rec.browse(mode, n_results, **filters), "browse", ranked=mode in ("Top rated", "Most popular"))

with tab_explore:
    df = rec.movies
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Movies", f"{len(df):,}")
    m2.metric("Genres", len(rec.all_genres))
    m3.metric("Average rating", f"{df['vote_average'].mean():.2f}")
    m4.metric("Average runtime", f"{df.loc[df['runtime'] > 0, 'runtime'].mean():.0f} min")
    st.write("")
    a, b = st.columns(2)
    a.markdown("**Movies per genre**")
    a.bar_chart(df.explode("genres_list")["genres_list"].value_counts(), color=GOLD)
    decade = (df.loc[df["year"] > 0, "year"] // 10 * 10).astype(str) + "s"
    b.markdown("**Average rating per decade**")
    b.bar_chart(df.loc[df["year"] > 0].groupby(decade)["vote_average"].mean(), color=GOLD)
    c, d = st.columns(2)
    c.markdown("**Most prolific directors**")
    c.bar_chart(df.loc[df["director"] != "", "director"].value_counts().head(12), color=GOLD)
    d.markdown("**Most frequent lead actors**")
    d.bar_chart(df["cast_list"].apply(lambda x: x[0] if x else None).value_counts().head(12), color=GOLD)

st.divider()
st.caption("Data: TMDB 5000 Movie Dataset. Posters via the TMDB API. This product uses the TMDB API but is not endorsed or certified by TMDB.")
