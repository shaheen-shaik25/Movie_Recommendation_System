"""TMDB API helper: posters and trailers, fetched in parallel and cached in memory.

The API key is read from the TMDB_API_KEY environment variable (or a .env file).
Without a key the app still works - it just shows generated placeholder posters.
"""
from __future__ import annotations

import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

API_URL = "https://api.themoviedb.org/3/movie/{id}"
IMG_URL = "https://image.tmdb.org/t/p/w342"

_cache: dict[int, dict] = {}
_lock = threading.Lock()


def get_api_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.getenv("TMDB_API_KEY", "").strip()


def fetch_info(movie_id: int, api_key: str, timeout: float = 6.0) -> dict:
    """Return {'poster': url|None, 'trailer': url|None, 'homepage': url|None}. Never raises."""
    empty = {"poster": None, "trailer": None, "homepage": None}
    if not api_key:
        return empty
    with _lock:
        if movie_id in _cache:
            return _cache[movie_id]
    try:
        r = requests.get(
            API_URL.format(id=movie_id),
            params={"api_key": api_key, "language": "en-US", "append_to_response": "videos"},
            timeout=timeout,
        )
        r.raise_for_status()
        d = r.json()
    except (requests.RequestException, ValueError):
        return empty  # not cached, so a later rerun retries
    trailer = next(
        (f"https://www.youtube.com/watch?v={v['key']}" for v in d.get("videos", {}).get("results", [])
         if v.get("site") == "YouTube" and v.get("type") == "Trailer"),
        None,
    )
    info = {
        "poster": IMG_URL + d["poster_path"] if d.get("poster_path") else None,
        "trailer": trailer,
        "homepage": d.get("homepage") or None,
    }
    with _lock:
        _cache[movie_id] = info
    return info


def fetch_many(movie_ids, api_key: str, workers: int = 8) -> dict[int, dict]:
    ids = list(dict.fromkeys(int(i) for i in movie_ids))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(ids, pool.map(lambda i: fetch_info(i, api_key), ids)))


def placeholder_poster(title: str) -> bytes:
    """Generate a themed PNG poster so cards look right offline / without an API key."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = 342, 513
    img = Image.new("RGB", (w, h))
    px = ImageDraw.Draw(img)
    for y in range(h):  # plum gradient
        t = y / h
        px.line([(0, y), (w, y)], fill=(int(58 - 30 * t), int(28 - 12 * t), int(50 - 22 * t)))
    gold = (232, 181, 99)
    px.rectangle([16, 16, w - 16, h - 16], outline=gold, width=2)
    px.rectangle([24, 24, w - 24, h - 24], outline=(120, 90, 70), width=1)

    font = None
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", "DejaVuSerif-Bold.ttf",
                 "/Library/Fonts/Georgia.ttf", "C:/Windows/Fonts/georgiab.ttf"):
        try:
            font = ImageFont.truetype(path, 26)
            break
        except OSError:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=26)
        except TypeError:
            font = ImageFont.load_default()

    lines, line = [], ""
    for word in title.split():
        if len(line) + len(word) > 13 and line:
            lines.append(line.strip())
            line = ""
        line += word + " "
    lines.append(line.strip())
    lines = lines[:5]
    y = h // 2 - 20 * len(lines)
    for ln in lines:
        box = px.textbbox((0, 0), ln, font=font)
        px.text(((w - (box[2] - box[0])) / 2, y), ln, fill=(243, 233, 220), font=font)
        y += 40
    px.line([(w / 2 - 28, y + 6), (w / 2 + 28, y + 6)], fill=gold, width=2)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
