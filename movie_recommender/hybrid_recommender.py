from __future__ import annotations

from difflib import get_close_matches
from functools import lru_cache
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


TMDB_PATH = "./TMDB_movie_dataset_v11.csv"
ML_MOVIES_PATH = "./ml-32m/movies.csv"
ML_LINKS_PATH = "./ml-32m/links.csv"
ML_RATINGS_PATH = "./ml-32m/ratings.csv"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w342"


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _extract_year(title: str) -> str:
    if not isinstance(title, str) or len(title) < 6:
        return ""
    if title.endswith(")") and title[-6] == "(" and title[-5:-1].isdigit():
        return title[-5:-1]
    return ""


def _strip_year_suffix(title: str) -> str:
    if not isinstance(title, str):
        return ""
    text = title.strip()
    if len(text) >= 6 and text.endswith(")") and text[-6] == "(" and text[-5:-1].isdigit():
        return text[:-7].strip()
    return text


def _normalize_title_for_lookup(title: str) -> str:
    if not isinstance(title, str):
        return ""
    text = title.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\d{4}\)$", "", text).strip()
    if "," in text:
        left, right = text.rsplit(",", 1)
        article = right.strip()
        if article in {"the", "a", "an"}:
            text = f"{article} {left.strip()}"
    return text


def _title_tokens(title: str) -> set[str]:
    text = _normalize_title_for_lookup(title)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    raw_tokens = [t for t in text.split() if len(t) > 1]
    stop = {
        "the",
        "a",
        "an",
        "of",
        "and",
        "in",
        "on",
        "to",
        "for",
        "part",
        "episode",
        "chapter",
        "movie",
        "film",
    }
    return {t for t in raw_tokens if t not in stop}


def _series_anchor_tokens(title: str) -> list[str]:
    text = _normalize_title_for_lookup(title)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1]
    skip = {"episode", "chapter", "part", "vol", "volume", "the", "a", "an"}
    cleaned = [t for t in tokens if t not in skip and not t.isdigit()]
    return cleaned[:2]


def _looks_like_same_series(selected_title: str, candidate_title: str) -> bool:
    selected_tokens = _title_tokens(selected_title)
    candidate_tokens = _title_tokens(candidate_title)
    if len(selected_tokens) < 2 or len(candidate_tokens) < 2:
        return False
    anchor = _series_anchor_tokens(selected_title)
    if len(anchor) == 2 and all(tok in candidate_tokens for tok in anchor):
        return True
    common = selected_tokens & candidate_tokens
    if len(common) < 2:
        return False
    overlap_selected = len(common) / len(selected_tokens)
    overlap_candidate = len(common) / len(candidate_tokens)
    return overlap_selected >= 0.45 or overlap_candidate >= 0.45


def _prepare_datasets() -> pd.DataFrame:
    tmdb = pd.read_csv(TMDB_PATH)
    links = pd.read_csv(ML_LINKS_PATH)
    ratings = pd.read_csv(ML_RATINGS_PATH, usecols=["movieId", "rating"])
    ml_movies = pd.read_csv(ML_MOVIES_PATH, usecols=["movieId", "title"])

    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype(int)

    ratings_agg = (
        ratings.groupby("movieId")["rating"]
        .agg(ml_rating_mean="mean", ml_rating_count="count")
        .reset_index()
    )

    movie_base = links.merge(ratings_agg, on="movieId", how="left").merge(
        ml_movies, on="movieId", how="left"
    )
    movie_base["ml_year"] = movie_base["title"].apply(_extract_year)

    tmdb = tmdb.rename(columns={"id": "tmdbId"}).copy()
    tmdb["tmdbId"] = pd.to_numeric(tmdb["tmdbId"], errors="coerce")
    tmdb = tmdb.dropna(subset=["tmdbId"]).copy()
    tmdb["tmdbId"] = tmdb["tmdbId"].astype(int)

    merged = movie_base.merge(tmdb, on="tmdbId", how="inner")
    merged = merged.drop_duplicates(subset=["tmdbId"]).copy()

    for col in ["overview", "genres", "keywords", "tagline", "original_title"]:
        if col not in merged:
            merged[col] = ""
        merged[col] = merged[col].fillna("")

    merged["title_for_search"] = merged["title_x"].fillna(merged["title_y"]).fillna("")
    merged["display_title"] = (
        merged["title_y"].fillna("").astype(str).str.strip()
    )
    empty_display = merged["display_title"].eq("")
    merged.loc[empty_display, "display_title"] = (
        merged.loc[empty_display, "title_for_search"].map(_strip_year_suffix)
    )
    merged["ml_rating_mean"] = merged["ml_rating_mean"].fillna(0.0)
    merged["ml_rating_count"] = merged["ml_rating_count"].fillna(0).astype(int)
    merged["vote_average"] = pd.to_numeric(merged["vote_average"], errors="coerce").fillna(0.0)
    merged["popularity"] = pd.to_numeric(merged["popularity"], errors="coerce").fillna(0.0)
    merged["release_year"] = pd.to_datetime(
        merged["release_date"], errors="coerce"
    ).dt.year
    merged["content_text"] = (
        merged["overview"].map(_clean_text)
        + " "
        + merged["genres"].map(_clean_text)
        + " "
        + merged["keywords"].map(_clean_text)
        + " "
        + merged["tagline"].map(_clean_text)
        + " "
        + merged["original_title"].map(_clean_text)
    )

    merged = merged[merged["title_for_search"].str.len() > 0].reset_index(drop=True)
    merged["genre_tokens"] = merged["genres"].fillna("").map(
        lambda g: {x.strip().lower() for x in str(g).split(",") if x.strip()}
    )
    return merged


@lru_cache(maxsize=1)
def _build_model():
    data = _prepare_datasets()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=30000)
    tfidf_matrix = vectorizer.fit_transform(data["content_text"])
    titles = data["title_for_search"].tolist()
    normalized_map = {t.lower(): i for i, t in enumerate(titles)}
    for i, t in enumerate(titles):
        normalized_map.setdefault(_strip_year_suffix(t).lower(), i)
        normalized_map.setdefault(_normalize_title_for_lookup(t), i)
        normalized_map.setdefault(_normalize_title_for_lookup(_strip_year_suffix(t)), i)
        if "title_y" in data.columns and isinstance(data.iloc[i]["title_y"], str):
            tmdb_title = data.iloc[i]["title_y"]
            normalized_map.setdefault(tmdb_title.lower(), i)
            normalized_map.setdefault(_strip_year_suffix(tmdb_title).lower(), i)
            normalized_map.setdefault(_normalize_title_for_lookup(tmdb_title), i)
        if "original_title" in data.columns and isinstance(data.iloc[i]["original_title"], str):
            original_title = data.iloc[i]["original_title"]
            normalized_map.setdefault(original_title.lower(), i)
            normalized_map.setdefault(_normalize_title_for_lookup(original_title), i)
    rating_mean = data["ml_rating_mean"].to_numpy(dtype=np.float32)
    rating_count = data["ml_rating_count"].to_numpy(dtype=np.float32)
    c = max(float(data["ml_rating_count"].quantile(0.75)), 1.0)
    return data, tfidf_matrix, titles, normalized_map, rating_mean, rating_count, c


def _resolve_movie_index(
    query: str, titles: list[str], normalized_map: dict[str, int]
) -> int | None:
    query_norm = query.strip().lower()
    if query_norm in normalized_map:
        return normalized_map[query_norm]
    query_norm2 = _normalize_title_for_lookup(query)
    if query_norm2 in normalized_map:
        return normalized_map[query_norm2]

    match = get_close_matches(query_norm2 or query_norm, normalized_map.keys(), n=1, cutoff=0.6)
    if not match:
        return None
    return normalized_map[match[0]]


def recommend_movies(
    movie_query: str,
    top_k: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
    min_vote_average: float | None = None,
    sort_by: str = "hybrid",
    exclude_same_series: bool = True,
    genres: list[str] | None = None,
):
    (
        data,
        tfidf_matrix,
        titles,
        normalized_map,
        rating_mean,
        rating_count,
        c,
    ) = _build_model()

    idx = _resolve_movie_index(movie_query, titles, normalized_map)
    if idx is None:
        return None, []

    selected_display_title = data.iloc[idx]["display_title"]
    selected_year = data.iloc[idx]["release_year"]
    if pd.notna(selected_year):
        selected_title = f"{selected_display_title} ({int(selected_year)})"
    else:
        selected_title = selected_display_title
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).ravel().astype(np.float32)
    sim_scores[idx] = -1.0

    vote_weight = np.divide(
        rating_count, rating_count + c, out=np.zeros_like(rating_count), where=rating_count > 0
    )
    quality = vote_weight * rating_mean + (1.0 - vote_weight) * 3.5
    final_scores = sim_scores * 0.7 + (quality / 5.0) * 0.3

    release_years = data["release_year"].to_numpy(dtype=np.float32)
    mask = np.ones(len(final_scores), dtype=bool)
    mask[idx] = False
    if year_from is not None:
        mask &= release_years >= float(year_from)
    if year_to is not None:
        mask &= release_years <= float(year_to)
    mask &= ~np.isnan(release_years)
    if min_vote_average is not None:
        mask &= data["vote_average"].to_numpy(dtype=np.float32) >= float(min_vote_average)
    if genres is not None:
        selected = {g.strip().lower() for g in genres if g and g.strip()}
        if not selected:
            return selected_title, []
        genre_mask = data["genre_tokens"].map(lambda s: bool(s & selected)).to_numpy(dtype=bool)
        mask &= genre_mask

    candidate_indices = np.where(mask)[0]
    if len(candidate_indices) == 0:
        return selected_title, []

    if exclude_same_series:
        selected_base = str(data.iloc[idx]["title_for_search"])
        filtered_indices = []
        for i in candidate_indices:
            candidate_base = str(data.iloc[int(i)]["title_for_search"])
            if not _looks_like_same_series(selected_base, candidate_base):
                filtered_indices.append(int(i))
        candidate_indices = np.array(filtered_indices, dtype=np.int32)
        if len(candidate_indices) == 0:
            return selected_title, []

    if sort_by == "popular":
        candidate_scores = data["popularity"].to_numpy(dtype=np.float32)[candidate_indices]
    else:
        candidate_scores = final_scores[candidate_indices]
    candidate_count = int(min(max(top_k * 30, 300), len(candidate_indices)))
    top_local = np.argpartition(candidate_scores, -candidate_count)[-candidate_count:]
    top_local = top_local[np.argsort(candidate_scores[top_local])[::-1]][:top_k]
    top_idx = candidate_indices[top_local]

    recommendations = []
    for i in top_idx:
        recommendations.append(
            {
                "title": data.iloc[int(i)]["title_for_search"],
                "display_title": data.iloc[int(i)]["display_title"],
                "release_year": int(data.iloc[int(i)]["release_year"]),
                "poster_url": (
                    f"{POSTER_BASE_URL}{data.iloc[int(i)]['poster_path']}"
                    if isinstance(data.iloc[int(i)]["poster_path"], str)
                    and data.iloc[int(i)]["poster_path"]
                    else ""
                ),
                "score": round(float(final_scores[i]), 4),
                "similarity": round(float(sim_scores[i]), 4),
                "ml_rating_mean": round(float(rating_mean[i]), 2),
                "ml_rating_count": int(rating_count[i]),
            }
        )

    return selected_title, recommendations


def suggest_movies_data(query: str, limit: int = 8) -> list[dict]:
    if not query or len(query.strip()) < 2:
        return []
    data, _, _, _, _, _, _ = _build_model()
    q = _normalize_title_for_lookup(query)
    if not q:
        return []

    normalized = data["display_title"].fillna("").map(_normalize_title_for_lookup)
    mask = normalized.str.contains(re.escape(q), na=False)
    candidates = data.loc[mask, ["display_title", "release_year", "popularity", "poster_path"]].copy()

    if candidates.empty:
        return []

    candidates["norm"] = candidates["display_title"].map(_normalize_title_for_lookup)
    candidates["match_rank"] = 2
    candidates.loc[candidates["norm"] == q, "match_rank"] = 0
    candidates.loc[
        (candidates["match_rank"] > 0) & candidates["norm"].str.startswith(q), "match_rank"
    ] = 1
    candidates["group_count"] = candidates.groupby("norm")["norm"].transform("count")
    candidates = candidates.drop_duplicates(subset=["display_title", "release_year", "poster_path"])
    candidates = candidates.sort_values(
        by=["match_rank", "group_count", "popularity", "release_year"],
        ascending=[True, False, False, False],
        na_position="last",
    ).head(limit)

    results: list[dict] = []
    for _, row in candidates.iterrows():
        title = str(row["display_title"]).strip()
        year = row["release_year"]
        year_text = str(int(year)) if pd.notna(year) else ""
        poster_path = row["poster_path"] if isinstance(row["poster_path"], str) else ""
        if pd.notna(year):
            label = f"{title} ({int(year)})"
        else:
            label = title
        results.append(
            {
                "label": label,
                "title": title,
                "year": year_text,
                "group_count": int(row["group_count"]),
                "poster_url": f"{POSTER_BASE_URL}{poster_path}" if poster_path else "",
            }
        )
    return results


def get_available_genres() -> list[str]:
    data, _, _, _, _, _, _ = _build_model()
    genres: set[str] = set()
    for tokens in data["genre_tokens"]:
        genres.update(tokens)
    return sorted(g.title() for g in genres)


def warmup_model() -> None:
    _build_model()
