"""
Media Recommender — core recommendation logic.
Supports movies, TV shows, and anime using content-based filtering
with optional collaborative signals from user ratings.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import ast
import os


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(
    movies_path: str = "data/movies.csv",
    shows_path: str = "data/shows.csv",
    anime_path: str = "data/anime.csv",
) -> pd.DataFrame:
    """
    Load and merge movie, show, and anime datasets into one unified DataFrame.

    Expected CSV columns (flexible — missing ones are filled with defaults):
        title, genres, overview/description, score/rating, year, type

    Returns a cleaned DataFrame with a consistent schema.
    """
    frames = []

    if os.path.exists(movies_path):
        movies = pd.read_csv(movies_path)
        movies = _normalise_movies(movies)
        frames.append(movies)

    if os.path.exists(shows_path):
        shows = pd.read_csv(shows_path)
        shows = _normalise_shows(shows)
        frames.append(shows)

    if os.path.exists(anime_path):
        anime = pd.read_csv(anime_path)
        anime = _normalise_anime(anime)
        frames.append(anime)

    if not frames:
        raise FileNotFoundError(
            "No dataset files found. See README for download instructions."
        )

    df = pd.concat(frames, ignore_index=True)
    df = _clean(df)
    return df


def _normalise_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Map TMDB-style movie CSV to unified schema."""
    rename = {
        "original_title": "title",
        "vote_average": "score",
        "release_date": "year",
        "genres": "genres_raw",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # TMDB stores genres as a JSON string list of dicts
    if "genres_raw" in df.columns:
        df["genres"] = df["genres_raw"].apply(_parse_tmdb_genres)
    elif "genres" not in df.columns:
        df["genres"] = ""

    if "year" in df.columns:
        df["year"] = pd.to_datetime(df["year"], errors="coerce").dt.year

    df["type"] = "Movie"
    return df


def _normalise_shows(df: pd.DataFrame) -> pd.DataFrame:
    """Map TMDB TV show CSV to unified schema."""
    rename = {
        "name": "title",
        "vote_average": "score",
        "first_air_date": "year",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "year" in df.columns:
        df["year"] = pd.to_datetime(df["year"], errors="coerce").dt.year

    if "genres" not in df.columns:
        df["genres"] = ""

    df["type"] = "TV Show"
    return df


def _normalise_anime(df: pd.DataFrame) -> pd.DataFrame:
    """Map MyAnimeList CSV to unified schema."""
    rename = {
        "name": "title",
        "rating": "score",
        "genre": "genres",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "year" not in df.columns:
        df["year"] = np.nan

    if "overview" not in df.columns:
        df["overview"] = ""

    df["type"] = "Anime"
    return df


def _parse_tmdb_genres(raw) -> str:
    """Turn '[{"id":28,"name":"Action"},...]' into 'Action Adventure'."""
    try:
        items = ast.literal_eval(str(raw))
        return " ".join(g["name"] for g in items if isinstance(g, dict))
    except Exception:
        return str(raw)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and drop rows with no title."""
    df = df.copy()

    # Keep a consistent set of columns
    for col in ["title", "genres", "overview", "score", "year", "type"]:
        if col not in df.columns:
            df[col] = np.nan if col in ("score", "year") else ""

    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["title", "type"]).reset_index(drop=True)

    df["overview"] = df["overview"].fillna("").astype(str)
    df["genres"] = df["genres"].fillna("").astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)

    # Normalise score to 0-1 for blending later
    scaler = MinMaxScaler()
    df["score_norm"] = scaler.fit_transform(df[["score"]])

    return df


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Create a TF-IDF matrix from each item's genres + overview.
    Genres are weighted higher by repeating them 3x.
    Returns the cosine similarity matrix.
    """
    # Weight genres more heavily
    df = df.copy()
    df["soup"] = (
        (df["genres"] + " ") * 3          # genres x3
        + df["overview"].str[:500]         # first 500 chars of overview
        + " " + df["type"]                 # include type as a weak signal
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=10_000)
    tfidf_matrix = tfidf.fit_transform(df["soup"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


# ──────────────────────────────────────────────
# Recommendation
# ──────────────────────────────────────────────

class MediaRecommender:
    """
    Content-based media recommender.

    Usage
    -----
    rec = MediaRecommender()
    rec.fit(df)                            # pass the unified DataFrame
    results = rec.recommend(
        watched=["Attack on Titan", "Arcane"],
        ratings={"Attack on Titan": 10, "Arcane": 9},
        n=10,
    )
    """

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.sim_matrix: np.ndarray | None = None
        self._index: pd.Series | None = None  # title → row index

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "MediaRecommender":
        """Build the similarity matrix from a cleaned DataFrame."""
        self.df = df.reset_index(drop=True)
        print("Building TF-IDF similarity matrix …")
        self.sim_matrix = build_feature_matrix(self.df)
        # Map lowercase title → integer index (first match wins)
        self._index = pd.Series(
            self.df.index, index=self.df["title"].str.lower()
        )
        print(f"Done. {len(self.df):,} titles indexed.")
        return self

    # ------------------------------------------------------------------
    def search(self, query: str, n: int = 10) -> pd.DataFrame:
        """Fuzzy-ish title search (case-insensitive substring match)."""
        mask = self.df["title"].str.contains(query, case=False, na=False)
        return self.df[mask][["title", "type", "genres", "score", "year"]].head(n)

    # ------------------------------------------------------------------
    def recommend(
        self,
        watched: list[str],
        ratings: dict[str, float] | None = None,
        n: int = 10,
        filter_type: str | None = None,   # "Movie", "TV Show", "Anime", or None
    ) -> pd.DataFrame:
        """
        Return top-N recommendations given a list of watched titles.

        Parameters
        ----------
        watched   : list of title strings the user has watched
        ratings   : optional dict of title → user rating (1-10).
                    Higher-rated titles influence results more.
        n         : number of recommendations to return
        filter_type : restrict results to one media type, or None for all
        """
        if self.df is None:
            raise RuntimeError("Call .fit(df) before .recommend()")

        ratings = ratings or {}
        score_map: dict[int, float] = {}  # row_index → weighted sim score

        found, not_found = [], []
        for title in watched:
            idx = self._lookup(title)
            if idx is None:
                not_found.append(title)
                continue
            found.append(title)

            # User rating weight — defaults to 1.0 if not provided
            weight = ratings.get(title, 7) / 10.0

            sim_scores = list(enumerate(self.sim_matrix[idx]))
            for row_idx, sim in sim_scores:
                if row_idx == idx:
                    continue
                score_map[row_idx] = score_map.get(row_idx, 0) + sim * weight

        if not score_map:
            print(f"None of the titles were found: {watched}")
            return pd.DataFrame()

        if not_found:
            print(f"Titles not found (check spelling): {not_found}")

        # Build results frame
        results = (
            pd.DataFrame(score_map.items(), columns=["idx", "sim_score"])
            .sort_values("sim_score", ascending=False)
        )

        # Remove already-watched titles
        watched_indices = {
            self._lookup(t) for t in found if self._lookup(t) is not None
        }
        results = results[~results["idx"].isin(watched_indices)]

        # Merge with metadata
        results = results.merge(
            self.df[["title", "type", "genres", "score", "year", "score_norm"]].reset_index().rename(columns={"index": "idx"}),
            on="idx",
        )

        # Blend similarity score with popularity score
        results["final_score"] = (
            results["sim_score"] * 0.75 + results["score_norm"] * 0.25
        )

        # Optional type filter
        if filter_type:
            results = results[results["type"] == filter_type]

        results = results.sort_values("final_score", ascending=False).head(n)
        return results[["title", "type", "genres", "score", "year", "final_score"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    def _lookup(self, title: str) -> int | None:
        """Return row index for a title (case-insensitive)."""
        key = title.lower().strip()
        if key in self._index.index:
            return int(self._index[key])
        return None
