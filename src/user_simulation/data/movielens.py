"""Download MovieLens and load it into a common interactions/items schema.

interactions: user_id, item_id, rating, timestamp (unix seconds), sorted by user then time.
items:        item_id, title, year, genres (list[str]).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASETS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}
DEFAULT_DATA_DIR = Path("data/raw")


def download(name: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Fetch and unzip a dataset once; later calls reuse the extracted folder."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    target = data_dir / name
    if target.exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASETS[name]) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(data_dir)
    return target


def load(name: str, data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = download(name, data_dir)
    if name == "ml-1m":
        ratings = pd.read_csv(
            root / "ratings.dat",
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        movies = pd.read_csv(
            root / "movies.dat",
            sep="::",
            engine="python",
            names=["item_id", "title", "genres"],
            encoding="latin-1",
        )
    else:
        ratings = pd.read_csv(root / "ratings.csv").rename(
            columns={"userId": "user_id", "movieId": "item_id"}
        )
        movies = pd.read_csv(root / "movies.csv").rename(columns={"movieId": "item_id"})

    interactions = ratings.astype(
        {"user_id": "int64", "item_id": "int64", "rating": "float64", "timestamp": "int64"}
    ).sort_values(["user_id", "timestamp", "item_id"], ignore_index=True)
    return interactions, _clean_items(movies)


def _clean_items(movies: pd.DataFrame) -> pd.DataFrame:
    """Split "Toy Story (1995)" into title + year and genres into a list."""
    parts = movies["title"].str.strip().str.extract(r"^(?P<title>.*?)\s*(?:\((?P<year>\d{4})\))?$")
    genres = movies["genres"].map(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    return pd.DataFrame(
        {
            "item_id": movies["item_id"].astype("int64"),
            "title": parts["title"],
            "year": pd.to_numeric(parts["year"]).astype("Int64"),
            "genres": genres,
        }
    )
