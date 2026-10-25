"""Download MovieLens and load it into a common interactions/items schema.

interactions: user_id, item_id, rating, timestamp (unix seconds), sorted by user then time.
items:        item_id, title, year, genres (list[str]).
"""

from __future__ import annotations

import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASETS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}
DEFAULT_DATA_DIR = Path("data/raw")

# files.grouplens.org chains to this root, which Mozilla-derived stores (certifi,
# Debian/Ubuntu ca-certificates, so Colab) lack. It is trusted on top of the
# system store for these downloads only.
# SHA-256 86:A1:EC:BA:08:9C:4A:8D:3B:BE:27:34:C6:12:BA:34:1D:81:3E:04:3C:F9:E8:A8:62:CD:5C:57:A3:6B:BE:6B
EMSIGN_ECC_ROOT_CA_G3 = """\
-----BEGIN CERTIFICATE-----
MIICTjCCAdOgAwIBAgIKPPYHqWhwDtqLhDAKBggqhkjOPQQDAzBrMQswCQYDVQQG
EwJJTjETMBEGA1UECxMKZW1TaWduIFBLSTElMCMGA1UEChMcZU11ZGhyYSBUZWNo
bm9sb2dpZXMgTGltaXRlZDEgMB4GA1UEAxMXZW1TaWduIEVDQyBSb290IENBIC0g
RzMwHhcNMTgwMjE4MTgzMDAwWhcNNDMwMjE4MTgzMDAwWjBrMQswCQYDVQQGEwJJ
TjETMBEGA1UECxMKZW1TaWduIFBLSTElMCMGA1UEChMcZU11ZGhyYSBUZWNobm9s
b2dpZXMgTGltaXRlZDEgMB4GA1UEAxMXZW1TaWduIEVDQyBSb290IENBIC0gRzMw
djAQBgcqhkjOPQIBBgUrgQQAIgNiAAQjpQy4LRL1KPOxst3iAhKAnjlfSU2fySU0
WXTsuwYc58Byr+iuL+FBVIcUqEqy6HyC5ltqtdyzdc6LBtCGI79G1Y4PPwT01xyS
fvalY8L1X44uT6EYGQIrMgqCZH0Wk9GjQjBAMB0GA1UdDgQWBBR8XQKEE9TMipuB
zhccLikenEhjQjAOBgNVHQ8BAf8EBAMCAQYwDwYDVR0TAQH/BAUwAwEB/zAKBggq
hkjOPQQDAwNpADBmAjEAvvNhzwIQHWSVB7gYboiFBS+DCBeQyh+KTOgNG3qxrdWB
CUfvO6wIBHxcmbHtRwfSAjEAnbpV/KlK6O3t5nYBQnvI+GDZjVGLVTv7jHvrZQnD
+JbNR6iC8hZVdyR+EhCVBCyj
-----END CERTIFICATE-----
"""


def download(name: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Fetch and unzip a dataset once; later calls reuse the extracted folder."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    target = data_dir / name
    if target.exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    context.load_verify_locations(cadata=EMSIGN_ECC_ROOT_CA_G3)
    with urllib.request.urlopen(DATASETS[name], context=context) as response:
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
