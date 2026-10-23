"""Non-LLM reference points the agent has to beat."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from user_simulation.data.split import Episode


class Random:
    name = "random"
    probabilistic = False

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def score(self, episode: Episode, history: pd.DataFrame) -> np.ndarray:
        return self._rng.random(len(episode.candidates))


class Popularity:
    """How many people interacted with the item before the cutoff."""

    name = "popularity"
    probabilistic = False

    def fit(self, train: pd.DataFrame) -> None:
        self._counts = train["item_id"].value_counts()

    def score(self, episode: Episode, history: pd.DataFrame) -> np.ndarray:
        return self._counts.reindex(list(episode.candidates), fill_value=0).to_numpy(float)


class ItemKNN:
    """Sum of cosine similarities between a candidate and the user's recent items.

    Item vectors are binary "who interacted" columns from train, so similarity
    means "watched by the same people".
    """

    name = "item_knn"
    probabilistic = False

    def __init__(self, max_history: int = 10):  # picked on validation from {10, 20, 50, 100}
        self.max_history = max_history

    def fit(self, train: pd.DataFrame) -> None:
        items = train["item_id"].astype("category")
        users = train["user_id"].astype("category")
        matrix = sparse.csr_matrix(
            (np.ones(len(train)), (items.cat.codes, users.cat.codes)),
            shape=(len(items.cat.categories), len(users.cat.categories)),
        )
        matrix.data[:] = 1.0  # collapse duplicate (item, user) pairs
        norms = np.sqrt(np.asarray(matrix.sum(axis=1)).ravel())
        self._vectors = sparse.diags(1.0 / norms) @ matrix
        self._index = {int(item): i for i, item in enumerate(items.cat.categories)}

    def score(self, episode: Episode, history: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(episode.candidates))
        recent = history["item_id"].tail(self.max_history)
        hist_rows = [self._index[i] for i in recent if i in self._index]
        known = [(pos, self._index[c]) for pos, c in enumerate(episode.candidates) if c in self._index]
        if not hist_rows or not known:
            return scores
        positions, cand_rows = zip(*known)
        sims = self._vectors[list(cand_rows)] @ self._vectors[hist_rows].T
        scores[list(positions)] = np.asarray(sims.sum(axis=1)).ravel()
        return scores
