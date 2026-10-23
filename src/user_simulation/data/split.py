"""Turn an interaction log into leakage-free decision episodes.

A single global time cutoff separates the past (train) from the future (test).
Global statistics (popularity, item similarity) may only use train. Each test
interaction becomes an episode: at that moment the user is shown a slate of
items, one of which they actually picked (the positive) plus sampled negatives.
A user's own history "up until now" (anything strictly before the episode) is
always visible, because a live system would know it too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    test_fraction: float = 0.2  # share of the timeline (by interaction count) after the cutoff
    min_history: int = 20  # users need this many pre-cutoff interactions to be evaluated
    episodes_per_user: int = 5  # first N post-cutoff interactions per user become episodes
    slate_size: int = 5  # 1 positive + (slate_size - 1) negatives
    # "matched": films released within `year_tolerance` years of the positive, weighted by
    # how often each was rated in the last `recent_window` ratings before the decision.
    # Returning users mostly rate new releases, so without year matching "the newest film
    # on the slate" is the answer about half the time. "popularity" (pre-cutoff counts)
    # and "uniform" are kept for comparison.
    negative_sampling: Literal["matched", "popularity", "uniform"] = "matched"
    recent_window: int = 100_000
    year_tolerance: int = 0
    seed: int = 0


@dataclass(frozen=True)
class Episode:
    episode_id: int
    user_id: int
    timestamp: int
    candidates: tuple[int, ...]  # shuffled, so position carries no signal
    labels: tuple[int, ...]  # 1 for the item the user actually picked

    @property
    def positive(self) -> int:
        return self.candidates[self.labels.index(1)]


@dataclass
class Split:
    config: SplitConfig
    cutoff: int
    train: pd.DataFrame
    episodes: list[Episode]
    _by_user: dict[int, pd.DataFrame] = field(repr=False)

    def history(self, user_id: int, before: int) -> pd.DataFrame:
        """Everything the user did strictly before `before`, oldest first."""
        frame = self._by_user[user_id]
        return frame[frame["timestamp"] < before]


def _cutoff(interactions: pd.DataFrame, config: SplitConfig) -> int:
    return int(interactions["timestamp"].quantile(1 - config.test_fraction))


def make_validation_split(
    interactions: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
    days: int = 14,
    item_years: pd.Series | None = None,
) -> Split:
    """The same split applied to the log before the test cutoff, with the last `days`
    before it as the future: an earlier past/future pair for tuning, which never sees
    anything at or after the test cutoff."""
    test_cutoff = _cutoff(interactions, config)
    past = interactions[interactions["timestamp"] < test_cutoff]
    return make_split(past, config, cutoff=test_cutoff - days * 86_400, item_years=item_years)


def make_split(
    interactions: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
    cutoff: int | None = None,
    item_years: pd.Series | None = None,
) -> Split:
    """`cutoff` defaults to the `test_fraction` quantile of the timeline. `item_years`
    (release year by item_id) is required for "matched" negatives."""
    if config.negative_sampling == "matched" and item_years is None:
        raise ValueError('"matched" negatives need item_years')
    rng = np.random.default_rng(config.seed)
    if cutoff is None:
        cutoff = _cutoff(interactions, config)
    train = interactions[interactions["timestamp"] < cutoff]

    train_counts = train.groupby("user_id").size()
    eligible = train_counts.index[train_counts >= config.min_history]
    test = interactions[
        (interactions["timestamp"] >= cutoff) & interactions["user_id"].isin(eligible)
    ]
    test = test.sort_values(["user_id", "timestamp", "item_id"])
    test = test.groupby("user_id").head(config.episodes_per_user)

    # An item can only be a negative once it exists, i.e. after its first rating.
    first_seen = interactions.groupby("item_id")["timestamp"].min()
    item_ids = first_seen.index.to_numpy()
    first_seen_ts = first_seen.to_numpy()
    if config.negative_sampling == "popularity":
        # +1 so items unseen in train can still be drawn.
        weights = train["item_id"].value_counts().reindex(item_ids, fill_value=0).to_numpy() + 1.0
    else:
        weights = np.ones(len(item_ids))
    matched = config.negative_sampling == "matched"
    if matched:
        years = item_years.reindex(item_ids).to_numpy(float)
        by_time = interactions.sort_values("timestamp", kind="stable")
        times = by_time["timestamp"].to_numpy()
        item_codes = np.searchsorted(item_ids, by_time["item_id"].to_numpy())

    # Exclude anything the user ever rated: unrated is our only proxy for "not chosen".
    user_items = interactions.groupby("user_id")["item_id"].agg(lambda s: s.to_numpy())

    episodes = []
    n_negatives = config.slate_size - 1
    for row in test.itertuples(index=False):
        allowed = (first_seen_ts <= row.timestamp) & ~np.isin(item_ids, user_items[row.user_id])
        if matched:
            end = np.searchsorted(times, row.timestamp)  # strictly before the decision
            window = item_codes[max(0, end - config.recent_window) : end]
            weights = np.bincount(window, minlength=len(item_ids)) + 1.0
            year = years[np.searchsorted(item_ids, row.item_id)]
            tolerance = config.year_tolerance
            # Widen the band for sparse years; a positive without a year is not matched.
            while not np.isnan(year) and tolerance < 200:
                band = allowed & (np.abs(years - year) <= tolerance)
                if band.sum() >= n_negatives:
                    allowed = band
                    break
                tolerance = max(1, 2 * tolerance)
        pool, pool_weights = item_ids[allowed], weights[allowed]
        if len(pool) < n_negatives:
            continue
        negatives = rng.choice(
            pool, size=n_negatives, replace=False, p=pool_weights / pool_weights.sum()
        )
        candidates = np.concatenate([[row.item_id], negatives])
        labels = np.zeros(config.slate_size, dtype=int)
        labels[0] = 1
        order = rng.permutation(config.slate_size)
        episodes.append(
            Episode(
                episode_id=len(episodes),
                user_id=int(row.user_id),
                timestamp=int(row.timestamp),
                candidates=tuple(int(c) for c in candidates[order]),
                labels=tuple(int(label) for label in labels[order]),
            )
        )

    by_user = {
        int(user): frame.reset_index(drop=True)
        for user, frame in interactions[interactions["user_id"].isin(eligible)].groupby("user_id")
    }
    return Split(config=config, cutoff=cutoff, train=train, episodes=episodes, _by_user=by_user)
