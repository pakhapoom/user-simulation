import numpy as np
import pandas as pd

from user_simulation.data import SplitConfig, make_split


def synthetic_log(n_users=30, n_items=60, per_user=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        items = rng.choice(n_items, size=per_user, replace=False)
        times = np.sort(rng.integers(0, 1_000_000, size=per_user))
        rows += [(user, int(i), 4.0, int(t)) for i, t in zip(items, times)]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_split_has_no_leakage():
    log = synthetic_log()
    split = make_split(log, SplitConfig(min_history=5, slate_size=5))
    assert split.episodes
    assert (split.train["timestamp"] < split.cutoff).all()

    first_seen = log.groupby("item_id")["timestamp"].min()
    rated = log.groupby("user_id")["item_id"].agg(set)
    for ep in split.episodes:
        assert ep.timestamp >= split.cutoff
        assert sum(ep.labels) == 1 and len(set(ep.candidates)) == len(ep.candidates)
        # The positive is a real future interaction; negatives were never rated by the user.
        assert ((log["user_id"] == ep.user_id) & (log["item_id"] == ep.positive)
                & (log["timestamp"] == ep.timestamp)).any()
        negatives = [c for c, y in zip(ep.candidates, ep.labels) if y == 0]
        assert not set(negatives) & rated[ep.user_id]
        # Negatives must already exist at decision time.
        assert all(first_seen[n] <= ep.timestamp for n in negatives)
        # History is strictly the past and never contains the answer.
        history = split.history(ep.user_id, ep.timestamp)
        assert (history["timestamp"] < ep.timestamp).all()
        assert ep.positive not in set(history["item_id"])


def test_split_is_deterministic_and_respects_limits():
    log = synthetic_log()
    config = SplitConfig(min_history=5, episodes_per_user=3, seed=7)
    a, b = make_split(log, config), make_split(log, config)
    assert a.episodes == b.episodes
    per_user = pd.Series([e.user_id for e in a.episodes]).value_counts()
    assert per_user.max() <= 3
    train_counts = a.train.groupby("user_id").size()
    assert all(train_counts[u] >= 5 for u in per_user.index)
