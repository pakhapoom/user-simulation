import numpy as np
import pandas as pd

from functools import partial

from user_simulation.data import SplitConfig, make_split, make_validation_split

# Release years cycling 1990-1994, so "matched" negatives have a real band to draw from.
YEARS = pd.Series(1990 + np.arange(200) % 5, index=np.arange(200))
split_of = partial(make_split, item_years=YEARS)
validation_of = partial(make_validation_split, item_years=YEARS)


def synthetic_log(n_users=30, n_items=60, per_user=40, seed=0, span=1_000_000):
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        items = rng.choice(n_items, size=per_user, replace=False)
        times = np.sort(rng.integers(0, span, size=per_user))
        rows += [(user, int(i), 4.0, int(t)) for i, t in zip(items, times)]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_split_has_no_leakage():
    log = synthetic_log()
    split = split_of(log, SplitConfig(min_history=5, slate_size=5))
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
    a, b = split_of(log, config), split_of(log, config)
    assert a.episodes == b.episodes
    per_user = pd.Series([e.user_id for e in a.episodes]).value_counts()
    assert per_user.max() <= 3
    train_counts = a.train.groupby("user_id").size()
    assert all(train_counts[u] >= 5 for u in per_user.index)


def test_validation_split_is_the_two_weeks_before_test_cutoff():
    log = synthetic_log(span=100 * 86_400)
    config = SplitConfig(min_history=5)
    test, validation = split_of(log, config), validation_of(log, config)
    assert validation.episodes
    assert validation.cutoff == test.cutoff - 14 * 86_400
    assert all(validation.cutoff <= ep.timestamp < test.cutoff for ep in validation.episodes)
    # Nothing at or after the test cutoff is visible: not in train, not in any history.
    assert (validation.train["timestamp"] < validation.cutoff).all()
    for ep in validation.episodes:
        assert (validation.history(ep.user_id, ep.timestamp + 1)["timestamp"] < test.cutoff).all()


def test_matched_negatives_share_the_positive_release_year():
    # 40 films per release year, so the band never needs widening.
    split = split_of(synthetic_log(n_items=200), SplitConfig(min_history=5))
    for ep in split.episodes:
        negatives = [c for c, y in zip(ep.candidates, ep.labels) if y == 0]
        assert all(YEARS[n] == YEARS[ep.positive] for n in negatives)


def test_matched_negatives_follow_what_was_popular_just_before_the_decision():
    # Item 99 is new: never rated before the cutoff, then rated by many users right after.
    log = synthetic_log(n_users=60, per_user=40)
    cutoff = split_of(log, SplitConfig(min_history=5)).cutoff
    burst = pd.DataFrame({"user_id": range(100, 160), "item_id": 99, "rating": 4.0, "timestamp": cutoff + 1})
    log = pd.concat([log, burst], ignore_index=True)

    def share_with_new_item(sampling):
        split = split_of(log, SplitConfig(min_history=5, negative_sampling=sampling, recent_window=200))
        late = [e for e in split.episodes if e.timestamp > cutoff + 1 and YEARS[e.positive] == YEARS[99]]
        return np.mean([99 in e.candidates for e in late])

    assert share_with_new_item("matched") > 3 * share_with_new_item("popularity")
