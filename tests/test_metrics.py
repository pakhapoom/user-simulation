import numpy as np
import pandas as pd
import pytest

from test_split import split_of, synthetic_log
from user_simulation.baselines import ItemKNN, Random
from user_simulation.data import Episode, SplitConfig
from user_simulation.eval import bootstrap_ci, evaluate
from user_simulation.eval.metrics import expected_calibration_error, slate_metrics


def test_perfect_slate():
    m = slate_metrics(np.array([0.1, 0.9, 0.2]), np.array([0, 1, 0]))
    assert m == {"hit@1": 1.0, "mrr": 1.0, "ndcg": 1.0, "gauc": 1.0}


def test_constant_scores_are_chance_level():
    m = slate_metrics(np.zeros(5), np.array([0, 0, 1, 0, 0]))
    assert m["hit@1"] == pytest.approx(0.2)
    assert m["gauc"] == pytest.approx(0.5)


def test_worst_slate():
    m = slate_metrics(np.array([3.0, 2.0, 1.0]), np.array([0, 0, 1]))
    assert m["hit@1"] == 0 and m["gauc"] == 0 and m["mrr"] == pytest.approx(1 / 3)


def test_calibration_error():
    labels = np.array([1, 0, 1, 0])
    assert expected_calibration_error(np.full(4, 0.5), labels) == pytest.approx(0.0)
    assert expected_calibration_error(np.full(4, 0.9), labels) == pytest.approx(0.4)


def test_item_knn_prefers_co_watched_items():
    # Items 1 and 2 are always watched together; item 3 never overlaps with them.
    train = pd.DataFrame(
        {"user_id": [0, 0, 1, 1, 2], "item_id": [1, 2, 1, 2, 3], "timestamp": [0] * 5}
    )
    knn = ItemKNN()
    knn.fit(train)
    episode = Episode(0, user_id=9, timestamp=1, candidates=(3, 2, 99), labels=(0, 1, 0))
    scores = knn.score(episode, pd.DataFrame({"item_id": [1]}))
    assert scores[1] > scores[0] == scores[2] == 0


def test_bootstrap_ci_covers_the_mean_and_shrinks_with_data():
    rng = np.random.default_rng(0)
    small = rng.random(100)
    large = rng.random(10_000)
    mean, lo, hi = bootstrap_ci(small, np.arange(100))
    assert lo < mean < hi and mean == pytest.approx(small.mean())
    _, lo_l, hi_l = bootstrap_ci(large, np.arange(10_000))
    assert hi_l - lo_l < (hi - lo) / 5


def test_bootstrap_ci_resamples_whole_users():
    # Two users with very different values: per-user resampling must widen the interval
    # compared with treating every slate as independent.
    values = np.r_[np.zeros(50), np.ones(50)]
    _, lo_u, hi_u = bootstrap_ci(values, np.r_[np.zeros(50), np.ones(50)])
    _, lo_e, hi_e = bootstrap_ci(values, np.arange(100))
    assert hi_u - lo_u > 3 * (hi_e - lo_e)


def test_random_vs_random_with_another_seed_is_no_difference():
    split = split_of(synthetic_log(n_users=200), SplitConfig(min_history=5))
    a = evaluate(Random(seed=0), split).per_slate
    b = evaluate(Random(seed=1), split).per_slate
    for metric in ["hit@1", "gauc"]:
        _, lo, hi = bootstrap_ci((a[metric] - b[metric]).to_numpy(), a["user_id"].to_numpy())
        assert lo < 0 < hi
