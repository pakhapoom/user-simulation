import numpy as np
import pandas as pd
import pytest

from user_simulation.baselines import ItemKNN
from user_simulation.data import Episode
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
