import json

import numpy as np
import pandas as pd

from test_split import synthetic_log
from user_simulation import agent
from user_simulation.agent import OfflineAgent, export, sample_episodes
from user_simulation.data import SplitConfig, make_split
from user_simulation.eval import evaluate


def setup(tmp_path):
    split = make_split(synthetic_log(), SplitConfig(min_history=5))
    items = pd.DataFrame(
        {"item_id": range(60), "title": [f"Movie {i}" for i in range(60)], "year": 1999, "genres": [["Drama"]] * 60}
    )
    episodes = sample_episodes(split, 6)
    paths = export(episodes, split, items, tmp_path, n_recent=5, batch_size=4)
    return split, items, episodes, paths


def keys(path):
    return [line[4:] for line in path.read_text().splitlines() if line.startswith("### ")]


def test_round_trip_scores_answers_and_counts_bad_ones(tmp_path):
    split, items, episodes, paths = setup(tmp_path)
    assert len(paths) == 2 and len(keys(paths[0])) == 4
    answers = {k: {"A": 2, "B": 1, "C": 1, "D": 0, "E": 0} for k in keys(paths[0])}
    answers[keys(paths[0])[0]] = {"A": "high"}  # malformed
    (tmp_path / "responses-01.json").write_text(json.dumps(answers))  # batch 2 left unanswered

    llm = OfflineAgent(items, tmp_path, n_recent=5)
    result = evaluate(llm, split, episodes)
    assert llm.unparsed == 1 + 2
    assert len(result.per_slate) == 6 and "log_loss" in result.summary
    history = split.history(episodes[1].user_id, episodes[1].timestamp)
    assert np.allclose(llm.score(episodes[1], history), [0.5, 0.25, 0.25, 0, 0])


def test_changing_the_prompt_invalidates_old_answers(tmp_path, monkeypatch):
    split, items, episodes, paths = setup(tmp_path)
    (tmp_path / "responses-01.json").write_text(
        json.dumps({k: {"A": 1, "B": 0, "C": 0, "D": 0, "E": 0} for k in keys(paths[0])})
    )
    monkeypatch.setattr(agent, "INSTRUCTIONS", agent.INSTRUCTIONS + " Be brief.")
    llm = OfflineAgent(items, tmp_path, n_recent=5)
    evaluate(llm, split, episodes)
    assert llm.unparsed == 6


def test_subset_has_one_episode_per_user():
    split = make_split(synthetic_log(), SplitConfig(min_history=5))
    episodes = sample_episodes(split, 10)
    assert len(episodes) == 10 == len({e.user_id for e in episodes})
    assert episodes == sample_episodes(split, 10)


def test_excluding_users_keeps_the_other_picks():
    split = make_split(synthetic_log(), SplitConfig(min_history=5))
    full = sample_episodes(split, 10)
    dropped = {full[0].user_id, full[3].user_id}
    assert sample_episodes(split, 10, exclude_users=dropped) == [e for e in full if e.user_id not in dropped]
