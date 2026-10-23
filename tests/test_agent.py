import json

import numpy as np
import pandas as pd

from test_split import split_of, synthetic_log
from user_simulation import agent, tools
from user_simulation.agent import OfflineAgent, Prompter, export, sample_episodes
from user_simulation.data import SplitConfig
from user_simulation.eval import evaluate

ITEMS = pd.DataFrame(
    {"item_id": range(60), "title": [f"Movie {i}" for i in range(60)], "year": 1999, "genres": [["Drama"]] * 60}
)


def setup(tmp_path, context="recent"):
    split = split_of(synthetic_log(), SplitConfig(min_history=5))
    episodes = sample_episodes(split, 6)
    prompter = Prompter(ITEMS, context, 5, tmp_path / context, tmp_path / "memo")
    return split, episodes, prompter


def keys(path):
    return [line[4:] for line in path.read_text().splitlines() if line.startswith("### ")]


def test_round_trip_scores_answers_and_counts_bad_ones(tmp_path):
    split, episodes, prompter = setup(tmp_path)
    kind, paths = export(episodes, split, prompter, batch_size=4)
    assert kind == "decision" and len(paths) == 2 and len(keys(paths[0])) == 4
    answers = {k: {"A": 2, "B": 1, "C": 1, "D": 0, "E": 0} for k in keys(paths[0])}
    answers[keys(paths[0])[0]] = {"A": "high"}  # malformed
    (prompter.run_dir / "responses-01.json").write_text(json.dumps(answers))  # batch 2 left unanswered

    llm = OfflineAgent(prompter)
    result = evaluate(llm, split, episodes)
    assert llm.unparsed == 1 + 2
    assert len(result.per_slate) == 6 and "log_loss" in result.summary
    history = split.history(episodes[1].user_id, episodes[1].timestamp)
    assert np.allclose(llm.score(episodes[1], history), [0.5, 0.25, 0.25, 0, 0])


def test_changing_the_prompt_invalidates_old_answers(tmp_path, monkeypatch):
    split, episodes, prompter = setup(tmp_path)
    _, paths = export(episodes, split, prompter, batch_size=10)
    answers = {k: {"A": 1, "B": 0, "C": 0, "D": 0, "E": 0} for k in keys(paths[0])}
    (prompter.run_dir / "responses-01.json").write_text(json.dumps(answers))
    monkeypatch.setattr(agent, "INSTRUCTIONS", agent.INSTRUCTIONS + " Be brief.")
    llm = OfflineAgent(Prompter(ITEMS, "recent", 5, prompter.run_dir, prompter.memo_dir))
    evaluate(llm, split, episodes)
    assert llm.unparsed == 6


def test_memo_contexts_ask_for_memos_first_and_never_show_the_slate_to_the_memo(tmp_path):
    split, episodes, prompter = setup(tmp_path, "memo")
    kind, paths = export(episodes, split, prompter, batch_size=10)
    assert kind == "memo" and paths[0].parent == prompter.memo_dir
    assert "Candidates:" not in paths[0].read_text()
    (prompter.memo_dir / "memos-01.json").write_text(json.dumps({k: "Likes dramas." for k in keys(paths[0])}))

    prompter = Prompter(ITEMS, "memo", 5, prompter.run_dir, prompter.memo_dir)
    kind, paths = export(episodes, split, prompter, batch_size=10)
    text = paths[0].read_text()
    assert kind == "decision" and "Likes dramas." in text and "| 4\n" not in text  # no raw ratings


def test_tools_answer_from_history_before_the_decision_and_log_calls(tmp_path, capsys):
    split, episodes, prompter = setup(tmp_path, "memo-tools")
    _, paths = export(episodes, split, prompter, batch_size=10)
    (prompter.memo_dir / "memos-01.json").write_text(json.dumps({k: "Likes dramas." for k in keys(paths[0])}))
    prompter = Prompter(ITEMS, "memo-tools", 5, prompter.run_dir, prompter.memo_dir)
    export(episodes, split, prompter, batch_size=10)

    ep = episodes[0]
    tools.main([str(prompter.run_dir), str(ep.episode_id), "recent_history", "50"])
    shown = capsys.readouterr().out.splitlines()
    assert len(shown) == min(50, len(split.history(ep.user_id, ep.timestamp)))
    for tool, arg in [("history_stats", []), ("item_details", ["A"]), ("similar_in_history", ["B"])]:
        tools.main([str(prompter.run_dir), str(ep.episode_id), tool, *arg])
    assert len((prompter.run_dir / "tool_calls.jsonl").read_text().splitlines()) == 4
    llm = OfflineAgent(prompter)
    evaluate(llm, split, episodes)
    decision, total = llm.tokens_per_decision()
    assert total > decision > 0


def test_subset_has_one_episode_per_user():
    split = split_of(synthetic_log(), SplitConfig(min_history=5))
    episodes = sample_episodes(split, 10)
    assert len(episodes) == 10 == len({e.user_id for e in episodes})
    assert episodes == sample_episodes(split, 10)


def test_excluding_users_keeps_the_other_picks():
    split = split_of(synthetic_log(), SplitConfig(min_history=5))
    full = sample_episodes(split, 10)
    dropped = {full[0].user_id, full[3].user_id}
    assert sample_episodes(split, 10, exclude_users=dropped) == [e for e in full if e.user_id not in dropped]
