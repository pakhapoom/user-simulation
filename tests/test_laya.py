import json

import numpy as np
import pytest

from test_agent import ITEMS, setup
from test_split import synthetic_log
from user_simulation.agent import Prompter
from user_simulation.data import SplitConfig, make_training_splits
from user_simulation.eval import evaluate
from user_simulation.laya_agent import LayaAgent, export_training


class FakeTok:
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class FakeLaya:
    """Words stand in for tokens; like Laya, a state past `room` is cut from the end."""

    def __init__(self, room):
        self.room, self.states, self.tok, self.cfg = room, [], FakeTok(), {}

    def predict(self, state, questions, max_len=None):
        self.states.append(state)
        n = len(state.split())
        options = list(questions["next"]["criteria"])
        probs = {o: (4.0 if i == 1 else 1.0) / (len(options) + 3) for i, o in enumerate(options)}
        usage = {"input_tokens": min(n, self.room) + 10, "truncated": n > self.room}
        return {"answers": {"next": {"probabilities": probs}}, "usage": usage}


@pytest.fixture(autouse=True)
def fake_room(monkeypatch):
    monkeypatch.setattr(LayaAgent, "_room", lambda self, question: self.model.room)


def test_scores_follow_the_slate_and_long_histories_keep_the_newest_ratings(tmp_path):
    split, episodes, prompter = setup(tmp_path, "full")
    model = FakeLaya(room=40)
    laya = LayaAgent(prompter, model, "multilingual")
    result = evaluate(laya, split, episodes)
    assert result.summary["hit@1"] == np.mean([ep.labels[1] for ep in episodes])
    assert laya.cut >= 1 and laya.overflow == 0

    ep = episodes[0]
    history = split.history(ep.user_id, ep.timestamp)
    assert np.allclose(laya.score(ep, history), [1 / 8, 4 / 8, 1 / 8, 1 / 8, 1 / 8])
    cut = model.states[-1]
    assert len(cut.split()) <= 40 < len(cut.split()) + 12 and "last" in cut  # one more rating would not fit
    shown = cut.splitlines()[1:]
    assert shown == prompter.state(ep, history)[-len(shown):]  # the newest ratings, in order


def test_missing_memos_score_uniformly(tmp_path):
    split, episodes, _ = setup(tmp_path)
    laya = LayaAgent(Prompter(ITEMS, "memo", 5, tmp_path / "memo", tmp_path / "memo"), FakeLaya(room=40), "english")
    evaluate(laya, split, episodes)
    assert laya.unparsed == len(episodes)


def test_training_windows_end_before_the_cutoff_and_export_what_inference_reads(tmp_path):
    log = synthetic_log()
    before = int(log["timestamp"].quantile(0.9))
    years = ITEMS.set_index("item_id")["year"]
    splits = make_training_splits(log, before, SplitConfig(min_history=5), days=3, item_years=years)
    assert splits and splits[0].cutoff == before - 3 * 86_400
    for newer, older in zip(splits, splits[1:]):
        assert older.cutoff == newer.cutoff - 3 * 86_400
    assert all(ep.timestamp < before for s in splits for ep in s.episodes)

    _, _, prompter = setup(tmp_path, "full")
    laya = LayaAgent(prompter, FakeLaya(room=40), "multilingual")
    path = tmp_path / "train.jsonl"
    assert export_training(splits, laya, path) == sum(len(s.episodes) for s in splits)
    row = json.loads(path.read_text().splitlines()[0])
    ep = splits[0].episodes[0]
    question = laya.question(ep)
    assert row == {
        "state": laya.state(ep, splits[0].history(ep.user_id, ep.timestamp), question),
        "questions": question,
        "expected": {"next": "ABCDE"[ep.labels.index(1)]},
    }
