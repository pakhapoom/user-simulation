import numpy as np

from test_agent import ITEMS, setup
from user_simulation.agent import Prompter
from user_simulation.eval import evaluate
from user_simulation.laya_agent import LayaAgent


class FakeLaya:
    """Words stand in for tokens; like Laya, a state past `room` is cut from the end."""

    def __init__(self, room):
        self.room, self.states = room, []

    def tok(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def predict(self, state, questions, max_len=None):
        self.states.append(state)
        n = len(state.split())
        options = list(questions["next"]["criteria"])
        probs = {o: (4.0 if i == 1 else 1.0) / (len(options) + 3) for i, o in enumerate(options)}
        usage = {"input_tokens": min(n, self.room) + 10, "state_tokens": n,
                 "state_tokens_dropped": max(0, n - self.room), "truncated": n > self.room}
        return {"answers": {"next": {"probabilities": probs}}, "usage": usage}


def test_scores_follow_the_slate_and_long_histories_keep_the_newest_ratings(tmp_path):
    split, episodes, prompter = setup(tmp_path, "full")
    model = FakeLaya(room=40)
    laya = LayaAgent(prompter, model, "multilingual")
    result = evaluate(laya, split, episodes)
    assert result.summary["hit@1"] == np.mean([ep.labels[1] for ep in episodes])

    ep = episodes[0]
    history = split.history(ep.user_id, ep.timestamp)
    model.states.clear()
    assert np.allclose(laya.score(ep, history), [1 / 8, 4 / 8, 1 / 8, 1 / 8, 1 / 8])
    assert len(model.states) == 2 and laya.cut >= 1 and laya.overflow == 0
    cut = model.states[-1]
    assert len(cut.split()) <= 40 and "last" in cut
    shown = cut.splitlines()[1:]
    assert shown == prompter.state(ep, history)[-len(shown):]  # the newest ratings, in order


def test_missing_memos_score_uniformly(tmp_path):
    split, episodes, _ = setup(tmp_path)
    laya = LayaAgent(Prompter(ITEMS, "memo", 5, tmp_path / "memo", tmp_path / "memo"), FakeLaya(room=40), "english")
    evaluate(laya, split, episodes)
    assert laya.unparsed == len(episodes)
