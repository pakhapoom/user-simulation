"""Laya, a decision model that reads a state and scores options without generating text.

Laya (huggingface.co/convaiinnovations/laya) is an encoder (ModernBERT-large for the English
checkpoint, mmBERT-base for the multilingual one) with a decision head. Given a state and a
`choice` question, it returns one probability per option from a single forward pass. Here the
state is exactly what the LLM agent's decision prompt shows about the user, from the same
`Prompter` contexts, and the question is a choice over the slate. It runs locally, zero-shot,
with the temperatures the checkpoint ships with.

Laya keeps the *start* of a state that is too long, which for a rating history would drop the
newest ratings. So a history that doesn't fit is cut here instead, to the newest ratings that
do, shown as in the `recent` context. Memo contexts are short enough not to need it.

Needs the optional dependency: `uv sync --extra laya`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from user_simulation.agent import LETTERS, Prompter, describe
from user_simulation.data.split import Episode

REPO = "convaiinnovations/laya"
CHECKPOINTS = {"english": None, "multilingual": "multilingual"}  # subfolder of REPO
CONTEXTS = ("recent", "full", "memo")  # memo-tools needs an agent that can call tools
QUESTION = """\
You are simulating one MovieLens user at a moment in time. Exactly one candidate is the movie
this user actually rated next; the others are movies they never rated. Which one did they rate next?"""


def load(checkpoint: str, device: str | None = None):
    """A Laya model, downloaded to the Hugging Face cache on first use."""
    import laya

    return laya.load(REPO, subfolder=CHECKPOINTS[checkpoint], device=device)


class LayaAgent:
    """Scores each slate with one Laya call (two when the history has to be cut). Missing
    memos count as unparsed and score uniformly, as for the LLM agent. `model` comes from
    `load(checkpoint)`; `max_len` overrides the checkpoint's token budget."""

    probabilistic = True

    def __init__(self, prompter: Prompter, model, checkpoint: str, max_len: int | None = None):
        if prompter.context not in CONTEXTS:
            raise ValueError(f"Laya has no {prompter.context!r} context; use one of {CONTEXTS}")
        self.prompter, self.model, self.max_len = prompter, model, max_len
        self.name = f"laya_{checkpoint[:2]}{max_len or ''}_{prompter.context}"
        self.unparsed = 0
        self.cut = 0  # episodes whose history was cut to the newest ratings that fit
        self.overflow = 0  # episodes Laya still truncated, which should not happen
        self.input_tokens: list[int] = []

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def score(self, episode: Episode, history: pd.DataFrame) -> np.ndarray:
        n = len(episode.candidates)
        lines = self.prompter.state(episode, history)
        if lines is None:
            self.unparsed += 1
            return np.full(n, 1 / n)
        question = {"next": {
            "type": "choice",
            "instructions": QUESTION,
            "criteria": {LETTERS[i]: describe(c, self.prompter.items) for i, c in enumerate(episode.candidates)},
        }}
        result = self._predict("\n".join(lines), question)
        usage = result["usage"]
        if usage["truncated"] and self.prompter.context != "memo":
            self.cut += 1
            room = usage["state_tokens"] - usage["state_tokens_dropped"]
            shown = history.tail(self.prompter.n_recent) if self.prompter.context == "recent" else history
            result = self._predict(self._newest_that_fit(episode, shown, room), question)
            usage = result["usage"]
        self.overflow += bool(usage["truncated"])
        self.input_tokens.append(usage["input_tokens"])
        probs = result["answers"]["next"]["probabilities"]
        probs = np.array([probs[LETTERS[i]] for i in range(n)], dtype=float)
        return probs / probs.sum()  # Laya rounds to 4 decimals

    def _predict(self, state: str, question: dict) -> dict:
        return self.model.predict(state, question, max_len=self.max_len)

    def _newest_that_fit(self, episode: Episode, history: pd.DataFrame, room: int) -> str:
        """The `recent` view of the most ratings, newest first, whose text fits `room` tokens."""
        def text(k: int) -> str:
            return "\n".join(self.prompter.recent_state(episode, history.tail(k)))

        lo, hi = 1, len(history)  # token count grows with k, so binary search the largest that fits
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(self.model.tok(text(mid), add_special_tokens=False)["input_ids"]) <= room:
                lo = mid
            else:
                hi = mid - 1
        return text(lo)

    def tokens_per_decision(self) -> float:
        """Laya's own count of the tokens it read, state plus question."""
        return float(np.mean(self.input_tokens)) if self.input_tokens else 0.0
