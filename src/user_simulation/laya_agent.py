"""Laya, a decision model that reads a state and scores options without generating text.

Laya (huggingface.co/convaiinnovations/laya) is an encoder (ModernBERT-large for the English
checkpoint, mmBERT-base for the multilingual one) with a decision head. Given a state and a
`choice` question, it returns one probability per option from a single forward pass. Here the
state is exactly what the LLM agent's decision prompt shows about the user, from the same
`Prompter` contexts, and the question is a choice over the slate. It runs locally, with the
temperatures the checkpoint ships with: zero-shot from the published checkpoints, or from one
fine-tuned on `export_training` rows (see notebooks/laya_finetune.ipynb).

Laya keeps the *start* of a state that is too long, at inference and in training, which for a
rating history would drop the newest ratings. So a history that doesn't fit is cut here
instead, to the newest ratings that do, shown as in the `recent` context. Memo contexts are
short enough not to need it.

Needs the optional dependency: `uv sync --extra laya`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from user_simulation.agent import LETTERS, Prompter, describe
from user_simulation.data.split import Episode, Split

REPO = "convaiinnovations/laya"
CHECKPOINTS = {"english": None, "multilingual": "multilingual"}  # subfolder of REPO
CONTEXTS = ("recent", "full", "memo")  # memo-tools needs an agent that can call tools
QUESTION = """\
You are simulating one MovieLens user at a moment in time. Exactly one candidate is the movie
this user actually rated next; the others are movies they never rated. Which one did they rate next?"""


def load(checkpoint: str, device: str | None = None):
    """A published checkpoint by name, downloaded to the Hugging Face cache on first use, or
    a fine-tuned one by its directory."""
    import laya

    if checkpoint in CHECKPOINTS:
        return laya.load(REPO, subfolder=CHECKPOINTS[checkpoint], device=device)
    return laya.load(checkpoint, device=device)


def short_name(checkpoint: str) -> str:
    return checkpoint[:2] if checkpoint in CHECKPOINTS else Path(checkpoint).name


class LayaAgent:
    """Scores each slate with one Laya call. Missing memos count as unparsed and score
    uniformly, as for the LLM agent. `model` comes from `load(checkpoint)`; `max_len`
    overrides the checkpoint's token budget."""

    probabilistic = True

    def __init__(self, prompter: Prompter, model, checkpoint: str, max_len: int | None = None):
        if prompter.context not in CONTEXTS:
            raise ValueError(f"Laya has no {prompter.context!r} context; use one of {CONTEXTS}")
        self.prompter, self.model, self.max_len = prompter, model, max_len
        self.name = f"laya_{short_name(checkpoint)}{max_len or ''}_{prompter.context}"
        self.unparsed = 0
        self.cut = 0  # episodes whose history was cut to the newest ratings that fit
        self.overflow = 0  # episodes Laya still truncated, which should not happen
        self.input_tokens: list[int] = []

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def score(self, episode: Episode, history: pd.DataFrame) -> np.ndarray:
        n = len(episode.candidates)
        question = self.question(episode)
        state = self.state(episode, history, question)
        if state is None:
            self.unparsed += 1
            return np.full(n, 1 / n)
        result = self.model.predict(state, question, max_len=self.max_len)
        self.overflow += bool(result["usage"]["truncated"])
        self.input_tokens.append(result["usage"]["input_tokens"])
        probs = result["answers"]["next"]["probabilities"]
        probs = np.array([probs[LETTERS[i]] for i in range(n)], dtype=float)
        return probs / probs.sum()  # Laya rounds to 4 decimals

    def question(self, episode: Episode) -> dict:
        criteria = {LETTERS[i]: describe(c, self.prompter.items) for i, c in enumerate(episode.candidates)}
        return {"next": {"type": "choice", "instructions": QUESTION, "criteria": criteria}}

    def state(self, episode: Episode, history: pd.DataFrame, question: dict) -> str | None:
        """The prompter's view of the user, with a history that doesn't fit beside `question`
        cut to the newest ratings that do; None while its memo is still missing."""
        lines = self.prompter.state(episode, history)
        if lines is None:
            return None
        room = self._room(question)
        if self.prompter.context == "memo" or self._tokens("\n".join(lines)) <= room:
            return "\n".join(lines)
        self.cut += 1
        shown = history.tail(self.prompter.n_recent) if self.prompter.context == "recent" else history

        def text(k: int) -> str:
            return "\n".join(self.prompter.recent_state(episode, shown.tail(k)))

        lo, hi = 1, len(shown)  # token count grows with k, so binary search the largest that fits
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._tokens(text(mid)) <= room:
                lo = mid
            else:
                hi = mid - 1
        return text(lo)

    def _room(self, question: dict) -> int:
        """State tokens the question leaves inside the token budget, as Laya computes it."""
        from laya.common import state_room
        from laya.train import to_internal

        max_len = self.max_len or self.model.cfg.get("max_len", 512)
        head_max_len = self.model.cfg.get("head_max_len", 192)
        return state_room(self.model.tok, to_internal("next", question["next"]), max_len, head_max_len)

    def _tokens(self, text: str) -> int:
        return len(self.model.tok(text.replace(self.model.tok.mask_token, " "), add_special_tokens=False)["input_ids"])

    def tokens_per_decision(self) -> float:
        """Laya's own count of the tokens it read, state plus question."""
        return float(np.mean(self.input_tokens)) if self.input_tokens else 0.0


def export_training(splits: list[Split], agent: LayaAgent, path: Path) -> int:
    """Write every episode of `splits` as a `laya-train` row, `{state, questions, expected}`,
    with the state cut exactly as `agent` cuts it at inference. Returns the number of rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w") as out:
        for split in splits:
            for ep in split.episodes:
                question = agent.question(ep)
                state = agent.state(ep, split.history(ep.user_id, ep.timestamp), question)
                if state is None:
                    continue
                row = {"state": state, "questions": question, "expected": {"next": LETTERS[ep.labels.index(1)]}}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
    return rows
