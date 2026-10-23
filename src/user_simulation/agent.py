"""An LLM agent whose decisions are made offline.

No API is called from here. `export` writes each episode's prompt (some view of the
user's history plus the slate, never the label) to batch files; the LLM answers them in a
Claude Code session by writing one JSON response file per batch; `OfflineAgent` reads
the answers back as a `Predictor`. Every answer is keyed by a hash of the episode and the
exact prompt, so editing the prompt can never silently reuse an old answer, and reruns
are free and reproducible.

Contexts (what the decision prompt shows about the user):

- recent: their last N ratings.
- full: every rating before the decision.
- memo: only a short profile, written beforehand from the full history without the slate.
- memo-tools: the same profile, plus lookup tools (see `user_simulation.tools`).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from user_simulation.data.split import Episode, Split

CONTEXTS = ("recent", "full", "memo", "memo-tools")
LETTERS = "ABCDEFGHIJ"
INSTRUCTIONS = """\
You are simulating one MovieLens user at a moment in time. You see their most recent
ratings (1-5 stars) and a slate of candidate movies. Exactly one candidate is the movie
this user actually rated next; the others are movies they never rated. For every
candidate give the probability that it is the one they picked; probabilities sum to 1.
Answer with a JSON object mapping each candidate letter to its probability."""
SEEN = {
    "full": "their full rating\nhistory (1-5 stars)",
    "memo": "a short profile of the\nuser, written from their rating history,",
}
TOOLS = """

Before answering you can look things up, from the repository root, with
  uv run python -m user_simulation.tools {run_dir} <episode id> <tool> [arg]
where <tool> is one of:
  recent_history N      the user's last N ratings (N <= 50), with dates
  similar_in_history X  the 5 movies in the user's history most often co-rated with
                        candidate X by other users, with the user's own ratings
  item_details X        candidate X's genres, year, and how many people rated it before
                        the cutoff, with their mean rating
  history_stats         totals: number of ratings, mean rating, top genres and decades
Use as many or as few lookups as you find useful."""
MEMO_INSTRUCTIONS = """\
Write a profile of this MovieLens user from their rating history (1-5 stars), for someone
who will later have to predict which movie they rate next without seeing the history.
At most 120 words: tastes and dislikes (genres, eras, tone, franchises), rating habits,
and what they have been rating most recently. Answer with the profile text only."""
DEFAULT_RUNS_DIR = Path("data/llm")


def sample_episodes(split: Split, n: int, seed: int = 0, exclude_users: set[int] = frozenset()) -> list[Episode]:
    """A fixed random subset, so the agent and every baseline see the same slates.

    One episode per user: a user's later episode has the earlier answer in its history,
    so two of them side by side would leak it to whoever answers the prompts. For the
    same reason `exclude_users` drops users whose later prompts the answerer has already
    read; it filters after sampling so the other picks stay the same.
    """
    rng = np.random.default_rng(seed)
    by_user: dict[int, list[Episode]] = {}
    for ep in split.episodes:
        by_user.setdefault(ep.user_id, []).append(ep)
    users = sorted(by_user)
    users = [users[i] for i in rng.choice(len(users), size=min(n, len(users)), replace=False)]
    picks = [by_user[u][rng.integers(len(by_user[u]))] for u in users]
    return sorted((ep for ep in picks if ep.user_id not in exclude_users), key=lambda ep: ep.episode_id)


def excluded_users(run_dir: Path) -> set[int]:
    """User ids listed one per line in `<run_dir>/excluded_users.txt`, if it exists."""
    path = run_dir / "excluded_users.txt"
    return {int(line) for line in path.read_text().split()} if path.exists() else set()


def describe(item_id: int, items: pd.DataFrame) -> str:
    """`items` is indexed by item_id."""
    row = items.loc[item_id]
    year = f" ({row['year']})" if pd.notna(row["year"]) else ""
    return f"{row['title']}{year} | {', '.join(row['genres']) or 'no genres'}"


def _today(episode: Episode) -> str:
    return str(datetime.fromtimestamp(episode.timestamp, tz=timezone.utc).date())


def _key(*parts: object) -> str:
    return hashlib.sha256("\n".join(map(str, parts)).encode()).hexdigest()[:16]


def _load(directory: Path, pattern: str) -> dict:
    merged: dict = {}
    for path in sorted(directory.glob(pattern)):
        merged |= json.loads(path.read_text())
    return merged


class Prompter:
    """Renders the prompts of one context and computes their cache keys."""

    def __init__(self, items: pd.DataFrame, context: str, n_recent: int, run_dir: Path, memo_dir: Path):
        if context not in CONTEXTS:
            raise ValueError(f"unknown context {context!r}")
        self.items = items.set_index("item_id")
        self.context, self.n_recent, self.run_dir, self.memo_dir = context, n_recent, run_dir, memo_dir
        self.uses_memo = context.startswith("memo")
        if context == "recent":
            self.instructions = INSTRUCTIONS  # unchanged since step 4, so its answers stay valid
        else:
            seen = SEEN["full" if context == "full" else "memo"]
            self.instructions = INSTRUCTIONS.replace("their most recent\nratings (1-5 stars)", seen)
        if context == "memo-tools":
            self.instructions += TOOLS.format(run_dir=run_dir)
        self.memos = _load(memo_dir, "memos-*.json") if self.uses_memo else {}

    def _ratings(self, history: pd.DataFrame) -> list[str]:
        return [f"- {describe(r.item_id, self.items)} | {r.rating:g}" for r in history.itertuples()]

    def memo_prompt(self, episode: Episode, history: pd.DataFrame) -> tuple[str, str]:
        text = "\n".join(
            [f"The user's {len(history)} ratings up to {_today(episode)}, oldest first:", *self._ratings(history)]
        )
        return _key(episode.episode_id, MEMO_INSTRUCTIONS, text), text

    def prompt(self, episode: Episode, history: pd.DataFrame) -> tuple[str, str] | None:
        """(key, text) of the decision prompt, or None while its memo is still missing."""
        if self.context == "recent":
            recent = history.tail(self.n_recent)
            lines = [f"Today is {_today(episode)}. The user's last {len(recent)} ratings, oldest first:"]
            lines += self._ratings(recent)
        elif self.context == "full":
            lines = [f"Today is {_today(episode)}. The user's {len(history)} ratings so far, oldest first:"]
            lines += self._ratings(history)
        else:
            memo = self.memos.get(self.memo_prompt(episode, history)[0])
            if memo is None:
                return None
            lines = [f"Today is {_today(episode)}. Profile of the user:", memo.strip()]
            if self.context == "memo-tools":
                lines.append(f"Episode id for lookups: {episode.episode_id}")
        lines.append("Candidates:")
        lines += [f"{LETTERS[i]}. {describe(c, self.items)}" for i, c in enumerate(episode.candidates)]
        text = "\n".join(lines)
        return _key(episode.episode_id, self.instructions, text), text


def export(episodes: list[Episode], split: Split, prompter: Prompter, batch_size: int) -> tuple[str, list[Path]]:
    """Write the next batch of work: memo prompts while any memo is missing (answers go to
    `<memo_dir>/memos-NN.json`), otherwise decision prompts (answers go to
    `<run_dir>/responses-NN.json`). Returns which kind was written and the files."""
    histories = [split.history(ep.user_id, ep.timestamp) for ep in episodes]
    if prompter.uses_memo:
        todo = [prompter.memo_prompt(ep, h) for ep, h in zip(episodes, histories)]
        todo = [(key, text) for key, text in todo if key not in prompter.memos]
        if todo:
            return "memo", _write_batches(todo, MEMO_INSTRUCTIONS, prompter.memo_dir, "memo-prompts", batch_size)
    if prompter.context == "memo-tools":
        from user_simulation.tools import write_tool_data

        write_tool_data(episodes, histories, split.train, prompter.items, prompter.run_dir)
    prompts = [prompter.prompt(ep, h) for ep, h in zip(episodes, histories)]
    return "decision", _write_batches(prompts, prompter.instructions, prompter.run_dir, "prompts", batch_size)


def _write_batches(prompts: list[tuple[str, str]], header: str, directory: Path, stem: str, size: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(prompts), size):
        blocks = [header, ""]
        for key, text in prompts[start : start + size]:
            blocks += [f"### {key}", text, ""]
        path = directory / f"{stem}-{start // size + 1:02d}.md"
        path.write_text("\n".join(blocks))
        paths.append(path)
    return paths


class OfflineAgent:
    """Reads answers written for `export`ed prompts. Missing or malformed answers count as
    unparsed and score uniformly, so they cost accuracy instead of being dropped."""

    probabilistic = True

    def __init__(self, prompter: Prompter, name: str | None = None):
        self.prompter = prompter
        self.name = name or f"llm_{prompter.context}"
        self._answers = _load(prompter.run_dir, "responses-*.json")
        self._tool_chars: dict[int, int] = {}
        log = prompter.run_dir / "tool_calls.jsonl"
        for line in log.read_text().splitlines() if log.exists() else []:
            call = json.loads(line)
            self._tool_chars[call["episode_id"]] = self._tool_chars.get(call["episode_id"], 0) + call["chars"]
        self.unparsed = 0
        self.decision_chars: list[int] = []
        self.memo_chars: list[int] = []

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def score(self, episode: Episode, history: pd.DataFrame) -> np.ndarray:
        rendered = self.prompter.prompt(episode, history)
        answer = None
        if rendered is not None:
            key, text = rendered
            answer = self._answers.get(key)
            tools = self._tool_chars.get(episode.episode_id, 0)
            self.decision_chars.append(len(self.prompter.instructions) + len(text) + tools)
        if self.prompter.uses_memo:
            memo_key, memo_text = self.prompter.memo_prompt(episode, history)
            self.memo_chars.append(
                len(MEMO_INSTRUCTIONS) + len(memo_text) + len(self.prompter.memos.get(memo_key, ""))
            )
        probs = _parse(answer, len(episode.candidates))
        if probs is None:
            self.unparsed += 1
            return np.full(len(episode.candidates), 1 / len(episode.candidates))
        return probs

    def tokens_per_decision(self) -> tuple[float, float]:
        """(at decision time, including writing the memo), estimated as characters / 4:
        there is no API usage to read. Decision time counts tool outputs."""
        decision = float(np.mean(self.decision_chars)) / 4 if self.decision_chars else 0.0
        memo = float(np.mean(self.memo_chars)) / 4 if self.memo_chars else 0.0
        return decision, decision + memo


def _parse(answer: dict | None, n: int) -> np.ndarray | None:
    if not isinstance(answer, dict) or set(answer) != set(LETTERS[:n]):
        return None
    try:
        probs = np.array([float(answer[letter]) for letter in LETTERS[:n]])
    except (TypeError, ValueError):
        return None
    if (probs < 0).any() or probs.sum() <= 0:
        return None
    return probs / probs.sum()  # absorbs rounding so each slate is a distribution
