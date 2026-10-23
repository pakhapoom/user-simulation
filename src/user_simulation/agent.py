"""An LLM agent whose decisions are made offline.

No API is called from here. `export` writes each episode's prompt (the user's recent
history plus the slate, never the label) to batch files; the LLM answers them in a
Claude Code session by writing one JSON response file per batch; `OfflineAgent` reads
the answers back as a `Predictor`. Every answer is keyed by a hash of the episode and the
exact prompt, so editing the prompt can never silently reuse an old answer, and reruns
are free and reproducible.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from user_simulation.data.split import Episode, Split

LETTERS = "ABCDEFGHIJ"
INSTRUCTIONS = """\
You are simulating one MovieLens user at a moment in time. You see their most recent
ratings (1-5 stars) and a slate of candidate movies. Exactly one candidate is the movie
this user actually rated next; the others are movies they never rated. For every
candidate give the probability that it is the one they picked; probabilities sum to 1.
Answer with a JSON object mapping each candidate letter to its probability."""
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


def _describe(item_id: int, items: pd.DataFrame) -> str:
    row = items.loc[item_id]
    year = f" ({row['year']})" if pd.notna(row["year"]) else ""
    return f"{row['title']}{year} | {', '.join(row['genres']) or 'no genres'}"


def render_prompt(episode: Episode, history: pd.DataFrame, items: pd.DataFrame, n_recent: int) -> str:
    """`items` is indexed by item_id."""
    today = datetime.fromtimestamp(episode.timestamp, tz=timezone.utc).date()
    recent = history.tail(n_recent)
    lines = [f"Today is {today}. The user's last {len(recent)} ratings, oldest first:"]
    lines += [f"- {_describe(r.item_id, items)} | {r.rating:g}" for r in recent.itertuples()]
    lines.append("Candidates:")
    lines += [f"{LETTERS[i]}. {_describe(c, items)}" for i, c in enumerate(episode.candidates)]
    return "\n".join(lines)


def prompt_key(episode: Episode, prompt: str) -> str:
    return hashlib.sha256(f"{episode.episode_id}\n{INSTRUCTIONS}\n{prompt}".encode()).hexdigest()[:16]


def export(
    episodes: list[Episode], split: Split, items: pd.DataFrame, run_dir: Path, n_recent: int, batch_size: int
) -> list[Path]:
    """Write prompts in batches; answers go to the matching `responses-NN.json`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    items = items.set_index("item_id")
    paths = []
    for start in range(0, len(episodes), batch_size):
        blocks = [INSTRUCTIONS, ""]
        for ep in episodes[start : start + batch_size]:
            prompt = render_prompt(ep, split.history(ep.user_id, ep.timestamp), items, n_recent)
            blocks += [f"### {prompt_key(ep, prompt)}", prompt, ""]
        path = run_dir / f"prompts-{start // batch_size + 1:02d}.md"
        path.write_text("\n".join(blocks))
        paths.append(path)
    return paths


class OfflineAgent:
    """Reads answers written for `export`ed prompts. Missing or malformed answers count as
    unparsed and score uniformly, so they cost accuracy instead of being dropped."""

    probabilistic = True

    def __init__(self, items: pd.DataFrame, run_dir: Path, n_recent: int, name: str = "llm_agent"):
        self.name = name
        self.n_recent = n_recent
        self._items = items.set_index("item_id")
        self._answers: dict[str, dict] = {}
        for path in sorted(run_dir.glob("responses-*.json")):
            self._answers |= json.loads(path.read_text())
        self.unparsed = 0
        self.prompt_chars: list[int] = []

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def score(self, episode: Episode, history: pd.DataFrame) -> np.ndarray:
        prompt = render_prompt(episode, history, self._items, self.n_recent)
        self.prompt_chars.append(len(INSTRUCTIONS) + len(prompt))
        probs = _parse(self._answers.get(prompt_key(episode, prompt)), len(episode.candidates))
        if probs is None:
            self.unparsed += 1
            return np.full(len(episode.candidates), 1 / len(episode.candidates))
        return probs

    @property
    def tokens_per_decision(self) -> float:
        """Prompt size estimated as characters / 4; there is no API usage to read."""
        return float(np.mean(self.prompt_chars)) / 4


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
