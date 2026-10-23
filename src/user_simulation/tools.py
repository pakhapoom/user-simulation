"""Lookup tools for the memo-tools context.

`write_tool_data` runs at export time and stores, per episode, only what the tools may
reveal: the user's ratings strictly before the decision, candidate metadata and
pre-cutoff statistics. The agent then calls

    uv run python -m user_simulation.tools <run_dir> <episode id> <tool> [arg]

which answers from that file and appends the call to `<run_dir>/tool_calls.jsonl`, so
tool output counts towards tokens per decision.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from user_simulation.data.split import Episode

LETTERS = "ABCDEFGHIJ"
TOOLS = ("recent_history", "similar_in_history", "item_details", "history_stats")


def write_tool_data(
    episodes: list[Episode], histories: list[pd.DataFrame], train: pd.DataFrame, items: pd.DataFrame, run_dir: Path
) -> None:
    """`items` is indexed by item_id."""
    from user_simulation.baselines import ItemKNN

    knn = ItemKNN()
    knn.fit(train)
    stats = train.groupby("item_id")["rating"].agg(["size", "mean"])
    out = run_dir / "tooldata"
    out.mkdir(parents=True, exist_ok=True)
    for ep, history in zip(episodes, histories):
        rows = [
            {
                "date": str(datetime.fromtimestamp(r.timestamp, tz=timezone.utc).date()),
                "title": _title(r.item_id, items),
                "genres": list(items.loc[r.item_id, "genres"]),
                "year": None if pd.isna(items.loc[r.item_id, "year"]) else int(items.loc[r.item_id, "year"]),
                "rating": float(r.rating),
            }
            for r in history.itertuples()
        ]
        candidates = {}
        for letter, item in zip(LETTERS, ep.candidates):
            n, mean = (int(stats.loc[item, "size"]), float(stats.loc[item, "mean"])) if item in stats.index else (0, None)
            candidates[letter] = {
                "title": _title(item, items),
                "genres": list(items.loc[item, "genres"]),
                "rated_before_cutoff": n,
                "mean_rating_before_cutoff": mean,
                "similar": _similar(knn, item, history, items),
            }
        (out / f"{ep.episode_id}.json").write_text(json.dumps({"history": rows, "candidates": candidates}))


def _title(item: int, items: pd.DataFrame) -> str:
    year = items.loc[item, "year"]
    return f"{items.loc[item, 'title']}" + ("" if pd.isna(year) else f" ({year})")


def _similar(knn, item: int, history: pd.DataFrame, items: pd.DataFrame, k: int = 5) -> list:
    if item not in knn._index:
        return []
    known = [(i, r) for i, r in zip(history["item_id"], history["rating"]) if i in knn._index]
    if not known:
        return []
    rows = [knn._index[i] for i, _ in known]
    sims = np.asarray((knn._vectors[[knn._index[item]]] @ knn._vectors[rows].T).todense()).ravel()
    top = [j for j in np.argsort(-sims)[:k] if sims[j] > 0]
    return [[_title(known[j][0], items), float(known[j][1]), round(float(sims[j]), 3)] for j in top]


def answer(data: dict, tool: str, arg: str | None) -> str:
    history, candidates = data["history"], data["candidates"]
    if tool == "recent_history":
        n = min(int(arg or 10), 50)
        return "\n".join(f"{r['date']} | {r['title']} | {', '.join(r['genres'])} | {r['rating']:g}" for r in history[-n:])
    if tool in ("similar_in_history", "item_details"):
        if arg not in candidates:
            return f"Unknown candidate {arg!r}; use one of {', '.join(candidates)}."
        c = candidates[arg]
        if tool == "item_details":
            mean = "n/a" if c["mean_rating_before_cutoff"] is None else f"{c['mean_rating_before_cutoff']:.2f}"
            return (
                f"{c['title']} | {', '.join(c['genres'])} | rated by {c['rated_before_cutoff']} people "
                f"before the cutoff, mean rating {mean}"
            )
        if not c["similar"]:
            return f"No movie in the user's history is co-rated with {c['title']}."
        return "\n".join(f"{title} | user rated {rating:g} | similarity {sim:.3f}" for title, rating, sim in c["similar"])
    if tool == "history_stats":
        ratings = [r["rating"] for r in history]
        genres = Counter(g for r in history for g in r["genres"])
        means = {g: np.mean([r["rating"] for r in history if g in r["genres"]]) for g in genres}
        decades = Counter(f"{r['year'] // 10 * 10}s" for r in history if r["year"])
        return (
            f"{len(history)} ratings from {history[0]['date']} to {history[-1]['date']}, mean {np.mean(ratings):.2f}.\n"
            "Top genres: " + ", ".join(f"{g} {n} (mean {means[g]:.1f})" for g, n in genres.most_common(6)) + ".\n"
            "Decades: " + ", ".join(f"{d} {n}" for d, n in sorted(decades.items())) + "."
        )
    return f"Unknown tool {tool!r}; use one of {', '.join(TOOLS)}."


def main(argv: list[str]) -> None:
    if len(argv) not in (3, 4):
        sys.exit("usage: python -m user_simulation.tools <run_dir> <episode id> <tool> [arg]")
    run_dir, episode_id, tool = Path(argv[0]), int(argv[1]), argv[2]
    path = run_dir / "tooldata" / f"{episode_id}.json"
    if not path.exists():
        sys.exit(f"no tool data for episode {episode_id}")
    output = answer(json.loads(path.read_text()), tool, argv[3] if len(argv) == 4 else None)
    with open(run_dir / "tool_calls.jsonl", "a") as log:
        log.write(json.dumps({"episode_id": episode_id, "tool": tool, "arg": argv[3:], "chars": len(output)}) + "\n")
    print(output)


if __name__ == "__main__":
    main(sys.argv[1:])
