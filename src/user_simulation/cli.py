"""`user-sim eval`: build episodes from MovieLens and score every predictor on them."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import pandas as pd

from user_simulation.baselines import ItemKNN, Popularity, Random
from user_simulation.data import DATASETS, SplitConfig, load, make_split
from user_simulation.eval import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(prog="user-sim")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("eval", help="score baselines on held-out decision episodes")
    run.add_argument("--dataset", choices=sorted(DATASETS), default="ml-1m")
    defaults = SplitConfig()
    run.add_argument("--test-fraction", type=float, default=defaults.test_fraction)
    run.add_argument("--min-history", type=int, default=defaults.min_history)
    run.add_argument("--episodes-per-user", type=int, default=defaults.episodes_per_user)
    run.add_argument("--slate-size", type=int, default=defaults.slate_size)
    run.add_argument("--negatives", choices=["popularity", "uniform"], default=defaults.negative_sampling)
    run.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()

    interactions, _items = load(args.dataset)
    config = SplitConfig(
        test_fraction=args.test_fraction,
        min_history=args.min_history,
        episodes_per_user=args.episodes_per_user,
        slate_size=args.slate_size,
        negative_sampling=args.negatives,
        seed=args.seed,
    )
    split = make_split(interactions, config)
    cutoff = datetime.fromtimestamp(split.cutoff, tz=timezone.utc).date()
    users = len({e.user_id for e in split.episodes})
    print(
        f"{args.dataset}: {len(interactions):,} interactions, cutoff {cutoff}, "
        f"{len(split.episodes):,} episodes from {users:,} users, "
        f"slate {config.slate_size} ({config.negative_sampling} negatives)\n"
    )

    rows = {}
    for predictor in [Random(seed=args.seed), Popularity(), ItemKNN()]:
        start = time.perf_counter()
        rows[predictor.name] = evaluate(predictor, split) | {"seconds": time.perf_counter() - start}
    print(pd.DataFrame(rows).T.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
