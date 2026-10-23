"""`user-sim eval`: build episodes from MovieLens and score every predictor on them."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import pandas as pd

from user_simulation.baselines import ItemKNN, Popularity, Random
from user_simulation.data import DATASETS, SplitConfig, load, make_split, make_validation_split
from user_simulation.eval import Evaluation, bootstrap_ci, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(prog="user-sim")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("eval", help="score baselines on held-out decision episodes")
    run.add_argument("--dataset", choices=sorted(DATASETS), default="ml-1m")
    run.add_argument(
        "--split", choices=["validation", "test"], default="validation",
        help="tune on validation; score on test only to report a final number",
    )
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
    split = (make_split if args.split == "test" else make_validation_split)(interactions, config)
    cutoff = datetime.fromtimestamp(split.cutoff, tz=timezone.utc).date()
    users = len({e.user_id for e in split.episodes})
    print(
        f"{args.dataset} ({args.split}): {len(interactions):,} interactions, cutoff {cutoff}, "
        f"{len(split.episodes):,} episodes from {users:,} users, "
        f"slate {config.slate_size} ({config.negative_sampling} negatives)\n"
    )

    results = {}
    for predictor in [Random(seed=args.seed), Popularity(), ItemKNN()]:
        start = time.perf_counter()
        results[predictor.name] = evaluate(predictor, split)
        results[predictor.name].summary["seconds"] = time.perf_counter() - start
    print_results(results, reference="popularity")


def print_results(results: dict[str, Evaluation], reference: str, metrics=("hit@1", "gauc")) -> None:
    """Point estimates, 95% user-bootstrap intervals, and paired differences vs `reference`."""
    summary = pd.DataFrame({name: r.summary for name, r in results.items()}).T
    print(summary.to_string(float_format="{:.3f}".format))

    base = results[reference].per_slate
    intervals, diffs = {}, {}
    for name, r in results.items():
        intervals[name] = {m: _interval(*bootstrap_ci(r.per_slate[m].to_numpy(), r.per_slate["user_id"].to_numpy())) for m in metrics}
        if name != reference:
            if not r.per_slate["episode_id"].equals(base["episode_id"]):
                raise ValueError(f"{name} and {reference} were scored on different episodes")
            diffs[name] = {}
            for m in metrics:
                mean, lo, hi = bootstrap_ci((r.per_slate[m] - base[m]).to_numpy(), base["user_id"].to_numpy())
                diffs[name][m] = _interval(mean, lo, hi, signed=True) + (" *" if lo > 0 or hi < 0 else "  ")
    print("\n95% CI (bootstrap over users)")
    print(pd.DataFrame(intervals).T.to_string())
    print(f"\nPaired difference vs {reference} (* = interval excludes 0)")
    print(pd.DataFrame(diffs).T.to_string())


def _interval(mean: float, lo: float, hi: float, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(mean)} [{f.format(lo)}, {f.format(hi)}]"


if __name__ == "__main__":
    main()
