"""`user-sim`: build episodes from MovieLens and score every predictor on them.

`eval` scores the baselines. `llm-export` writes prompts for the offline LLM agent on a
fixed subset of episodes, and `llm-eval` scores its answers next to the baselines on that
same subset."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from user_simulation.agent import DEFAULT_RUNS_DIR, OfflineAgent, excluded_users, export, sample_episodes
from user_simulation.baselines import ItemKNN, NewestRelease, Popularity, Random
from user_simulation.data import DATASETS, SplitConfig, load, make_split, make_validation_split
from user_simulation.eval import Evaluation, bootstrap_ci, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(prog="user-sim")
    commands = parser.add_subparsers(dest="command", required=True)
    split_args = argparse.ArgumentParser(add_help=False)
    split_args.add_argument("--dataset", choices=sorted(DATASETS), default="ml-1m")
    split_args.add_argument(
        "--split", choices=["validation", "test"], default="validation",
        help="tune on validation; score on test only to report a final number",
    )
    defaults = SplitConfig()
    split_args.add_argument("--test-fraction", type=float, default=defaults.test_fraction)
    split_args.add_argument("--min-history", type=int, default=defaults.min_history)
    split_args.add_argument("--episodes-per-user", type=int, default=defaults.episodes_per_user)
    split_args.add_argument("--slate-size", type=int, default=defaults.slate_size)
    split_args.add_argument(
        "--negatives", choices=["matched", "popularity", "uniform"], default=defaults.negative_sampling
    )
    split_args.add_argument("--seed", type=int, default=defaults.seed)
    llm_args = argparse.ArgumentParser(add_help=False)
    llm_args.add_argument("--n-episodes", type=int, default=300, help="fixed random subset of episodes")
    llm_args.add_argument("--n-recent", type=int, default=20, help="history items shown in the prompt")
    llm_args.add_argument("--run-dir", type=Path, help="default: data/llm/<split>-last<n-recent>")

    commands.add_parser("eval", parents=[split_args], help="score baselines on held-out decision episodes")
    export_cmd = commands.add_parser(
        "llm-export", parents=[split_args, llm_args], help="write LLM agent prompts for a subset"
    )
    export_cmd.add_argument("--batch-size", type=int, default=25)
    commands.add_parser(
        "llm-eval", parents=[split_args, llm_args], help="score the LLM agent's answers vs baselines"
    )
    args = parser.parse_args()

    interactions, items = load(args.dataset)
    config = SplitConfig(
        test_fraction=args.test_fraction,
        min_history=args.min_history,
        episodes_per_user=args.episodes_per_user,
        slate_size=args.slate_size,
        negative_sampling=args.negatives,
        seed=args.seed,
    )
    item_years = items.set_index("item_id")["year"]
    split = (make_split if args.split == "test" else make_validation_split)(
        interactions, config, item_years=item_years
    )
    episodes = split.episodes
    if args.command != "eval":
        run_dir = args.run_dir or DEFAULT_RUNS_DIR / f"{args.split}-last{args.n_recent}"
        episodes = sample_episodes(split, args.n_episodes, seed=args.seed, exclude_users=excluded_users(run_dir))
    cutoff = datetime.fromtimestamp(split.cutoff, tz=timezone.utc).date()
    users = len({e.user_id for e in episodes})
    print(
        f"{args.dataset} ({args.split}): {len(interactions):,} interactions, cutoff {cutoff}, "
        f"{len(episodes):,} episodes from {users:,} users, "
        f"slate {config.slate_size} ({config.negative_sampling} negatives)\n"
    )

    if args.command == "llm-export":
        paths = export(episodes, split, items, run_dir, args.n_recent, args.batch_size)
        print(f"wrote {len(paths)} prompt batches to {run_dir}/; answer each in responses-NN.json")
        return

    predictors = [Random(seed=args.seed), Popularity(), NewestRelease(items), ItemKNN()]
    if args.command == "llm-eval":
        agent = OfflineAgent(items, run_dir, args.n_recent)
        predictors.append(agent)
    results = {}
    for predictor in predictors:
        start = time.perf_counter()
        results[predictor.name] = evaluate(predictor, split, episodes)
        results[predictor.name].summary["seconds"] = time.perf_counter() - start
    print_results(results, reference="popularity")
    if args.command == "llm-eval":
        print(
            f"\n{agent.name}: {agent.unparsed} unparsed of {len(episodes)}, "
            f"~{agent.tokens_per_decision:,.0f} prompt tokens per decision (chars / 4)"
        )


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
