"""`user-sim`: build episodes from MovieLens and score every predictor on them.

`eval` scores the baselines. `llm-export` writes prompts for the offline LLM agent on a
fixed subset of episodes, and `llm-eval` scores its answers (one or more contexts) next to
the baselines on that same subset. `laya-eval` runs the Laya decision model locally on the
same subset and contexts, and `laya-export` writes training rows for fine-tuning it."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from user_simulation.agent import (
    CONTEXTS,
    DEFAULT_RUNS_DIR,
    OfflineAgent,
    Prompter,
    excluded_users,
    export,
    sample_episodes,
)
from user_simulation.baselines import ItemKNN, NewestRelease, Popularity, Random
from user_simulation.data import DATASETS, SplitConfig, load, make_split, make_training_splits, make_validation_split
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
    llm_args.add_argument("--context", choices=CONTEXTS, nargs="+", default=["recent"])
    llm_args.add_argument("--n-recent", type=int, default=20, help="history items shown by the recent context")
    llm_args.add_argument(
        "--run-dir", type=Path, help="one context only; default: data/llm/<split>-<negatives>-<context>"
    )
    llm_args.add_argument("--memo-dir", type=Path, help="default: data/llm/<split>-<negatives>-memo")

    commands.add_parser("eval", parents=[split_args], help="score baselines on held-out decision episodes")
    export_cmd = commands.add_parser(
        "llm-export", parents=[split_args, llm_args], help="write LLM agent prompts for a subset"
    )
    export_cmd.add_argument("--batch-size", type=int, default=25)
    eval_cmd = commands.add_parser(
        "llm-eval", parents=[split_args, llm_args], help="score the LLM agent's answers vs baselines"
    )
    eval_cmd.add_argument("--reference", default="popularity", help="predictor for paired differences")
    laya_cmd = commands.add_parser(
        "laya-eval", parents=[split_args, llm_args], help="score the Laya decision model vs baselines"
    )
    laya_cmd.add_argument(
        "--checkpoint", nargs="+", default=["multilingual"],
        help="english, multilingual, or the directory of a fine-tuned checkpoint",
    )
    laya_cmd.add_argument("--max-len", type=int, help="token budget per decision; default: the checkpoint's")
    laya_cmd.add_argument("--device", help="torch device; default: Laya picks one")
    laya_cmd.add_argument("--reference", default="popularity", help="predictor for paired differences")
    train_cmd = commands.add_parser(
        "laya-export", parents=[split_args], help="write Laya fine-tuning rows from before the validation cutoff"
    )
    train_cmd.add_argument("--context", choices=["recent", "full"], default="full")
    train_cmd.add_argument("--n-recent", type=int, default=20, help="history items shown by the recent context")
    train_cmd.add_argument("--checkpoint", default="multilingual", help="the checkpoint that will be fine-tuned")
    train_cmd.add_argument("--max-len", type=int, help="token budget per decision; default: the checkpoint's")
    train_cmd.add_argument("--days", type=int, default=14, help="length of each training window")
    train_cmd.add_argument("--out", type=Path, help="default: data/laya/<negatives>-<context>")
    args = parser.parse_args()
    if args.command == "laya-export":
        return laya_export(args)
    if args.command != "eval" and args.run_dir and len(args.context) > 1:
        parser.error("--run-dir needs a single --context")
    if args.command == "llm-export" and len(args.context) > 1:
        parser.error("llm-export takes a single --context")

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
        base = f"{args.split}-{args.negatives}"
        memo_dir = args.memo_dir or DEFAULT_RUNS_DIR / f"{base}-memo"
        prompters = []
        for context in args.context:
            name = f"{context}{args.n_recent}" if context == "recent" else context
            run_dir = args.run_dir or DEFAULT_RUNS_DIR / f"{base}-{name}"
            prompters.append(Prompter(items, context, args.n_recent, run_dir, memo_dir))
        excluded = set().union(*(excluded_users(p.run_dir) for p in prompters))
        episodes = sample_episodes(split, args.n_episodes, seed=args.seed, exclude_users=excluded)
    cutoff = datetime.fromtimestamp(split.cutoff, tz=timezone.utc).date()
    users = len({e.user_id for e in episodes})
    print(
        f"{args.dataset} ({args.split}): {len(interactions):,} interactions, cutoff {cutoff}, "
        f"{len(episodes):,} episodes from {users:,} users, "
        f"slate {config.slate_size} ({config.negative_sampling} negatives)\n"
    )

    if args.command == "llm-export":
        kind, paths = export(episodes, split, prompters[0], args.batch_size)
        if kind == "memo":
            print(f"wrote {len(paths)} memo prompt batches to {memo_dir}/; answer each in memos-NN.json, then rerun")
        else:
            print(f"wrote {len(paths)} prompt batches to {prompters[0].run_dir}/; answer each in responses-NN.json")
        return

    predictors = [Random(seed=args.seed), Popularity(), NewestRelease(items), ItemKNN()]
    agents = [OfflineAgent(p) for p in prompters] if args.command == "llm-eval" else []
    if args.command == "laya-eval":
        from user_simulation import laya_agent

        if bad := [p.context for p in prompters if p.context not in laya_agent.CONTEXTS]:
            parser.error(f"laya-eval has no context {bad[0]!r}; use {', '.join(laya_agent.CONTEXTS)}")
        for checkpoint in args.checkpoint:
            model = laya_agent.load(checkpoint, args.device)
            agents += [laya_agent.LayaAgent(p, model, checkpoint, args.max_len) for p in prompters]
    results = {}
    for predictor in predictors + agents:
        start = time.perf_counter()
        results[predictor.name] = evaluate(predictor, split, episodes)
        results[predictor.name].summary["seconds"] = time.perf_counter() - start
    print_results(results, reference=args.reference if args.command != "eval" else "popularity")
    if args.command == "laya-eval":
        for agent in agents:
            print(
                f"\n{agent.name}: {agent.unparsed} unparsed of {len(episodes)}, {agent.cut} histories cut to fit, "
                f"{agent.overflow} still truncated, {agent.tokens_per_decision():,.0f} tokens per decision, "
                f"{results[agent.name].summary['seconds'] / len(episodes):.2f} s per decision"
            )
        return
    for agent in agents:
        decision, total = agent.tokens_per_decision()
        print(
            f"\n{agent.name}: {agent.unparsed} unparsed of {len(episodes)}, ~{decision:,.0f} tokens per decision"
            + (f" (~{total:,.0f} including writing the memo)" if agent.prompter.uses_memo else "")
            + " (chars / 4)"
        )


def laya_export(args: argparse.Namespace) -> None:
    """Training rows from rolling windows before the validation cutoff, so that neither the
    validation nor the test episodes are ever trained on, plus the validation episodes for
    `laya-train --eval` to report on."""
    from user_simulation import laya_agent

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
    validation = make_validation_split(interactions, config, item_years=item_years)
    training = make_training_splits(interactions, validation.cutoff, config, args.days, item_years=item_years)
    name = f"{args.context}{args.n_recent}" if args.context == "recent" else args.context
    out = args.out or Path("data/laya") / f"{args.negatives}-{name}"
    prompter = Prompter(items, args.context, args.n_recent, out, out)
    agent = laya_agent.LayaAgent(prompter, laya_agent.load(args.checkpoint, "cpu"), args.checkpoint, args.max_len)
    rows = laya_agent.export_training(training, agent, out / "train.jsonl")
    oldest = datetime.fromtimestamp(training[-1].cutoff, tz=timezone.utc).date()
    cutoff = datetime.fromtimestamp(validation.cutoff, tz=timezone.utc).date()
    print(f"wrote {rows:,} training rows from {len(training)} windows ({oldest} to {cutoff}), {agent.cut:,} histories cut")
    agent.cut = 0
    rows = laya_agent.export_training([validation], agent, out / "validation.jsonl")
    print(f"wrote {rows:,} validation rows, {agent.cut:,} histories cut, to {out}/")


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
