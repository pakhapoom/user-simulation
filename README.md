# user-simulation

An experiment in building an LLM agent that simulates a user's decision — e.g. *given
everything this user has done up until now, will they click/watch this movie?* — and
measuring whether it actually predicts real behavior better than classic recommenders.

## Workflow

```mermaid
flowchart LR
    A["Interaction log<br/>MovieLens"] --> B["Temporal split<br/>past / future"]
    B --> C["Episodes<br/>history + slate of candidates"]
    C --> D{"Predictor"}
    D --> D1["Baselines<br/>random, popularity, item-kNN"]
    D --> D2["LLM agent<br/>planned"]
    D1 --> E["Metrics<br/>hit@1, gauc, auc, calibration"]
    D2 --> E
    E --> F["Compare: accuracy,<br/>calibration, cost per decision"]
```

1. **Load** a real interaction log (who watched what, and when).
2. **Split** it at a point in time: the past is what the simulator may learn from, the
   future is what it has to predict.
3. **Build episodes**: each future interaction becomes a decision — the user's history up
   to that moment plus a slate of movies, one of which they actually chose.
4. **Predict**: every simulator scores the same slates through one shared interface, from
   simple baselines to the LLM agent.
5. **Score** the predictions against what users really did, and compare simulators on
   accuracy, calibration and cost.

## Framing

Every simulator answers the same question on the same **episodes**:

- A global time **cutoff** splits the log. Global statistics (popularity, item
  similarity) are learned only from interactions before it.
- Each post-cutoff interaction becomes an episode: the user is shown a **slate** of
  candidates, one of which they actually chose, plus sampled negatives.
- The simulator sees the user's own history strictly before the episode, then scores
  every candidate.

Negatives are sampled only from items that already existed at decision time and that the
user never rated. By default (`matched`) they come from films released in the **same year**
as the real choice, weighted by how often each was rated in the last 100,000 ratings
before the decision. Year matching matters: episodes are a returning user's first ratings
after the cutoff, and those are mostly new releases (45% of choices on validation are
1999–2000 films vs 14% of popularity-weighted negatives). Without it, "pick the newest
film on the slate" scores hit@1 ≈ 0.46. `popularity` (pre-cutoff counts) and `uniform`
are kept for comparison.

Metrics: `hit@1` (did the top pick match the real choice), `mrr`, `ndcg`, `gauc`
(per-slate AUC), pooled `auc`; plus `log_loss` and `ece` for predictors that output
probabilities. Ties count as their random-ordering expectation, so constant scores give
chance-level numbers.

`hit@1` and `gauc` come with 95% bootstrap intervals, plus a paired difference against the
popularity baseline on the same episodes. The bootstrap resamples whole users, since one
user's slates are not independent; random vs random with another seed comes out as no
difference.

## Usage

```bash
uv sync
uv run user-sim eval                                    # validation split: ml-1m, slate of 5, popularity negatives
uv run user-sim eval --split test                       # test split: only for reporting final numbers
uv run user-sim eval --negatives uniform --slate-size 10
uv run pytest
```

The LLM agent runs offline: `llm-export` writes prompts for a fixed subset of episodes
(one per user, no labels) to `data/llm/<split>-last<N>/prompts-NN.md`; the LLM answers
each batch in a `responses-NN.json` mapping prompt key to `{"A": p, ...}`; `llm-eval`
scores those answers next to the baselines on the same subset.

```bash
uv run user-sim llm-export --split test        # 300 episodes, last 20 ratings per prompt
uv run user-sim llm-eval --split test
```

Each answer is keyed by a hash of the episode and the exact prompt, so editing the prompt
invalidates old answers. Users listed in `<run dir>/excluded_users.txt` are dropped from
the subset (see the step-4 notes below).

Datasets download to `data/raw/` on first use (git-ignored).

## Baselines (ml-1m test split, default settings, seed 0)

4,874 episodes from 1,044 users; cutoff 2000-12-02.

| negatives  | predictor      | hit@1 | gauc  | auc   |
|------------|----------------|-------|-------|-------|
| matched    | random         | 0.187 | 0.493 | 0.489 |
| matched    | popularity     | 0.384 | 0.644 | 0.633 |
| matched    | newest_release | 0.199 | 0.500 | 0.500 |
| matched    | item_knn       | 0.399 | 0.659 | 0.620 |
| popularity | random         | 0.201 | 0.500 | 0.499 |
| popularity | popularity     | 0.293 | 0.565 | 0.564 |
| popularity | newest_release | 0.456 | 0.659 | 0.659 |
| popularity | item_knn       | 0.298 | 0.561 | 0.549 |
| uniform    | random         | 0.199 | 0.498 | 0.498 |
| uniform    | popularity     | 0.620 | 0.835 | 0.832 |
| uniform    | newest_release | 0.440 | 0.642 | 0.642 |
| uniform    | item_knn       | 0.580 | 0.815 | 0.795 |

Observations so far:

- Item-kNN uses the user's last 10 items, picked on validation (matched negatives; the
  same value also won under popularity negatives):

  | max_history | 10    | 20    | 50    | 100   |
  |-------------|-------|-------|-------|-------|
  | hit@1       | 0.389 | 0.377 | 0.372 | 0.367 |
  | gauc        | 0.647 | 0.635 | 0.630 | 0.627 |

- Recency dominates: the median gap between an episode and the user's previous
  interaction is 0 days, and kNN gets worse with every extra slice of history. That's a
  context-engineering result before any LLM is involved: more history is not more signal.
- `newest_release` (score = release year) is the check that the benchmark doesn't leak
  release recency: it is at chance with matched negatives and the best baseline without.
- `ml-latest-small` yields only ~130 episodes under a global cutoff (most users rate in
  a single burst), so it's for smoke tests only.

## LLM agent, first run (step 4, test split, popularity negatives)

Claude (Opus 5.5, in a Claude Code session, no API) answered all 293 prompts of the test
subset: the user's last 20 ratings (title, year, genres, stars) and the 5-movie slate.
One fixed prompt, not tuned; 0 unparsed answers; ~421 prompt tokens per decision.

| predictor      | hit@1 [95% CI]        | gauc [95% CI]         | log_loss | ece   |
|----------------|-----------------------|-----------------------|----------|-------|
| random         | 0.235 [0.188, 0.283]  | 0.515 [0.474, 0.558]  |          |       |
| popularity     | 0.280 [0.229, 0.331]  | 0.565 [0.523, 0.610]  |          |       |
| item_knn       | 0.287 [0.232, 0.338]  | 0.540 [0.497, 0.584]  |          |       |
| newest_release | 0.487 [0.432, 0.544]  | 0.683 [0.639, 0.726]  |          |       |
| llm_agent      | 0.489 [0.435, 0.544]  | 0.733 [0.698, 0.767]  | 0.446    | 0.068 |

This run used the old popularity negatives and is what exposed the leak; the
step-5 runs below use matched negatives. Paired against `newest_release`: hit@1 +0.001 [−0.066, +0.073], gauc +0.049
[0.000, +0.102]. The agent beats popularity and kNN by ~0.2 hit@1, but almost all of
that is the release-recency leak above; on ranking the full slate it is at best slightly
better. A constant 0.2 would score log-loss 0.500, so its probabilities carry some signal
and are reasonably calibrated.

How the subset was built, and what it excludes:

- One episode per user. A user's later prompt contains their earlier answer in its
  history, so two episodes of the same user side by side leak it.
- 7 users (`excluded_users.txt` in that run directory) were dropped because their later
  prompts had already been read while an earlier, leaky export was being answered.
- Same model answering as wrote the code: it saw only aggregate baseline numbers on test
  before answering, never per-episode labels.

## Caveats

- MovieLens records *ratings*, not impressions. "Rated" stands in for "watched", and
  "never rated" stands in for "shown but skipped". Ratings also arrive in bulk sessions.
- Hyper-parameters must be chosen on the validation split (`--split validation`, the
  default), not on these episodes. Validation reruns the same split on the log before the
  test cutoff, with the last 14 days as its future: cutoff 2000-11-18, 656 episodes from
  174 users.

## Layout

```
src/user_simulation/
  data/        MovieLens loader, temporal split, episode + negative sampling
  baselines/   random, popularity, newest-release, item-kNN
  agent.py     offline LLM agent: prompt export, answer cache, Predictor
  eval/        metrics and the evaluate() runner
  predictor.py the Predictor protocol every simulator (incl. the LLM agent) implements
```
