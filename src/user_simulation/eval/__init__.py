"""Run any predictor over a split's episodes and score it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from user_simulation.data.split import Split
from user_simulation.eval.metrics import bootstrap_ci, slate_metrics, summarize
from user_simulation.predictor import Predictor

__all__ = ["Evaluation", "bootstrap_ci", "evaluate"]


@dataclass
class Evaluation:
    summary: dict[str, float]
    per_slate: pd.DataFrame  # one row per episode: episode_id, user_id and the slate metrics


def evaluate(predictor: Predictor, split: Split) -> Evaluation:
    predictor.fit(split.train)
    per_slate, all_scores, all_labels = [], [], []
    for episode in split.episodes:
        history = split.history(episode.user_id, episode.timestamp)
        scores = np.asarray(predictor.score(episode, history), dtype=float)
        labels = np.asarray(episode.labels)
        if scores.shape != labels.shape:
            raise ValueError(f"{predictor.name} returned {scores.shape} scores for {labels.shape} candidates")
        per_slate.append(slate_metrics(scores, labels))
        all_scores.append(scores)
        all_labels.append(labels)
    summary = summarize(
        per_slate, np.concatenate(all_scores), np.concatenate(all_labels), predictor.probabilistic
    )
    ids = pd.DataFrame(
        {"episode_id": [e.episode_id for e in split.episodes], "user_id": [e.user_id for e in split.episodes]}
    )
    return Evaluation(summary, pd.concat([ids, pd.DataFrame(per_slate)], axis=1))
