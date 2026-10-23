"""Run any predictor over a split's episodes and score it."""

from __future__ import annotations

import numpy as np

from user_simulation.data.split import Split
from user_simulation.eval.metrics import slate_metrics, summarize
from user_simulation.predictor import Predictor


def evaluate(predictor: Predictor, split: Split) -> dict[str, float]:
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
    return summarize(
        per_slate, np.concatenate(all_scores), np.concatenate(all_labels), predictor.probabilistic
    )
