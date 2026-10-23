"""The contract every simulator implements, from popularity to the LLM agent."""

from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd

from user_simulation.data.split import Episode


class Predictor(Protocol):
    name: str
    # True when scores are click probabilities, enabling log-loss / calibration metrics.
    probabilistic: bool

    def fit(self, train: pd.DataFrame) -> None:
        """Learn global statistics from pre-cutoff interactions only."""

    def score(self, episode: Episode, history: pd.DataFrame) -> Sequence[float]:
        """One score per candidate in `episode.candidates`; higher means more likely to click."""
