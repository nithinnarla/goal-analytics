"""
Feature engineering for match outcome prediction.

Feature vector per match:
  elo_diff           – home_elo − away_elo (positive = home favoured)
  home_advantage     – 1 if home-field game (not neutral), 0 otherwise
  form_diff          – home recent form − away recent form (avg pts/game)
  scored_diff        – home avg goals scored − away avg goals scored
  conceded_diff      – away avg goals conceded − home avg goals conceded (positive = home better defence)
  elo_sq_diff        – elo_diff² / 10000 (captures non-linearity)

Label encoding:
  2 = home win, 1 = draw, 0 = away win
"""
import numpy as np
import pandas as pd
from typing import Optional


FEATURE_COLS = [
    "elo_diff",
    "home_advantage",
    "form_diff",
    "scored_diff",
    "conceded_diff",
    "elo_sq_diff",
]

LABEL_MAP = {2: "home_win", 1: "draw", 0: "away_win"}
LABEL_IDX = {"home_win": 2, "draw": 1, "away_win": 0}


def _outcome_label(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 2
    if home_score < away_score:
        return 0
    return 1


def build_feature_row(
    home_elo: float,
    away_elo: float,
    is_neutral: bool = False,
    home_form: float = 1.5,
    away_form: float = 1.5,
    home_scored: float = 1.3,
    away_scored: float = 1.3,
    home_conceded: float = 1.1,
    away_conceded: float = 1.1,
) -> np.ndarray:
    """Build a single feature vector for one match."""
    elo_diff = home_elo - away_elo
    return np.array([
        elo_diff,
        0.0 if is_neutral else 1.0,
        home_form - away_form,
        home_scored - away_scored,
        away_conceded - home_conceded,     # positive = home team faces a leakier defence
        (elo_diff ** 2) / 10_000.0,
    ], dtype=float)


def build_match_features_df(
    wc_df: pd.DataFrame,
    elos: dict[str, float],
    form_lookup: dict[str, dict],
    default_elo: float = 1500.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and label vector y from historical WC matches.

    Parameters
    ----------
    wc_df       : DataFrame from data.historical.get_wc_matches()
    elos        : dict team → Elo (from compute_elo_ratings or teams.py preset)
    form_lookup : dict team → {form, avg_scored, avg_conceded}
    default_elo : fallback Elo for unknown teams

    Returns
    -------
    X : pd.DataFrame with FEATURE_COLS columns
    y : pd.Series of labels {0, 1, 2}
    """
    rows, labels = [], []

    def _elo(team: str) -> float:
        return elos.get(team, default_elo)

    def _form(team: str) -> dict:
        return form_lookup.get(team, {"form": 1.5, "avg_scored": 1.3, "avg_conceded": 1.1})

    # itertuples() over the relevant columns is dramatically faster than
    # iterrows() on large frames (~25k rows for the full historical
    # dataset used by the RF/XGBoost models) while producing identical
    # output to the row-by-row version.
    if "neutral" not in wc_df.columns:
        wc_df = wc_df.assign(neutral=False)

    cols = wc_df[["home_team", "away_team", "home_score", "away_score", "neutral"]]
    for home, away, hs, as_, neutral in cols.itertuples(index=False, name=None):
        hs, as_ = int(hs), int(as_)
        neutral = bool(neutral)

        fh, fa = _form(home), _form(away)
        vec = build_feature_row(
            home_elo=_elo(home),
            away_elo=_elo(away),
            is_neutral=neutral,
            home_form=fh["form"],
            away_form=fa["form"],
            home_scored=fh["avg_scored"],
            away_scored=fa["avg_scored"],
            home_conceded=fh["avg_conceded"],
            away_conceded=fa["avg_conceded"],
        )
        rows.append(vec)
        labels.append(_outcome_label(hs, as_))

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    y = pd.Series(labels, name="outcome")
    return X, y
