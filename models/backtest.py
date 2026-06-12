"""
Historical backtest harness — model accuracy on the 2018 & 2022 FIFA World Cups
=================================================================================

Evaluates every prediction model used by the live dashboard (Elo, Elo+Poisson,
Monte Carlo, Logistic Regression, Random Forest, XGBoost) against the ACTUAL
match results of the 2018 and 2022 World Cups.

POINT-IN-TIME TRAINING
-----------------------
Naively evaluating cached_rf_model() / cached_xgb_model() / cached_logreg_model()
on 2018/2022 WC matches would be testing models on data they were trained on —
those functions train on the FULL historical dataset, which already includes
the 2018 and 2022 results (data leakage / in-sample evaluation).

Instead, for each backtested year, _train_point_in_time() builds a SEPARATE
copy of every model using ONLY matches strictly before that tournament's
opening kickoff:
  - Elo ratings + recent-form lookup: replayed only through the cutoff date
  - Logistic Regression: trained on prior World Cups only (mirrors
    cached_logreg_model(), but on the pre-cutoff slice)
  - Random Forest / XGBoost: trained on the "recent era" (RECENT_ERA_START+)
    slice of international results up to the cutoff date (mirrors
    cached_rf_model() / cached_xgb_model())

These point-in-time copies are used ONLY by this backtest — the live
dashboard's cached_*_model() functions are untouched and continue to train on
the full dataset for actual 2026 predictions.

METRICS
-------
For each match, every model produces P(home_win), P(draw), P(away_win):
  - Accuracy: 1 if argmax(prediction) == actual outcome, else 0
  - Brier score (multi-class): sum over the 3 outcomes of
    (predicted_prob - actual_indicator)^2 — ranges 0 (perfect) to 2 (maximally
    wrong, confidently); lower is better

"Monte Carlo (N sims)" derives its outcome probabilities empirically: for each
match, the same Elo -> lambda -> Poisson-scoreline distribution used by
models/monte_carlo.py's tournament simulator is sampled N times (default
5,000) and the resulting win/draw/loss frequencies are reported — a
sampling-based cross-check on the analytical "Elo + Poisson" formula.

Usage
-----
    from models.backtest import run_full_backtest
    results = run_full_backtest()  # {2018: {...}, 2022: {...}}
"""

import random

import pandas as pd

from data.historical import (
    fetch_results,
    get_wc_matches,
    compute_elo_ratings,
    build_form_lookup,
)
from models.elo import match_probabilities
from models.poisson import match_result_probs_poisson, scoreline_distribution
from models.monte_carlo import elo_to_lambda
from models.logistic import LogisticMatchPredictor
from models.random_forest import RandomForestMatchPredictor
from models.xgboost_model import XGBoostMatchPredictor


# Mirrors dashboard/app.py:RECENT_ERA_START — the "recent era" cutoff used to
# train Random Forest / XGBoost on a much larger slice of international
# results than the ~900-match World-Cup-only set the logistic model uses.
RECENT_ERA_START = "2010-01-01"

# Mirrors compute_elo_ratings()'s default home_advantage Elo bonus. Applied
# here to the home team's point-in-time Elo for non-neutral-venue matches
# before computing Elo-derived (Elo W/D/L, Elo+Poisson, Monte Carlo) outcome
# probabilities — keeps those three models internally consistent with how the
# Elo ratings themselves were fit.
HOME_ADVANTAGE_ELO = 100

# Opening-match dates: matches strictly before this date are "available" for
# point-in-time training of that tournament's backtest.
TOURNAMENT_CUTOFFS = {
    2018: "2018-06-14",   # Russia 5-0 Saudi Arabia
    2022: "2022-11-20",   # Qatar 0-2 Ecuador
}

DEFAULT_FORM = {"form": 1.5, "avg_scored": 1.3, "avg_conceded": 1.1}

OUTCOME_CLASSES = ("home_win", "draw", "away_win")


def _actual_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def _brier_score(probs: dict, actual: str) -> float:
    """
    Multi-class Brier score: sum over {home_win, draw, away_win} of
    (predicted_prob - actual_indicator)^2. Range [0, 2]; lower is better.
    """
    return sum(
        (probs.get(c, 0.0) - (1.0 if c == actual else 0.0)) ** 2
        for c in OUTCOME_CLASSES
    )


def _monte_carlo_match_probs(elo_home: float, elo_away: float, n_sims: int = 5000) -> dict:
    """
    Empirical P(home_win)/P(draw)/P(away_win) from n_sims draws of the same
    Elo -> lambda -> Poisson-scoreline distribution used by
    models/monte_carlo.py's tournament simulator (sample_score()).

    Builds the scoreline distribution once and draws n_sims samples from it
    via random.choices(), rather than calling sample_score() in a loop —
    statistically identical, but avoids rebuilding the 81-entry distribution
    dict on every draw.
    """
    lh, la = elo_to_lambda(elo_home, elo_away)
    dist = scoreline_distribution(round(lh, 3), round(la, 3))
    outcomes = list(dist.keys())
    weights = list(dist.values())
    draws = random.choices(outcomes, weights=weights, k=n_sims)

    home_wins = sum(1 for h, a in draws if h > a)
    away_wins = sum(1 for h, a in draws if h < a)
    ties = n_sims - home_wins - away_wins

    n = float(n_sims)
    return {"home_win": home_wins / n, "draw": ties / n, "away_win": away_wins / n}


def _train_point_in_time(full_df: pd.DataFrame, cutoff_date: str) -> dict:
    """
    Build point-in-time copies of every model, trained ONLY on data strictly
    before `cutoff_date`. Used exclusively by run_backtest() — does not touch
    the live dashboard's cached_*_model() functions or their cached state.
    """
    cutoff = pd.to_datetime(cutoff_date)
    pit_df = full_df[full_df["date"] < cutoff].copy()

    elos = compute_elo_ratings(pit_df)
    form_lookup = build_form_lookup(pit_df, as_of=cutoff_date)

    wc_pit = get_wc_matches(pit_df)
    recent_pit = pit_df[pit_df["date"] >= RECENT_ERA_START]

    # Logistic Regression — prior-WC-only training set (mirrors
    # cached_logreg_model()'s "wc_df" + "len(wc_df) < 20" guard).
    if wc_pit is not None and len(wc_pit) >= 20 and elos:
        logreg = LogisticMatchPredictor.train_from_history(wc_pit, elos, form_lookup)
    else:
        logreg = LogisticMatchPredictor()

    # Random Forest / XGBoost — recent-era training set (mirrors
    # cached_rf_model() / cached_xgb_model()'s "len(recent_df) < 200" guard).
    if len(recent_pit) >= 200 and elos:
        rf = RandomForestMatchPredictor.train_from_history(recent_pit, elos, form_lookup)
        xgb = XGBoostMatchPredictor.train_from_history(recent_pit, elos, form_lookup)
    else:
        rf = RandomForestMatchPredictor()
        xgb = XGBoostMatchPredictor()

    return {
        "elos": elos,
        "form_lookup": form_lookup,
        "logreg": logreg,
        "rf": rf,
        "xgb": xgb,
        "n_train_wc": 0 if wc_pit is None else len(wc_pit),
        "n_train_recent": len(recent_pit),
    }


def run_backtest(
    year: int,
    n_mc_sims: int = 5000,
    mc_seed: int = 42,
    full_df: pd.DataFrame | None = None,
) -> dict:
    """
    Run the point-in-time backtest for a single World Cup year (2018 or 2022).

    Returns a dict with keys:
      year, n_matches, n_train_wc, n_train_recent,
      rf_trained, xgb_trained, logreg_trained,
      models: {model_name: {"accuracy": float, "brier": float, "n": int}, ...},
      match_details: [ {Match, Score, Actual, "<model> pick": ...}, ... ]

    On failure (no network / no data), returns {"year": year, "error": "..."}.
    """
    if year not in TOURNAMENT_CUTOFFS:
        raise ValueError(f"No cutoff date configured for {year} (have: {list(TOURNAMENT_CUTOFFS)})")

    if full_df is None:
        full_df = fetch_results()
    if full_df is None:
        return {"year": year, "error": "Historical match data unavailable (no network access)."}

    wc_all = get_wc_matches(full_df)
    if wc_all is None:
        return {"year": year, "error": "Historical match data unavailable (no network access)."}

    target = wc_all[wc_all["date"].dt.year == year].copy()
    if target.empty:
        return {"year": year, "error": f"No {year} World Cup matches found in the dataset."}

    cutoff_date = TOURNAMENT_CUTOFFS[year]
    pit = _train_point_in_time(full_df, cutoff_date)
    elos, form_lookup = pit["elos"], pit["form_lookup"]
    logreg, rf, xgb = pit["logreg"], pit["rf"], pit["xgb"]

    if "neutral" not in target.columns:
        target = target.assign(neutral=True)

    mc_label = f"Monte Carlo ({n_mc_sims:,} sims)"
    model_names = [
        "Elo (W/D/L)",
        "Elo + Poisson",
        mc_label,
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    ]
    stats = {name: {"brier_sum": 0.0, "correct": 0, "n": 0} for name in model_names}
    match_details = []

    random.seed(mc_seed)

    cols = target[["home_team", "away_team", "home_score", "away_score", "neutral"]]
    for home, away, hs, as_, neutral in cols.itertuples(index=False, name=None):
        hs, as_ = int(hs), int(as_)
        neutral = bool(neutral)
        actual = _actual_outcome(hs, as_)

        eh = elos.get(home, 1500.0)
        ea = elos.get(away, 1500.0)
        eh_adj = eh if neutral else eh + HOME_ADVANTAGE_ELO

        fh = form_lookup.get(home, DEFAULT_FORM)
        fa = form_lookup.get(away, DEFAULT_FORM)

        ph_wdl = match_probabilities(eh_adj, ea)
        ph, pdraw, pa = match_result_probs_poisson(eh_adj, ea)

        ml_kwargs = dict(
            home_elo=eh, away_elo=ea, is_neutral=neutral,
            home_form=fh["form"], away_form=fa["form"],
            home_scored=fh["avg_scored"], away_scored=fa["avg_scored"],
            home_conceded=fh["avg_conceded"], away_conceded=fa["avg_conceded"],
        )

        preds = {
            "Elo (W/D/L)": ph_wdl,
            "Elo + Poisson": {"home_win": ph, "draw": pdraw, "away_win": pa},
            mc_label: _monte_carlo_match_probs(eh_adj, ea, n_sims=n_mc_sims),
            "Logistic Regression": logreg.predict(**ml_kwargs),
            "Random Forest": rf.predict(**ml_kwargs),
            "XGBoost": xgb.predict(**ml_kwargs),
        }

        row = {"Match": f"{home} vs {away}", "Score": f"{hs}-{as_}", "Actual": actual}
        for name, probs in preds.items():
            predicted = max(probs, key=probs.get)
            brier = _brier_score(probs, actual)
            s = stats[name]
            s["brier_sum"] += brier
            s["correct"] += int(predicted == actual)
            s["n"] += 1
            row[f"{name} pick"] = predicted
        match_details.append(row)

    models_summary = {
        name: {
            "accuracy": (s["correct"] / s["n"]) if s["n"] else 0.0,
            "brier": (s["brier_sum"] / s["n"]) if s["n"] else 0.0,
            "n": s["n"],
        }
        for name, s in stats.items()
    }

    return {
        "year": year,
        "n_matches": len(target),
        "n_train_wc": pit["n_train_wc"],
        "n_train_recent": pit["n_train_recent"],
        "rf_trained": rf.trained,
        "xgb_trained": xgb.trained,
        "logreg_trained": logreg.trained,
        "models": models_summary,
        "match_details": match_details,
    }


def run_full_backtest(years=(2018, 2022), n_mc_sims: int = 5000) -> dict:
    """Run run_backtest() for each year, sharing a single fetch_results() call."""
    full_df = fetch_results()
    if full_df is None:
        # Avoid each run_backtest() call re-attempting its own fetch_results()
        # (and re-printing the same network-failure message) when the shared
        # fetch already failed.
        return {
            year: {"year": year, "error": "Historical match data unavailable (no network access)."}
            for year in years
        }
    return {year: run_backtest(year, n_mc_sims=n_mc_sims, full_df=full_df) for year in years}
