"""
Bivariate Poisson model for scoreline probabilities.

Methodology:
  - Goals scored by each team are modelled as independent Poisson distributions
  - Expected goals (λ) derived from Elo ratings via models/elo.py
  - P(score = x-y) = Poisson(x; λ_home) * Poisson(y; λ_away)
  - Match result probabilities are computed analytically (not by simulation)

This approach is standard in sports analytics literature.
References:
  - Dixon & Coles (1997). Modelling Association Football Scores.
  - Maher (1982). Modelling Association Football Scores.
"""

import math
from functools import lru_cache
from typing import Dict, Tuple

from .elo import expected_goals_from_elo


# Maximum scoreline to consider in the exact distribution
MAX_GOALS = 8


@lru_cache(maxsize=512)
def _poisson_pmf(k: int, lam: float) -> float:
    """P(X=k) for X ~ Poisson(λ). Cached for speed."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def scoreline_distribution(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = MAX_GOALS,
) -> Dict[Tuple[int, int], float]:
    """
    Returns a dict mapping (home_goals, away_goals) -> probability.
    Probabilities sum to 1 (within floating-point tolerance).
    """
    dist: Dict[Tuple[int, int], float] = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            dist[(h, a)] = _poisson_pmf(h, lambda_home) * _poisson_pmf(a, lambda_away)
    # Renormalise to account for truncation at MAX_GOALS
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}


def match_result_probs_poisson(
    elo_home: float,
    elo_away: float,
) -> Tuple[float, float, float]:
    """
    Returns (p_home_win, p_draw, p_away_win) from the Poisson scoreline model.
    Uses Elo-derived expected goals.
    """
    lh, la = expected_goals_from_elo(elo_home, elo_away)
    dist = scoreline_distribution(round(lh, 4), round(la, 4))

    p_home = sum(p for (h, a), p in dist.items() if h > a)
    p_draw = sum(p for (h, a), p in dist.items() if h == a)
    p_away = sum(p for (h, a), p in dist.items() if a > h)
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def expected_goals(elo_home: float, elo_away: float) -> Tuple[float, float]:
    """Convenience wrapper."""
    return expected_goals_from_elo(elo_home, elo_away)


def most_likely_score(lambda_home: float, lambda_away: float) -> Tuple[int, int]:
    """Returns the most probable scoreline."""
    dist = scoreline_distribution(lambda_home, lambda_away)
    return max(dist, key=dist.__getitem__)


def top_scorelines(
    lambda_home: float,
    lambda_away: float,
    n: int = 5,
) -> list[Tuple[Tuple[int, int], float]]:
    """Returns top-n most probable scorelines with probabilities."""
    dist = scoreline_distribution(lambda_home, lambda_away)
    return sorted(dist.items(), key=lambda x: x[1], reverse=True)[:n]
