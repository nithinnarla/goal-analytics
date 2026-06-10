"""
Elo rating model for match outcome probabilities.

Methodology:
  - Expected score: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
  - Win/Draw/Loss probs derived from Elo difference + empirical WC draw rate
  - Home advantage: +100 Elo applied when team plays in home city

References:
  - Elo, A. (1978). The Rating of Chessplayers, Past and Present.
  - Hvattum & Arntzen (2010). Using ELO ratings for match result prediction.
  - eloratings.net football adaptation
"""

import math
from typing import Tuple


# World Cup average draw rate (~22% based on 1994-2022 data)
DRAW_RATE = 0.22

# Elo scale factor
ELO_SCALE = 400.0


def elo_win_probability(elo_a: float, elo_b: float) -> float:
    """P(A beats B) in a two-outcome model (excluding draws)."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / ELO_SCALE))


def match_probabilities(elo_home: float, elo_away: float) -> dict:
    """
    Returns {'home_win': p, 'draw': p, 'away_win': p} for a match.

    Draw allocation method:
      1. Compute raw win probability for home team (two-outcome Elo)
      2. Scale by (1 - DRAW_RATE) to compress win/loss prob
      3. Distribute DRAW_RATE as draw probability
      4. Adjust draw rate based on closeness of match
         (closer match = higher draw probability)

    Returns probabilities that sum to 1.0.
    """
    raw_p_home = elo_win_probability(elo_home, elo_away)

    # Adjust draw rate: close matches draw more often
    # max draw rate at raw_p_home = 0.5 (evenly matched)
    closeness = 1.0 - abs(raw_p_home - 0.5) * 2.0  # 0..1
    adjusted_draw = DRAW_RATE * (0.7 + 0.6 * closeness)
    adjusted_draw = min(adjusted_draw, 0.38)  # cap at 38%

    scale = 1.0 - adjusted_draw
    p_home = raw_p_home * scale
    p_away = (1.0 - raw_p_home) * scale
    p_draw = adjusted_draw

    # Normalise (floating point safety)
    total = p_home + p_draw + p_away
    return {
        "home_win": p_home / total,
        "draw":     p_draw / total,
        "away_win": p_away / total,
    }


def elo_to_str(elo: float) -> str:
    """Human-readable tier label."""
    if elo >= 2050:
        return "World Elite"
    if elo >= 1900:
        return "Top 10"
    if elo >= 1750:
        return "Contender"
    if elo >= 1600:
        return "Solid"
    if elo >= 1450:
        return "Qualifier"
    return "Underdog"


def expected_goals_from_elo(elo_home: float, elo_away: float,
                             avg_goals: float = 1.25) -> Tuple[float, float]:
    """
    Derive expected goals (λ_home, λ_away) from Elo difference.

    Uses the log-goals scaling from the Poisson model:
      λ_home = avg_goals * 10^(diff/800)
      λ_away = avg_goals * 10^(-diff/800)

    avg_goals = 1.25 (WC average goals per team per game, 1994-2022 ≈ 1.25)
    """
    diff = elo_home - elo_away
    scale = 10.0 ** (diff / 800.0)
    lambda_home = avg_goals * scale
    lambda_away = avg_goals / scale
    return lambda_home, lambda_away
