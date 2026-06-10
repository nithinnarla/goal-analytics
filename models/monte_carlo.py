"""
Monte Carlo tournament simulator for World Cup 2026.

Each simulation:
  1. Plays all 72 group-stage matches by sampling from Poisson scoreline distributions
  2. Ranks teams in each group (pts → GD → GF → H2H → tiebreak)
  3. Selects top-2 from each group + best 8 third-placed teams → 32 qualifiers
  4. Simulates knockout rounds (R32 → R16 → QF → SF → Final)
  5. In knockout matches, draws go to penalty shootout (50-50 Poisson-weighted)

Running 10,000 simulations gives stable win-probability estimates.
"""

import random
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from data.teams import GROUPS, TEAM_GROUP, get_elo
from data.fixtures import FIXTURES, get_group_fixtures, Fixture
from models.poisson import scoreline_distribution


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_score(lambda_home: float, lambda_away: float) -> Tuple[int, int]:
    """Draw a (home_goals, away_goals) from the Poisson joint distribution."""
    dist = scoreline_distribution(round(lambda_home, 3), round(lambda_away, 3))
    outcomes = list(dist.keys())
    probs = list(dist.values())
    chosen = random.choices(outcomes, weights=probs, k=1)[0]
    return chosen


def elo_to_lambda(elo_home: float, elo_away: float,
                  avg_goals: float = 1.25) -> Tuple[float, float]:
    diff = elo_home - elo_away
    scale = 10.0 ** (diff / 800.0)
    return avg_goals * scale, avg_goals / scale


# ---------------------------------------------------------------------------
# Group stage simulation
# ---------------------------------------------------------------------------

class TeamRecord:
    """Running record for one team in the group stage."""
    __slots__ = ["team", "pts", "gf", "ga", "gd", "played",
                 "wins", "draws", "losses"]

    def __init__(self, team: str):
        self.team = team
        self.pts = self.gf = self.ga = self.gd = 0
        self.played = self.wins = self.draws = self.losses = 0

    def add_result(self, gf: int, ga: int):
        self.played += 1
        self.gf += gf
        self.ga += ga
        self.gd += gf - ga
        if gf > ga:
            self.pts += 3
            self.wins += 1
        elif gf == ga:
            self.pts += 1
            self.draws += 1
        else:
            self.losses += 1

    def sort_key(self):
        # Higher is better: pts, GD, GF, then random tiebreak
        return (self.pts, self.gd, self.gf, random.random())


def simulate_group(group: str,
                   known_results: Optional[Dict[str, Tuple[int, int]]] = None
                   ) -> List[TeamRecord]:
    """
    Simulate one group. If known_results provides actual scores for some
    fixtures (match_id -> (hg, ag)), those are used verbatim; the rest
    are simulated.

    Returns list of TeamRecord sorted best-first.
    """
    known = known_results or {}
    records: Dict[str, TeamRecord] = {t: TeamRecord(t) for t in GROUPS[group]}

    for fix in get_group_fixtures(group):
        if fix.match_id in known:
            hg, ag = known[fix.match_id]
        elif fix.played:
            hg, ag = fix.home_goals, fix.away_goals  # type: ignore
        else:
            elo_h = get_elo(fix.home, fix.city)
            elo_a = get_elo(fix.away, fix.city)
            lh, la = elo_to_lambda(elo_h, elo_a)
            hg, ag = sample_score(lh, la)

        records[fix.home].add_result(hg, ag)
        records[fix.away].add_result(ag, hg)

    ranked = sorted(records.values(), key=lambda r: r.sort_key(), reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Third-place qualification
# ---------------------------------------------------------------------------

# Points-based ranking criterion for 8 best thirds (FIFA rules):
# pts → GD → GF → disciplinary → drawing
def thirds_sort_key(rec: TeamRecord):
    return (rec.pts, rec.gd, rec.gf, random.random())


def pick_best_thirds(thirds: List[TeamRecord], n: int = 8) -> List[str]:
    """Return names of the n best third-placed teams."""
    sorted_thirds = sorted(thirds, key=thirds_sort_key, reverse=True)
    return [r.team for r in sorted_thirds[:n]]


# ---------------------------------------------------------------------------
# Knockout simulation
# ---------------------------------------------------------------------------

def simulate_knockout_match(team_a: str, team_b: str) -> str:
    """
    Simulate a single knockout match.
    If 90-min result is a draw, simulate penalty shootout (approx. 50-50).
    """
    elo_a = get_elo(team_a)
    elo_b = get_elo(team_b)
    lh, la = elo_to_lambda(elo_a, elo_b)
    hg, ag = sample_score(lh, la)

    if hg > ag:
        return team_a
    if ag > hg:
        return team_b
    # Penalty shootout – slight Elo edge maintained
    p_a = 0.5 + (elo_a - elo_b) / 8000.0  # ~±5% edge at 400 Elo difference
    p_a = max(0.3, min(0.7, p_a))
    return team_a if random.random() < p_a else team_b


# ---------------------------------------------------------------------------
# WC 2026 knockout bracket (simplified linear bracket)
# ---------------------------------------------------------------------------
# The official bracket is path-based. We simplify: after round of 32,
# bracket is seeded by group position in fixed slots.

def simulate_full_tournament(
    known_results: Optional[Dict[str, Tuple[int, int]]] = None
) -> Dict[str, str]:
    """
    Simulate one full tournament. Returns dict of team -> furthest round reached.
    Possible values: "Group", "R32", "R16", "QF", "SF", "Final", "Winner"
    """
    known = known_results or {}
    rounds_reached: Dict[str, str] = {}

    # ── Group stage ─────────────────────────────────────────────────────────
    all_groups_ranked: Dict[str, List[TeamRecord]] = {}
    thirds: List[TeamRecord] = []

    for group in "ABCDEFGHIJKL":
        ranked = simulate_group(group, known)
        all_groups_ranked[group] = ranked
        # Mark group exit
        for rec in ranked:
            rounds_reached[rec.team] = "Group"
        thirds.append(ranked[2])

    # ── Qualification ────────────────────────────────────────────────────────
    qualifiers: List[str] = []
    for group in "ABCDEFGHIJKL":
        ranked = all_groups_ranked[group]
        qualifiers.append(ranked[0].team)   # 1st
        qualifiers.append(ranked[1].team)   # 2nd

    best_thirds = pick_best_thirds(thirds, 8)
    qualifiers.extend(best_thirds)
    # Mark all 32 qualifiers as R32
    for t in qualifiers:
        rounds_reached[t] = "R32"

    # ── Knockout rounds ───────────────────────────────────────────────────────
    round_names = ["R32", "R16", "QF", "SF", "Final"]
    remaining = list(qualifiers)  # 32 teams
    assert len(remaining) == 32, f"Expected 32 qualifiers, got {len(remaining)}"

    for rnd_name in round_names:
        next_round: List[str] = []
        random.shuffle(remaining)  # shuffle for bracket randomisation
        for i in range(0, len(remaining), 2):
            winner = simulate_knockout_match(remaining[i], remaining[i + 1])
            loser = remaining[i] if winner == remaining[i + 1] else remaining[i + 1]
            next_round.append(winner)
            # Loser is eliminated at this round
            rounds_reached[loser] = rnd_name

        remaining = next_round

    # ── Champion ─────────────────────────────────────────────────────────────
    assert len(remaining) == 1
    rounds_reached[remaining[0]] = "Winner"
    return rounds_reached


# ---------------------------------------------------------------------------
# Run N simulations and aggregate
# ---------------------------------------------------------------------------

ROUND_ORDER = ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]


def run_simulations(
    n: int = 10_000,
    known_results: Optional[Dict[str, Tuple[int, int]]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run n simulations and return per-team probabilities for each round.

    Returns:
        {team: {round: probability, ...}, ...}
    """
    if seed is not None:
        random.seed(seed)

    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _ in range(n):
        result = simulate_full_tournament(known_results)
        for team, best_round in result.items():
            # Count each round reached (includes reaching AND exceeding)
            idx = ROUND_ORDER.index(best_round)
            for r in ROUND_ORDER[: idx + 1]:
                counts[team][r] += 1

    # Convert to probabilities
    probs: Dict[str, Dict[str, float]] = {}
    for team in counts:
        probs[team] = {rnd: counts[team][rnd] / n for rnd in ROUND_ORDER}

    return probs


def win_probabilities(
    n: int = 10_000,
    known_results: Optional[Dict[str, Tuple[int, int]]] = None,
    seed: int = 42,
) -> List[Tuple[str, float]]:
    """
    Convenience: return sorted list of (team, win_probability) desc.
    """
    probs = run_simulations(n=n, known_results=known_results, seed=seed)
    result = [(team, probs[team].get("Winner", 0.0)) for team in probs]
    return sorted(result, key=lambda x: x[1], reverse=True)
