"""
Real FIFA World Cup 2026 knockout-stage bracket structure (Matches 73-104).

Source: FIFA Tournament Regulations (Round of 32 matchup definitions and
Annex C third-place qualification rules) and the official bracket tree, as
published on Wikipedia's "2026 FIFA World Cup knockout stage" article
(verified June 2026). Match dates/venues are best-effort, cross-referenced
across multiple public schedules.

This module is pure data + pure functions — no randomness, no Streamlit,
no model imports — so it can be shared by both the Monte Carlo simulator
(models/monte_carlo.py) and the dashboard's bracket renderer
(dashboard/app.py).

-------------------------------------------------------------------------
"Best third-placed team" slots and Annex C
-------------------------------------------------------------------------
8 of the 16 Round-of-32 matches pair a group winner against the "best
third-placed team" from a published set of 5 groups. Which actual group's
3rd-placed team lands in which of these 8 slots depends on which 8 (of 12)
groups produce a qualifying 3rd-placed team — FIFA pre-computed this for
all C(12, 8) = 495 combinations in "Annex C" of the tournament regulations,
subject to one rule: a 3rd-placed team can never be drawn against a team
from its own group (which is why the published "best 3rd place" eligible
sets below never include the anchor group itself).

Rather than hardcode all 495 Annex C rows, `resolve_third_place_assignment`
computes a valid assignment algorithmically via backtracking over these same
eligibility sets. It always satisfies FIFA's two hard constraints (each
qualifying group used exactly once, never matched against its own anchor
group). Spot-checked against a published Annex C row (the all-{E,F,G,H,I,J,
K,L} qualifying case), this produces FIFA's exact published pairing — but
for combinations where multiple valid assignments exist, ours could in
principle differ from FIFA's specific published choice.
"""

from typing import Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Round of 32 (Matches 73-88)
#
# Slot codes:
#   "1X"            -> winner of Group X
#   "2X"            -> runner-up of Group X
#   "3:A,B,C,D,E"   -> best qualifying 3rd-placed team from groups A/B/C/D/E
# ---------------------------------------------------------------------------
R32_MATCHES: Dict[int, Tuple[str, str]] = {
    73: ("2A", "2B"),
    74: ("1E", "3:A,B,C,D,F"),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", "3:C,D,F,G,H"),
    78: ("2E", "2I"),
    79: ("1A", "3:C,E,F,H,I"),
    80: ("1L", "3:E,H,I,J,K"),
    81: ("1D", "3:B,E,F,I,J"),
    82: ("1G", "3:A,E,H,I,J"),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", "3:E,F,G,I,J"),
    86: ("1J", "2H"),
    87: ("1K", "3:D,E,I,J,L"),
    88: ("2D", "2G"),
}

# Eligible "best 3rd place" group sets, keyed by the R32 match that has a
# "3:..." slot. Derived directly from R32_MATCHES.
THIRD_PLACE_ELIGIBLE: Dict[int, frozenset] = {
    m: frozenset(slots[1][2:].split(","))
    for m, slots in R32_MATCHES.items()
    if slots[1].startswith("3:")
}
THIRD_PLACE_SLOTS: List[int] = list(THIRD_PLACE_ELIGIBLE.keys())  # [74,77,79,80,81,82,85,87]

# ---------------------------------------------------------------------------
# Bracket tree: Matches 89-102 (Round of 16, Quarter-Finals, Semi-Finals)
#
# Each entry maps a match number to the two earlier match numbers whose
# WINNERS meet in this match. Keys must stay in ascending order so that every
# match's inputs have already been resolved by the time it's processed.
# ---------------------------------------------------------------------------
BRACKET_TREE: Dict[int, Tuple[int, int]] = {
    89: (74, 77),
    90: (73, 75),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
    101: (97, 98),
    102: (99, 100),
}

FINAL_MATCH = 104        # Winner(101) vs Winner(102)
THIRD_PLACE_MATCH = 103  # Loser(101) vs Loser(102)

# ---------------------------------------------------------------------------
# Bracket layout: which 8 R32 matches feed each half of the draw, in the
# nesting order build_half()/render_half() expects (pairs of adjacent
# entries form one R16 match, pairs of those form one QF match, etc.)
#
#   Left half  -> Semi-Final 101 -> QF 97 = (R16 89, R16 90)
#                                 -> QF 98 = (R16 93, R16 94)
#   Right half -> Semi-Final 102 -> QF 99 = (R16 91, R16 92)
#                                 -> QF 100 = (R16 95, R16 96)
# ---------------------------------------------------------------------------
LEFT_R32_ORDER: List[int] = [74, 77, 73, 75, 83, 84, 81, 82]
RIGHT_R32_ORDER: List[int] = [76, 78, 79, 80, 86, 88, 85, 87]

# Round in which the LOSER of each match is eliminated (used by
# simulate_full_tournament_detailed to populate rounds_reached).
ROUND_OF_LOSER: Dict[int, str] = {}
for _m in R32_MATCHES:
    ROUND_OF_LOSER[_m] = "R32"
for _m in (89, 90, 91, 92, 93, 94, 95, 96):
    ROUND_OF_LOSER[_m] = "R16"
for _m in (97, 98, 99, 100):
    ROUND_OF_LOSER[_m] = "QF"
for _m in (101, 102):
    ROUND_OF_LOSER[_m] = "SF"
ROUND_OF_LOSER[FINAL_MATCH] = "Final"
del _m

# ---------------------------------------------------------------------------
# Match dates / venues (best-effort; cosmetic display only — does not affect
# simulation logic). City names match the conventions used in
# data/teams.py's HOST_NATIONS and data/fixtures.py.
# ---------------------------------------------------------------------------
MATCH_INFO: Dict[int, Dict[str, str]] = {
    73:  {"date": "2026-06-28", "time": "20:00", "venue": "SoFi Stadium", "city": "Los Angeles"},
    74:  {"date": "2026-06-29", "time": "18:00", "venue": "NRG Stadium", "city": "Houston"},
    75:  {"date": "2026-06-29", "time": "21:30", "venue": "Gillette Stadium", "city": "Boston"},
    76:  {"date": "2026-06-30", "time": "02:00", "venue": "Estadio BBVA", "city": "Monterrey"},
    77:  {"date": "2026-06-30", "time": "18:00", "venue": "AT&T Stadium", "city": "Dallas"},
    78:  {"date": "2026-06-30", "time": "22:00", "venue": "MetLife Stadium", "city": "New York/New Jersey"},
    79:  {"date": "2026-07-01", "time": "02:00", "venue": "Estadio Azteca", "city": "Mexico City"},
    80:  {"date": "2026-07-01", "time": "17:00", "venue": "Mercedes-Benz Stadium", "city": "Atlanta"},
    81:  {"date": "2026-07-01", "time": "21:00", "venue": "Lumen Field", "city": "Seattle"},
    82:  {"date": "2026-07-02", "time": "17:00", "venue": "Levi's Stadium", "city": "San Francisco Bay Area"},
    83:  {"date": "2026-07-02", "time": "17:00", "venue": "BC Place", "city": "Vancouver"},
    84:  {"date": "2026-07-02", "time": "21:00", "venue": "BMO Field", "city": "Toronto"},
    85:  {"date": "2026-07-03", "time": "17:00", "venue": "AT&T Stadium", "city": "Dallas"},
    86:  {"date": "2026-07-03", "time": "17:00", "venue": "SoFi Stadium", "city": "Los Angeles"},
    87:  {"date": "2026-07-03", "time": "21:00", "venue": "Hard Rock Stadium", "city": "Miami"},
    88:  {"date": "2026-07-03", "time": "21:00", "venue": "Arrowhead Stadium", "city": "Kansas City"},
    89:  {"date": "2026-07-04", "time": "", "venue": "Lincoln Financial Field", "city": "Philadelphia"},
    90:  {"date": "2026-07-04", "time": "", "venue": "NRG Stadium", "city": "Houston"},
    91:  {"date": "2026-07-05", "time": "", "venue": "MetLife Stadium", "city": "New York/New Jersey"},
    92:  {"date": "2026-07-05", "time": "", "venue": "Estadio Azteca", "city": "Mexico City"},
    93:  {"date": "2026-07-06", "time": "", "venue": "AT&T Stadium", "city": "Dallas"},
    94:  {"date": "2026-07-06", "time": "", "venue": "Lumen Field", "city": "Seattle"},
    95:  {"date": "2026-07-07", "time": "", "venue": "Mercedes-Benz Stadium", "city": "Atlanta"},
    96:  {"date": "2026-07-07", "time": "", "venue": "BC Place", "city": "Vancouver"},
    97:  {"date": "2026-07-09", "time": "", "venue": "Gillette Stadium", "city": "Boston"},
    98:  {"date": "2026-07-10", "time": "", "venue": "SoFi Stadium", "city": "Los Angeles"},
    99:  {"date": "2026-07-11", "time": "", "venue": "Hard Rock Stadium", "city": "Miami"},
    100: {"date": "2026-07-11", "time": "", "venue": "Arrowhead Stadium", "city": "Kansas City"},
    101: {"date": "2026-07-14", "time": "", "venue": "AT&T Stadium", "city": "Dallas"},
    102: {"date": "2026-07-15", "time": "", "venue": "Mercedes-Benz Stadium", "city": "Atlanta"},
    103: {"date": "2026-07-18", "time": "", "venue": "Hard Rock Stadium", "city": "Miami"},
    104: {"date": "2026-07-19", "time": "", "venue": "MetLife Stadium", "city": "New York/New Jersey"},
}


# ---------------------------------------------------------------------------
# Slot resolution
# ---------------------------------------------------------------------------

def describe_slot(slot_code: str) -> str:
    """Human-readable description of an R32 slot code, for tooltips."""
    kind = slot_code[0]
    if kind == "1":
        return f"Winner Group {slot_code[1]}"
    if kind == "2":
        return f"Runner-up Group {slot_code[1]}"
    if slot_code.startswith("3:"):
        groups = "/".join(slot_code[2:].split(","))
        return f"Best 3rd Place ({groups})"
    return slot_code


def resolve_third_place_assignment(qualifying_groups: List[str]) -> Dict[int, str]:
    """
    Given the 8 group letters whose 3rd-placed team qualifies for the
    knockout stage, return {match_number: group_letter} mapping each of the
    8 "best 3rd place" R32 slots (THIRD_PLACE_SLOTS) to the group whose
    3rd-placed team fills it.

    Uses deterministic backtracking over THIRD_PLACE_ELIGIBLE — see module
    docstring for the relationship to FIFA's published Annex C table.
    """
    groups = sorted(qualifying_groups)
    if len(groups) != 8 or len(set(groups)) != 8:
        raise ValueError(f"Expected 8 distinct qualifying third-place groups, got {groups}")

    assignment: Dict[int, str] = {}
    used: set = set()

    def backtrack(i: int) -> bool:
        if i == len(THIRD_PLACE_SLOTS):
            return True
        m = THIRD_PLACE_SLOTS[i]
        for g in groups:
            if g in used or g not in THIRD_PLACE_ELIGIBLE[m]:
                continue
            assignment[m] = g
            used.add(g)
            if backtrack(i + 1):
                return True
            used.discard(g)
            del assignment[m]
        return False

    if not backtrack(0):
        raise ValueError(f"No valid third-place assignment found for groups {groups}")

    return assignment


def _resolve_slot(
    slot_code: str,
    match_num: int,
    winners: Dict[str, str],
    runnersup: Dict[str, str],
    thirds: Dict[str, str],
    third_assignment: Dict[int, str],
) -> str:
    kind = slot_code[0]
    if kind == "1":
        return winners[slot_code[1]]
    if kind == "2":
        return runnersup[slot_code[1]]
    if slot_code.startswith("3:"):
        group = third_assignment[match_num]
        return thirds[group]
    raise ValueError(f"Unrecognised slot code: {slot_code!r}")


def resolve_r32_pairing(
    winners: Dict[str, str],
    runnersup: Dict[str, str],
    thirds: Dict[str, str],
    third_assignment: Dict[int, str],
) -> Dict[int, Tuple[str, str]]:
    """
    Resolve all 16 Round-of-32 matchups (matches 73-88) to actual team names.

    winners / runnersup / thirds: {group_letter: team_name}
    third_assignment: from resolve_third_place_assignment()
    """
    return {
        m: (
            _resolve_slot(slot_a, m, winners, runnersup, thirds, third_assignment),
            _resolve_slot(slot_b, m, winners, runnersup, thirds, third_assignment),
        )
        for m, (slot_a, slot_b) in R32_MATCHES.items()
    }


def walk_bracket(
    r32_pairing: Dict[int, Tuple[str, str]],
    decide_fn: Callable[[str, str, int], str],
) -> Dict[int, Tuple[str, str]]:
    """
    Walk the full knockout bracket (matches 73-104) given resolved R32
    pairings and a decision function.

    decide_fn(team_a, team_b, match_num) -> winning team name

    Returns {match_number: (winner, loser)} for all 32 matches 73-104,
    including the 3rd-place match (103) and Final (104).
    """
    results: Dict[int, Tuple[str, str]] = {}

    for m, (a, b) in r32_pairing.items():
        w = decide_fn(a, b, m)
        results[m] = (w, b if w == a else a)

    for m, (ma, mb) in BRACKET_TREE.items():
        a, b = results[ma][0], results[mb][0]
        w = decide_fn(a, b, m)
        results[m] = (w, b if w == a else a)

    a, b = results[101][0], results[102][0]
    w = decide_fn(a, b, FINAL_MATCH)
    results[FINAL_MATCH] = (w, b if w == a else a)

    la, lb = results[101][1], results[102][1]
    w3 = decide_fn(la, lb, THIRD_PLACE_MATCH)
    results[THIRD_PLACE_MATCH] = (w3, lb if w3 == la else la)

    return results
