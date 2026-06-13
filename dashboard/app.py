"""
GoalAnalytics — World Cup 2026 Prediction Dashboard
=====================================================
Streamlit app with 4 tabs:
  1. 🏆 Win Probabilities  — who wins the tournament?
  2. 📊 Group Predictions  — predicted standings + match results
  3. 📍 Match Predictor    — enter any two teams, get scoreline odds
  4. 📈 Live Tracker       — enter actual results, track model accuracy

Run locally:  streamlit run dashboard/app.py
"""

import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import json
from collections import defaultdict

from data.teams import TEAMS, GROUPS, TEAM_GROUP, get_elo
from data.fixtures import FIXTURES, get_group_fixtures, get_team_fixtures, FIXTURE_BY_ID
from models.elo import match_probabilities
from models.poisson import (
    match_result_probs_poisson,
    expected_goals,
    top_scorelines,
    scoreline_distribution,
)
from models.monte_carlo import (
    run_simulations,
    win_probabilities,
    simulate_group,
    ROUND_ORDER,
)
from data.knockout_fixtures import (
    LEFT_R32_ORDER,
    RIGHT_R32_ORDER,
    resolve_r32_pairing,
    resolve_third_place_assignment,
)
from data.fifa_rankings import get_fifa_rankings_table

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Goal Analytics | WC 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling — FIFA World Cup 2026 Football Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ═══════════════════════════════════════════
   STADIUM BACKGROUND (animated)
   Note: Streamlit strips <video>/<source> tags from
   st.markdown, so the "motion picture" wallpaper is
   a packed-stadium photo (Allianz Arena, Munich,
   sold-out matchday crowd) with a slow Ken-Burns
   pan/zoom animation to give the "running"
   motion-picture feel.
═══════════════════════════════════════════ */
.bg-stadium {
    position: fixed;
    top: -3%; left: -3%;
    width: 106vw; height: 106vh;
    z-index: -10;
    background-image: url('https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Fu%C3%9Fball-Bundesliga_2021-2022_-_FC_Bayern_M%C3%BCnchen_vs_Borussia_Dortmund_001.jpg/1920px-Fu%C3%9Fball-Bundesliga_2021-2022_-_FC_Bayern_M%C3%BCnchen_vs_Borussia_Dortmund_001.jpg');
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    opacity: 0.55;
    filter: brightness(0.75) saturate(1.35);
    animation: stadiumPan 45s ease-in-out infinite alternate;
}

@keyframes stadiumPan {
    0%   { transform: scale(1.0)  translate(0%, 0%); }
    50%  { transform: scale(1.09) translate(-1.5%, -1%); }
    100% { transform: scale(1.05) translate(1.5%, 0.8%); }
}

/* ═══════════════════════════════════════════
   Make Streamlit's own containers transparent so
   the fixed-position background layers above
   (.bg-stadium, .bg-overlay, .spotlight, etc.)
   are actually visible instead of being painted
   over by Streamlit's opaque app background.
═══════════════════════════════════════════ */
html, body {
    background-color: #05070d;
}
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"],
[data-testid="stBottomBlockContainer"],
.stApp {
    background: transparent !important;
}

.bg-overlay {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: -9;
    background: linear-gradient(
        160deg,
        rgba(2,8,16,0.32) 0%,
        rgba(4,15,8,0.26) 50%,
        rgba(8,2,20,0.32) 100%
    );
}

.spotlight {
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    z-index: -9;
    background: radial-gradient(
        ellipse 55% 35% at 30% 30%,
        rgba(255,255,255,0.025) 0%,
        transparent 70%
    );
    animation: spotMove 14s ease-in-out infinite;
}

@keyframes spotMove {
    0%   { transform: translate(0%, 0%); }
    33%  { transform: translate(40%, 10%); }
    66%  { transform: translate(15%, 25%); }
    100% { transform: translate(0%, 0%); }
}

.pitch-lines {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    z-index: -8;
    background:
        repeating-linear-gradient(
            90deg,
            transparent 0px, transparent 79px,
            rgba(255,255,255,0.012) 79px, rgba(255,255,255,0.012) 80px
        ),
        repeating-linear-gradient(
            0deg,
            transparent 0px, transparent 79px,
            rgba(255,255,255,0.012) 79px, rgba(255,255,255,0.012) 80px
        );
    animation: pitchSlide 50s linear infinite;
}

@keyframes pitchSlide {
    0%   { background-position: 0px 0px; }
    100% { background-position: 80px 80px; }
}

/* ═══════════════════════════════════════════
   MAIN HEADER
═══════════════════════════════════════════ */
.main-header {
    background: linear-gradient(135deg, #080f1e 0%, #0f1f10 50%, #0c0820 100%);
    border: 1px solid rgba(200, 16, 46, 0.25);
    border-top: 4px solid #c8102e;
    border-bottom: 2px solid rgba(255, 215, 0, 0.2);
    padding: 2.5rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow:
        0 8px 48px rgba(0,0,0,0.7),
        0 0 100px rgba(200,16,46,0.07),
        inset 0 1px 0 rgba(255,255,255,0.04);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -30%; left: -20%;
    width: 60%; height: 160%;
    background: radial-gradient(ellipse, rgba(200,16,46,0.07) 0%, transparent 70%);
    animation: hdrGlow 5s ease-in-out infinite;
}

.main-header::after {
    content: '';
    position: absolute;
    top: -30%; right: -20%;
    width: 60%; height: 160%;
    background: radial-gradient(ellipse, rgba(0,70,200,0.06) 0%, transparent 70%);
    animation: hdrGlow 5s ease-in-out infinite reverse;
}

@keyframes hdrGlow {
    0%, 100% { opacity: 0.5; transform: scale(1); }
    50%       { opacity: 1;   transform: scale(1.12); }
}

.wc-logo {
    height: 110px;
    margin-bottom: 0.8rem;
    position: relative;
    z-index: 1;
    filter: drop-shadow(0 4px 20px rgba(200,16,46,0.5));
}

.main-header h1 {
    color: #ffffff;
    font-size: 2.8rem;
    margin: 0;
    font-weight: 900;
    text-shadow: 0 0 40px rgba(200,16,46,0.45), 0 2px 6px rgba(0,0,0,0.9);
    position: relative;
    z-index: 1;
    letter-spacing: -1px;
}

.hdr-sub {
    color: rgba(255, 215, 0, 0.9);
    margin: 0.5rem 0 0.2rem;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    position: relative;
    z-index: 1;
}

.hdr-model {
    color: rgba(255,255,255,0.38);
    font-size: 0.78rem;
    margin-top: 0.3rem;
    position: relative;
    z-index: 1;
    letter-spacing: 0.5px;
}

/* ═══════════════════════════════════════════
   TABS
═══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: rgba(255,255,255,0.55);
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0.5rem 1.1rem;
    border: none;
}

.stTabs [data-baseweb="tab"]:hover {
    color: rgba(255,255,255,0.9);
    background: rgba(255,255,255,0.06);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #c8102e 0%, #8b0c20 100%) !important;
    color: white !important;
    box-shadow: 0 2px 16px rgba(200,16,46,0.5);
}

/* ═══════════════════════════════════════════
   METRIC CARDS
═══════════════════════════════════════════ */
[data-testid="stMetricValue"] {
    color: #ffd700 !important;
    font-size: 1.5rem !important;
    font-weight: 900 !important;
}

[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.65) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ═══════════════════════════════════════════
   HEADERS & TEXT
═══════════════════════════════════════════ */
h2, h3 { color: #ffffff !important; }

h3 {
    border-left: 3px solid #c8102e;
    padding-left: 0.7rem;
    margin-top: 1.5rem !important;
}

/* ═══════════════════════════════════════════
   GROUP HEADER
═══════════════════════════════════════════ */
.group-header {
    background: linear-gradient(135deg, #c8102e 0%, #8b0c20 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: 800;
    font-size: 0.95rem;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
    border-left: 4px solid #ffd700;
    box-shadow: 0 2px 16px rgba(200,16,46,0.35);
}

/* ═══════════════════════════════════════════
   MATCH CARD
═══════════════════════════════════════════ */
.match-card {
    border-left: 4px solid #c8102e;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    background: rgba(255,255,255,0.04);
    border-radius: 0 10px 10px 0;
}

/* ═══════════════════════════════════════════
   MISC
═══════════════════════════════════════════ */
.accuracy-good { color: #4caf50; font-weight: bold; }
.accuracy-bad  { color: #f44336; font-weight: bold; }
.accuracy-ok   { color: #ff9800; font-weight: bold; }

.winner-badge {
    background: linear-gradient(135deg, #ffd700 0%, #ff8c00 100%);
    color: #0a0f1e;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: bold;
}

.team-prob { font-size: 1.4rem; font-weight: bold; color: #ffd700; }

hr { border-color: rgba(255,255,255,0.06) !important; }

/* ═══════════════════════════════════════════
   FOOTER
═══════════════════════════════════════════ */
.footer-bar {
    text-align: center;
    color: rgba(255,255,255,0.3);
    font-size: 0.8rem;
    padding: 1rem 0 0.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}

.footer-bar a {
    color: #ffd700 !important;
    text-decoration: none;
}

/* ═══════════════════════════════════════════
   MARCH-MADNESS STYLE TOURNAMENT BRACKET
═══════════════════════════════════════════ */
.mm-bracket {
    display: flex;
    align-items: center;
    gap: 0;
    overflow-x: auto;
    padding: 1rem 0.5rem 1.75rem;
    min-width: max-content;
}
.mm-half {
    position: relative;
    width: 582px;
    height: 368px;
    flex-shrink: 0;
}
.mm-half.mm-mirror { transform: scaleX(-1); }
.mm-half.mm-mirror .mm-box { transform: scaleX(-1); }
.mm-svg {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    overflow: visible;
}
.mm-box {
    position: absolute;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 6px;
    overflow: hidden;
}
.mm-box2 { border-left: 3px solid #c8102e; }
.mm-box1 {
    border-left: 3px solid #ffd700;
    background: rgba(255,215,0,0.07);
    display: flex;
    align-items: center;
}
.mm-row, .mm-row-lg {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 6px;
    font-size: 0.62rem;
    line-height: 18px;
    height: 18px;
    color: rgba(255,255,255,0.8);
    white-space: nowrap;
    overflow: hidden;
}
.mm-row + .mm-row { border-top: 1px solid rgba(255,255,255,0.06); }
.mm-row-lg {
    font-size: 0.7rem;
    font-weight: 800;
    color: #ffd700;
    height: 100%;
    width: 100%;
}
.mm-flag { margin-right: 5px; }
.mm-name { flex: 1; overflow: hidden; text-overflow: ellipsis; }
.mm-pct {
    color: #ffd700;
    font-weight: 700;
    margin-left: 6px;
    opacity: 0.85;
    font-size: 0.6rem;
}
.mm-row-lg .mm-pct { color: rgba(255,255,255,0.75); }
.mm-row.mm-winner, .mm-row-lg.mm-winner {
    background: rgba(255,215,0,0.12);
}
.mm-row.mm-winner .mm-name, .mm-row-lg.mm-winner .mm-name {
    color: #ffd700;
    font-weight: 800;
}
.mm-row.mm-winner .mm-pct, .mm-row-lg.mm-winner .mm-pct { color: #ffd700; opacity: 1; }
.mm-row:not(.mm-winner) .mm-name, .mm-row:not(.mm-winner) .mm-flag {
    opacity: 0.5;
}
.mm-center {
    flex-shrink: 0;
    width: 168px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 0 12px;
}
.mm-final-label {
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 4px;
    color: #c8102e;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.mm-final-box {
    width: 100%;
    height: 24px;
    border: 1px solid rgba(255,255,255,0.09);
    border-left: 3px solid #ffd700;
    border-radius: 6px;
    background: rgba(255,215,0,0.07);
    display: flex;
    align-items: center;
}
.mm-champion {
    margin-top: 8px;
    text-align: center;
    font-size: 0.85rem;
    font-weight: 800;
    color: #ffd700;
    background: rgba(255,215,0,0.12);
    border: 1px solid rgba(255,215,0,0.4);
    border-radius: 8px;
    padding: 12px 8px;
    width: 100%;
}
.mm-champ-pct {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.75);
    font-weight: 600;
    margin-top: 4px;
}
.mm-round-labels {
    display: flex;
    width: 582px;
    flex-shrink: 0;
}
.mm-round-labels span {
    width: 132px;
    margin-right: 18px;
    text-align: center;
    font-size: 0.6rem;
    font-weight: 800;
    color: #ffd700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.mm-round-labels.mm-mirror { flex-direction: row-reverse; }
.mm-round-labels.mm-mirror span { margin-right: 0; margin-left: 18px; }
.mm-round-labels span:last-child { margin-right: 0; }
.mm-round-labels.mm-mirror span:last-child { margin-left: 0; }
.mm-labels-row {
    display: flex;
    align-items: center;
    overflow-x: auto;
    min-width: max-content;
}
.mm-labels-row .mm-center { visibility: hidden; }
.group-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
    margin-bottom: 1.2rem;
}
.group-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 3px solid #c8102e;
    border-radius: 8px;
    padding: 8px 10px;
}
.group-card-title {
    font-size: 0.6rem;
    font-weight: 800;
    color: #ffd700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 5px;
}
.group-card-team {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.8);
    padding: 2px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.group-card-team.first { color: #ffd700; font-weight: 700; }
.group-card-team.second { color: rgba(255,255,255,0.65); }
.group-card-team.third-qualified { color: #7fdbff; font-weight: 600; }
.group-card-team.eliminated { color: rgba(255,255,255,0.3); font-size: 0.65rem; }

/* Prediction Engine — model cards & Final cross-check (Tab 5) */
.model-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 1rem;
}
.model-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 3px solid #7fdbff;
    border-radius: 8px;
    padding: 10px 12px;
}
.model-card-title {
    font-size: 0.78rem;
    font-weight: 800;
    color: #7fdbff;
    margin-bottom: 4px;
}
.model-card-body {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.75);
    line-height: 1.45;
}
.model-card-status {
    display: inline-block;
    margin-top: 6px;
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(127,219,255,0.15);
    color: #7fdbff;
}
.crosscheck-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin: 0.6rem 0 1.2rem;
}
.crosscheck-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.crosscheck-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: rgba(255,255,255,0.55);
    margin-bottom: 4px;
}
.crosscheck-value {
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffd700;
}
.crosscheck-sub {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.6);
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Animated background layers
# ---------------------------------------------------------------------------
st.markdown("""
<div class="bg-stadium"></div>
<div class="bg-overlay"></div>
<div class="spotlight"></div>
<div class="pitch-lines"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <img class="wc-logo"
         src="https://upload.wikimedia.org/wikipedia/en/thumb/1/17/2026_FIFA_World_Cup_emblem.svg/250px-2026_FIFA_World_Cup_emblem.svg.png"
         alt="FIFA World Cup 2026"
         onerror="this.style.display='none'"/>
    <h1>⚽ Goal Analytics</h1>
    <div class="hdr-sub">FIFA World Cup 2026 · Prediction Engine</div>
    <div class="hdr-model">Elo · Bivariate Poisson · Logistic Regression · 10,000 Monte Carlo Simulations · 48 Teams · 104 Matches</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state: actual results input by user (persisted to disk so they
# survive page reloads, new browser sessions, and app restarts)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_FILE = os.path.join(_PROJECT_ROOT, "data", "actual_results.json")


def _load_actual_results():
    try:
        with open(_RESULTS_FILE) as f:
            raw = json.load(f)
        return {k: tuple(v) for k, v in raw.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_actual_results(results):
    os.makedirs(os.path.dirname(_RESULTS_FILE), exist_ok=True)
    with open(_RESULTS_FILE, "w") as f:
        json.dump({k: list(v) for k, v in results.items()}, f)


if "actual_results" not in st.session_state:
    st.session_state.actual_results = _load_actual_results()  # match_id -> (hg, ag)

# ---------------------------------------------------------------------------
# Cache heavy computation
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running 10,000 simulations…", ttl=3600)
def cached_win_probs(results_key: str):
    """Results key is a JSON string of known_results for cache invalidation."""
    known = json.loads(results_key) if results_key else {}
    # Convert string keys back to original form
    known_parsed = {k: tuple(v) for k, v in known.items()}
    return run_simulations(n=10_000, known_results=known_parsed, seed=42)


@st.cache_data(show_spinner="Simulating group standings…", ttl=3600)
def cached_group_sim(group: str, results_key: str):
    known = json.loads(results_key) if results_key else {}
    known_parsed = {k: tuple(v) for k, v in known.items()}
    records = []
    for _ in range(5000):
        ranked = simulate_group(group, known_parsed)
        records.append([r.team for r in ranked])
    # Average finishing position
    pos_counts = defaultdict(lambda: defaultdict(int))
    for sim in records:
        for pos, team in enumerate(sim):
            pos_counts[team][pos] += 1
    result = {}
    for team, pos_dict in pos_counts.items():
        avg_pos = sum(p * c for p, c in pos_dict.items()) / 5000
        result[team] = {
            "avg_pos": avg_pos,
            "p1st": pos_dict[0] / 5000,
            "p2nd": pos_dict[1] / 5000,
            "p3rd": pos_dict[2] / 5000,
            "p4th": pos_dict[3] / 5000,
        }
    return result


@st.cache_data(show_spinner="Fetching live FIFA World Rankings…", ttl=3600)
def cached_fifa_rankings():
    """
    Live FIFA World Ranking comparison table (informational only — does
    NOT feed into Elo, the Poisson model, or the Monte Carlo simulation).
    Falls back to {} if the live source is unreachable; callers should
    treat that as "show static model ranks only".
    """
    try:
        return get_fifa_rankings_table()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Shared group standings + bracket qualifiers
#
# Tab 2 ranks each group by avg finishing position (cached_group_sim, 5,000
# group-only sims). Tab 5's bracket needs the same top-2-per-group order PLUS
# the 8 best third-placed teams (the "wildcard" route), picked using the
# Round-of-32 probabilities from the full 10,000-sim Monte Carlo run
# (cached_win_probs). Computing both here from the SAME inputs guarantees
# Tab 2 and Tab 5 always show the same qualifiers for every group.
# ---------------------------------------------------------------------------
def get_bracket_qualifiers(rkey: str, all_probs: dict):
    """
    Returns
    -------
    standings       : dict[group] -> list of 4 teams, ordered best-to-worst
                       (identical ordering to Tab 2's group tables)
    best_thirds_set : set of the 8 third-placed teams that advance as
                       wildcards, chosen by Round-of-32 probability
    teams32         : list of the 32 knockout-stage qualifiers (top-2 per
                       group + best 8 thirds), seeded by Round-of-32
                       probability for the bracket
    """
    standings = {}
    third_place_teams = []

    for grp in "ABCDEFGHIJKL":
        sims = cached_group_sim(grp, rkey)
        ordered = sorted(GROUPS[grp], key=lambda t: sims[t]["avg_pos"])
        standings[grp] = ordered
        third_place_teams.append(ordered[2])

    best_thirds = sorted(
        third_place_teams,
        key=lambda t: all_probs.get(t, {}).get("R32", 0),
        reverse=True,
    )[:8]
    best_thirds_set = set(best_thirds)

    auto_qualifiers = [t for grp in "ABCDEFGHIJKL" for t in standings[grp][:2]]
    teams32 = sorted(
        auto_qualifiers + best_thirds,
        key=lambda t: all_probs.get(t, {}).get("R32", 0),
        reverse=True,
    )

    return standings, best_thirds_set, teams32


# ---------------------------------------------------------------------------
# Logistic Regression model (cached resource — trained once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training logistic regression on WC history…")
def cached_logreg_model():
    """
    Train the multinomial logistic-regression match predictor on historical
    World Cup results. Falls back to an untrained model (which uses a pure
    Elo-based estimate) if historical data can't be fetched — e.g. no network
    access on Streamlit Cloud.
    """
    from models.logistic import LogisticMatchPredictor
    try:
        from data.historical import fetch_results, get_wc_matches, compute_elo_ratings, build_form_lookup
        df = fetch_results()
        if df is None:
            return LogisticMatchPredictor(), False
        wc_df = get_wc_matches(df)
        elos = compute_elo_ratings(df)
        form_lookup = build_form_lookup(df)
        if wc_df is None or len(wc_df) < 20 or not elos:
            return LogisticMatchPredictor(), False
        model = LogisticMatchPredictor.train_from_history(wc_df, elos, form_lookup)
        return model, model.trained
    except Exception:
        return LogisticMatchPredictor(), False


# ---------------------------------------------------------------------------
# Random Forest & XGBoost models (cached resources — trained once per session)
#
# Unlike the logistic model (trained on ~900 WC-only matches), these two
# train on the "recent era" of ALL international results (every
# competition, not just World Cups) — far more rows, which non-linear
# tree ensembles can make better use of. RECENT_ERA_START bounds that
# window so training stays on data that reflects the modern game.
# ---------------------------------------------------------------------------
RECENT_ERA_START = "2010-01-01"


@st.cache_resource(show_spinner="Training random forest on recent-era results…")
def cached_rf_model():
    """
    Train the random-forest match predictor on international results since
    RECENT_ERA_START. Falls back to an untrained model (pure Elo-based
    estimate) if historical data can't be fetched.
    """
    from models.random_forest import RandomForestMatchPredictor
    try:
        from data.historical import fetch_results, compute_elo_ratings, build_form_lookup
        df = fetch_results()
        if df is None:
            return RandomForestMatchPredictor(), False
        recent_df = df[df["date"] >= RECENT_ERA_START]
        elos = compute_elo_ratings(df)
        form_lookup = build_form_lookup(df)
        if recent_df is None or len(recent_df) < 200 or not elos:
            return RandomForestMatchPredictor(), False
        model = RandomForestMatchPredictor.train_from_history(recent_df, elos, form_lookup)
        return model, model.trained
    except Exception:
        return RandomForestMatchPredictor(), False


@st.cache_resource(show_spinner="Training XGBoost on recent-era results…")
def cached_xgb_model():
    """
    Train the XGBoost match predictor on international results since
    RECENT_ERA_START. Falls back to an untrained model (pure Elo-based
    estimate) if historical data or the xgboost package isn't available.
    """
    from models.xgboost_model import XGBoostMatchPredictor
    try:
        from data.historical import fetch_results, compute_elo_ratings, build_form_lookup
        df = fetch_results()
        if df is None:
            return XGBoostMatchPredictor(), False
        recent_df = df[df["date"] >= RECENT_ERA_START]
        elos = compute_elo_ratings(df)
        form_lookup = build_form_lookup(df)
        if recent_df is None or len(recent_df) < 200 or not elos:
            return XGBoostMatchPredictor(), False
        model = XGBoostMatchPredictor.train_from_history(recent_df, elos, form_lookup)
        return model, model.trained
    except Exception:
        return XGBoostMatchPredictor(), False


# ---------------------------------------------------------------------------
# Historical model backtest (2018 & 2022 World Cups)
#
# Trains separate, point-in-time copies of every model (using only data
# available before each tournament kicked off) and scores them against that
# tournament's actual results. Cached for 24h since it retrains RF/XGBoost
# twice; gated behind a button in the UI so it doesn't run on every page load.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running point-in-time backtest on 2018 & 2022 World Cups…", ttl=86400)
def cached_backtest():
    from models.backtest import run_full_backtest
    try:
        return run_full_backtest(years=(2018, 2022), n_mc_sims=5000)
    except Exception as e:
        return {
            2018: {"year": 2018, "error": str(e)},
            2022: {"year": 2022, "error": str(e)},
        }


# ---------------------------------------------------------------------------
# Build results key for cache
# ---------------------------------------------------------------------------
def results_key() -> str:
    return json.dumps(st.session_state.actual_results) if st.session_state.actual_results else ""


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Win Probabilities",
    "📊 Group Predictions",
    "📍 Match Predictor",
    "📈 Live Tracker",
    "🗺️ Bracket",
    "🧪 Model Backtest",
])

# ============================================================================
# TAB 1 — Win Probabilities
# ============================================================================
with tab1:
    st.subheader("Tournament Win Probability")
    st.caption("Based on 10,000 Monte Carlo simulations using Elo + Poisson model")

    rkey = results_key()
    all_probs = cached_win_probs(rkey)

    # Sort by win probability
    sorted_teams = sorted(
        all_probs.keys(),
        key=lambda t: all_probs[t].get("Winner", 0),
        reverse=True
    )

    # Top 10 bar chart
    top10 = sorted_teams[:10]
    chart_data = pd.DataFrame({
        "Team": [f"{TEAMS[t]['flag']} {t}" for t in top10],
        "Win %": [round(all_probs[t].get("Winner", 0) * 100, 1) for t in top10],
    })

    st.bar_chart(chart_data.set_index("Team"), height=400, use_container_width=True)

    # Full probability table
    st.markdown("### Full 48-Team Probability Table")
    rows = []
    for rank, team in enumerate(sorted_teams, 1):
        tp = all_probs[team]
        rows.append({
            "Rank": rank,
            "Team": f"{TEAMS[team]['flag']} {team}",
            "Group": TEAM_GROUP[team],
            "Elo": TEAMS[team]["elo"],
            "R32 %": f"{tp.get('R32', 0)*100:.1f}",
            "R16 %": f"{tp.get('R16', 0)*100:.1f}",
            "QF %": f"{tp.get('QF', 0)*100:.1f}",
            "SF %": f"{tp.get('SF', 0)*100:.1f}",
            "Final %": f"{tp.get('Final', 0)*100:.1f}",
            "Win %": f"{tp.get('Winner', 0)*100:.2f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------
    # Live FIFA World Ranking (vs. model) — informational comparison only
    # -----------------------------------------------------------------
    with st.expander("🌍 Live FIFA World Ranking vs. model rank"):
        st.caption(
            "Live ranks are pulled from the current published FIFA World Ranking. "
            "This panel is for comparison only — it does **not** feed into the "
            "Elo ratings, Poisson model, or Monte Carlo simulation above."
        )
        fifa_table = cached_fifa_rankings()
        if not fifa_table or not any(v["live_rank"] is not None for v in fifa_table.values()):
            st.warning(
                "Live FIFA rankings aren't reachable right now — showing this "
                "app's static model ranks only (live source may be temporarily "
                "unavailable, e.g. on a network-restricted host)."
            )
        else:
            fifa_rows = []
            for team, info in fifa_table.items():
                delta = info["delta"]
                if delta is None:
                    delta_str = "—"
                elif delta > 0:
                    delta_str = f"▲ live {delta} better"
                elif delta < 0:
                    delta_str = f"▼ live {abs(delta)} worse"
                else:
                    delta_str = "= same"
                fifa_rows.append({
                    "Team": f"{TEAMS[team]['flag']} {team}",
                    "Model Rank": info["static_rank"],
                    "Live FIFA Rank": info["live_rank"] if info["live_rank"] is not None else "—",
                    "Live Points": f"{info['live_points']:.0f}" if info["live_points"] is not None else "—",
                    "Δ Model vs. Live": delta_str,
                })
            sort_key = lambda r: r["Live FIFA Rank"] if isinstance(r["Live FIFA Rank"], int) else r["Model Rank"]
            fifa_rows.sort(key=sort_key)
            fifa_df = pd.DataFrame(fifa_rows)
            st.dataframe(fifa_df, use_container_width=True, hide_index=True)
            st.caption(
                "▲ = live FIFA ranking is BETTER (lower number) than this app's static "
                "model rank, i.e. the model may be underrating that team. "
                "▼ = live ranking is worse than the model rank. "
                "Source: FIFA World Ranking (via whereig.com), cached up to 24h."
            )

    st.markdown("---")
    st.info(
        "**Model notes:** Win probabilities are from 10,000 simulations. "
        "Each match uses a bivariate Poisson model calibrated from Elo ratings. "
        "Home advantage (+100 Elo) applied for Mexico, USA, and Canada. "
        "Knockout draws resolved by simulated penalty shootout."
    )

# ============================================================================
# TAB 2 — Group Predictions
# ============================================================================
with tab2:
    st.subheader("Group Stage Predictions")

    rkey = results_key()
    group_cols = st.columns(3)

    for g_idx, group in enumerate("ABCDEFGHIJKL"):
        col = group_cols[g_idx % 3]
        with col:
            st.markdown(f"<div class='group-header'>GROUP {group}</div>",
                        unsafe_allow_html=True)
            teams = GROUPS[group]
            sims = cached_group_sim(group, rkey)

            group_df = pd.DataFrame([
                {
                    "Team": f"{TEAMS[t]['flag']} {t}",
                    "Elo": TEAMS[t]["elo"],
                    "P(1st)": f"{sims[t]['p1st']*100:.0f}%",
                    "P(2nd)": f"{sims[t]['p2nd']*100:.0f}%",
                    "P(3rd)": f"{sims[t]['p3rd']*100:.0f}%",
                }
                for t in teams
            ])
            # Sort by avg position
            group_df["_sort"] = [sims[t]["avg_pos"] for t in teams]
            group_df = group_df.sort_values("_sort").drop(columns=["_sort"])
            st.dataframe(group_df, hide_index=True, use_container_width=True)

            # Show group fixtures
            with st.expander(f"Group {group} Fixtures"):
                for fix in get_group_fixtures(group):
                    elo_h = get_elo(fix.home, fix.city)
                    elo_a = get_elo(fix.away, fix.city)
                    lh, la = expected_goals(elo_h, elo_a)
                    ph, pd_, pa = match_result_probs_poisson(elo_h, elo_a)

                    actual = st.session_state.actual_results.get(fix.match_id)
                    if actual:
                        result_str = f"✅ **{actual[0]}-{actual[1]}**"
                    elif fix.played:
                        result_str = f"✅ **{fix.home_goals}-{fix.away_goals}**"
                    else:
                        result_str = f"~{lh:.1f}-{la:.1f}"

                    st.markdown(
                        f"**{TEAMS[fix.home]['flag']} {fix.home}** vs "
                        f"**{TEAMS[fix.away]['flag']} {fix.away}**  \n"
                        f"{fix.date} · {result_str}  \n"
                        f"H {ph*100:.0f}% · D {pd_*100:.0f}% · A {pa*100:.0f}%",
                    )
            st.markdown("")

# ============================================================================
# TAB 3 — Match Predictor
# ============================================================================
with tab3:
    st.subheader("Head-to-Head Match Predictor")
    st.caption("Select any two teams to see predicted outcome probabilities and top scorelines")

    all_team_names = sorted(TEAMS.keys())
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        home_team = st.selectbox("Team A", all_team_names,
                                  index=all_team_names.index("Argentina"))
    with c2:
        st.markdown("<br><div style='text-align:center; font-size:1.5rem'>⚔️</div>",
                    unsafe_allow_html=True)
    with c3:
        away_team = st.selectbox("Team B", all_team_names,
                                  index=all_team_names.index("France"))

    WC2026_VENUES = [
        "— Neutral venue —",
        # 🇺🇸 USA (11 cities)
        "Los Angeles (USA)", "Dallas (USA)", "New York/New Jersey (USA)",
        "San Francisco Bay Area (USA)", "Miami (USA)", "Seattle (USA)",
        "Boston (USA)", "Houston (USA)", "Kansas City (USA)",
        "Atlanta (USA)", "Philadelphia (USA)",
        # 🇨🇦 Canada (2 cities)
        "Toronto (Canada)", "Vancouver (Canada)",
        # 🇲🇽 Mexico (3 cities)
        "Mexico City (Mexico)", "Guadalajara (Mexico)", "Monterrey (Mexico)",
    ]
    venue_sel = st.selectbox("📍 Venue city (applies host-nation Elo boost)", WC2026_VENUES)
    # Strip the country suffix before passing to get_elo
    venue_input = None if venue_sel.startswith("—") else venue_sel.split(" (")[0]

    if home_team != away_team:
        elo_h = get_elo(home_team, venue_input or None)
        elo_a = get_elo(away_team, venue_input or None)
        lh, la = expected_goals(elo_h, elo_a)
        ph, pd_, pa = match_result_probs_poisson(elo_h, elo_a)

        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(f"{TEAMS[home_team]['flag']} Elo", elo_h)
        m2.metric(f"{TEAMS[home_team]['flag']} Win %", f"{ph*100:.1f}%")
        m3.metric("Draw %", f"{pd_*100:.1f}%")
        m4.metric(f"{TEAMS[away_team]['flag']} Win %", f"{pa*100:.1f}%")
        m5.metric(f"{TEAMS[away_team]['flag']} Elo", elo_a)

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Expected Goals")
            eg_df = pd.DataFrame({
                "Team": [f"{TEAMS[home_team]['flag']} {home_team}",
                         f"{TEAMS[away_team]['flag']} {away_team}"],
                "xG": [round(lh, 2), round(la, 2)],
            })
            st.bar_chart(eg_df.set_index("Team"), height=200)

        with col_right:
            st.markdown("#### Top 8 Scorelines")
            top8 = top_scorelines(lh, la, n=8)
            score_rows = [
                {"Score": f"{h}-{a}", "Probability": f"{p*100:.1f}%",
                 "Winner": (f"🏆 {TEAMS[home_team]['flag']} {home_team}" if h > a
                            else (f"🏆 {TEAMS[away_team]['flag']} {away_team}" if a > h
                                  else "Draw 🤝"))}
                for (h, a), p in top8
            ]
            st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Full Scoreline Heatmap (probability %)")
        max_show = 7
        dist = scoreline_distribution(round(lh, 3), round(la, 3))
        heatmap_data = {}
        for a_goals in range(max_show):
            row = {}
            for h_goals in range(max_show):
                row[f"{h_goals}"] = round(dist.get((h_goals, a_goals), 0) * 100, 2)
            heatmap_data[f"Away {a_goals}"] = row
        heatmap_df = pd.DataFrame(heatmap_data).T
        heatmap_df.columns.name = "Home Goals →"
        heatmap_df.index.name = "Away Goals ↓"
        st.dataframe(heatmap_df.style.background_gradient(cmap="RdYlGn_r"),
                     use_container_width=True)
    else:
        st.warning("Select two different teams.")

# ============================================================================
# TAB 4 — Live Tracker
# ============================================================================
with tab4:
    st.subheader("📈 Live Results Tracker")
    st.caption(
        "Enter actual match scores as results come in. "
        "The model updates in real-time and tracks prediction accuracy."
    )

    st.markdown("### Enter Match Results")
    st.info(
        "Results entered here update the Win Probability and Group tables. "
        "The accuracy score shows how well the model's predictions match reality."
    )

    # Group selector
    sel_group = st.selectbox("Show group fixtures", list("ABCDEFGHIJKL"))
    group_fixes = get_group_fixtures(sel_group)

    for fix in group_fixes:
        col_a, col_vs, col_b, col_hg, col_ag, col_btn = st.columns(
            [2.5, 0.5, 2.5, 0.8, 0.8, 1.0]
        )
        col_a.write(f"**{TEAMS[fix.home]['flag']} {fix.home}**")
        col_vs.write("vs")
        col_b.write(f"**{TEAMS[fix.away]['flag']} {fix.away}**")

        existing = st.session_state.actual_results.get(fix.match_id)
        default_h = existing[0] if existing else 0
        default_a = existing[1] if existing else 0

        hg = col_hg.number_input(
            f"H_{fix.match_id}", min_value=0, max_value=20,
            value=default_h, label_visibility="collapsed",
        )
        ag = col_ag.number_input(
            f"A_{fix.match_id}", min_value=0, max_value=20,
            value=default_a, label_visibility="collapsed",
        )
        if col_btn.button("Save", key=f"save_{fix.match_id}"):
            st.session_state.actual_results[fix.match_id] = (int(hg), int(ag))
            _save_actual_results(st.session_state.actual_results)
            st.success(f"Saved: {fix.home} {hg}-{ag} {fix.away}")
            st.rerun()

    st.markdown("---")
    st.markdown("### Prediction Accuracy")

    played_fixtures = [
        fix for fix in FIXTURES
        if fix.match_id in st.session_state.actual_results
    ]

    if not played_fixtures:
        st.info("No results entered yet. Add some above to see accuracy tracking.")
    else:
        correct_result = 0
        correct_score = 0
        correct_winner = 0
        total = len(played_fixtures)

        accuracy_rows = []
        for fix in played_fixtures:
            actual = st.session_state.actual_results[fix.match_id]
            elo_h = get_elo(fix.home, fix.city)
            elo_a = get_elo(fix.away, fix.city)
            lh, la = expected_goals(elo_h, elo_a)
            ph, pd_, pa = match_result_probs_poisson(elo_h, elo_a)

            hg, ag = actual
            actual_result = "H" if hg > ag else ("A" if ag > hg else "D")
            pred_result = "H" if ph > pa and ph > pd_ else ("A" if pa > ph and pa > pd_ else "D")

            # Most likely scoreline
            from models.poisson import most_likely_score
            pred_score = most_likely_score(lh, la)

            r_correct = actual_result == pred_result
            s_correct = pred_score == (hg, ag)
            if r_correct:
                correct_result += 1
            if s_correct:
                correct_score += 1

            # Winner prediction accuracy (ignore draws)
            if actual_result != "D":
                correct_winner += 1 if r_correct else 0

            accuracy_rows.append({
                "Match": f"{TEAMS[fix.home]['flag']} {fix.home} vs {TEAMS[fix.away]['flag']} {fix.away}",
                "Actual": f"{hg}-{ag}",
                "Predicted": f"{pred_score[0]}-{pred_score[1]}",
                "Model Had": f"H {ph*100:.0f}% D {pd_*100:.0f}% A {pa*100:.0f}%",
                "Result ✓": "✅" if r_correct else "❌",
                "Score ✓": "✅" if s_correct else "❌",
            })

        # KPI row
        ka, kb, kc = st.columns(3)
        ka.metric("Result Accuracy", f"{correct_result}/{total}",
                  f"{correct_result/total*100:.1f}%")
        kb.metric("Exact Score Accuracy", f"{correct_score}/{total}",
                  f"{correct_score/total*100:.1f}%")
        kc.metric("Win/Loss Accuracy",
                  f"{correct_winner}/{sum(1 for f in played_fixtures if st.session_state.actual_results[f.match_id][0] != st.session_state.actual_results[f.match_id][1])}",
                  "excl. draws")

        st.dataframe(pd.DataFrame(accuracy_rows), hide_index=True, use_container_width=True)

        if st.button("🗑️ Clear all results"):
            st.session_state.actual_results = {}
            _save_actual_results(st.session_state.actual_results)
            st.rerun()

# ============================================================================
# TAB 5 — Tournament Bracket
# ============================================================================
with tab5:
    st.subheader("🗺️ Tournament Bracket — Predicted Progression")
    st.caption("Group-stage qualifiers and knockout probabilities from 10,000 Monte Carlo simulations")

    rkey = results_key()
    all_probs = cached_win_probs(rkey)
    standings, best_thirds_set, teams32 = get_bracket_qualifiers(rkey, all_probs)

    def get_flag(team: str) -> str:
        return TEAMS.get(team, {}).get("flag", "🏴")

    # ── Group Stage: predicted qualifiers ─────────────────────────────────────
    # Same ordering as the "Group Predictions" tab (avg finishing position),
    # plus the 8 best third-placed "wildcard" teams highlighted to match the
    # bracket below — so the two tabs always agree on who advances.
    st.markdown("#### 📋 Group Stage — Predicted Qualifiers")
    st.caption(
        "Top 2 per group advance automatically (🥇🥈). The 8 best third-placed "
        "teams (🎟️) also advance to the Round of 32 — matches the rankings in "
        "the Group Predictions tab."
    )
    group_html = '<div class="group-grid">'
    for grp in "ABCDEFGHIJKL":
        ordered = standings[grp]
        group_html += f'<div class="group-card"><div class="group-card-title">Group {grp}</div>'
        for i, team in enumerate(ordered):
            prob = all_probs.get(team, {}).get("R32", 0)
            flag = get_flag(team)
            if i == 0:
                cls = "first"
                medal = "🥇"
            elif i == 1:
                cls = "second"
                medal = "🥈"
            elif i == 2 and team in best_thirds_set:
                cls = "third-qualified"
                medal = "🎟️"
            else:
                cls = "eliminated"
                medal = "·"
            group_html += (
                f'<div class="group-card-team {cls}" title="{team} — R32 prob: {prob:.0%}">'
                f'{medal} {flag} {team}'
                f'</div>'
            )
        group_html += '</div>'
    group_html += '</div>'
    st.markdown(group_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Knockout Bracket (March-Madness style) ────────────────────────────────
    st.markdown("#### 🏆 Knockout Stage — Bracket")
    st.caption("Projected bracket — at every match the team with the higher probability of reaching the next round (highlighted in gold) advances along the connector to the next round's box")

    def render_mm_bracket(all_probs, get_flag, standings, best_thirds_set):
        BOX_H   = 36     # team-pair box height (2 rows x 18px)
        SLOT0   = 46     # R32 slot height
        MATCH_W = 132    # box width
        RW      = 150    # round width (box + gap)
        GAP     = RW - MATCH_W
        HALF_H  = 368    # total height of one half (8 * SLOT0)

        # Real FIFA WC2026 Round-of-32 bracket (data/knockout_fixtures.py),
        # resolved from the same group standings + best-3rd-place wildcards
        # shown in the "Group Stage — Predicted Qualifiers" panel above, so
        # the bracket and the group tables always agree on who advances.
        winners_map = {g: standings[g][0] for g in "ABCDEFGHIJKL"}
        runnersup_map = {g: standings[g][1] for g in "ABCDEFGHIJKL"}
        thirds_map = {g: standings[g][2] for g in "ABCDEFGHIJKL"}

        qualifying_third_groups = sorted(TEAM_GROUP[t] for t in best_thirds_set)
        third_assignment = resolve_third_place_assignment(qualifying_third_groups)
        r32_pairing = resolve_r32_pairing(winners_map, runnersup_map, thirds_map, third_assignment)

        # Flatten each half's 8 R32 matches into 16 teams, in the nesting
        # order build_half() expects (pairs of adjacent entries form one R16
        # match, pairs of those form one QF match, etc.)
        left16 = [team for m in LEFT_R32_ORDER for team in r32_pairing[m]]
        right16 = [team for m in RIGHT_R32_ORDER for team in r32_pairing[m]]

        def winner(a, b, decide_key):
            pa = all_probs.get(a, {}).get(decide_key, 0)
            pb = all_probs.get(b, {}).get(decide_key, 0)
            return a if pa >= pb else b

        def build_half(seed16):
            r32_matches = [(seed16[2 * i], seed16[2 * i + 1]) for i in range(8)]
            r32_winners = [winner(a, b, "R16") for a, b in r32_matches]
            r16_matches = [(r32_winners[2 * i], r32_winners[2 * i + 1]) for i in range(4)]
            r16_winners = [winner(a, b, "QF") for a, b in r16_matches]
            qf_matches  = [(r16_winners[2 * i], r16_winners[2 * i + 1]) for i in range(2)]
            qf_winners  = [winner(a, b, "SF") for a, b in qf_matches]
            sf_match    = (qf_winners[0], qf_winners[1])
            sf_winner   = winner(sf_match[0], sf_match[1], "Final")
            return r32_matches, r32_winners, r16_matches, r16_winners, qf_matches, qf_winners, sf_match, sf_winner

        def team_row(team, decide_key, win_team=None, big=False):
            prob = all_probs.get(team, {}).get(decide_key, 0)
            flag = get_flag(team)
            cls = "mm-row-lg" if big else "mm-row"
            if win_team is not None and team == win_team:
                cls += " mm-winner"
            return (
                f'<div class="{cls}">'
                f'<span class="mm-flag">{flag}</span>'
                f'<span class="mm-name">{team}</span>'
                f'<span class="mm-pct">{prob:.0%}</span>'
                f'</div>'
            )

        def render_half(side, seed16):
            (r32_matches, r32_winners, r16_matches, r16_winners,
             qf_matches, qf_winners, sf_match, sf_winner) = build_half(seed16)

            y_r32 = [i * SLOT0 + SLOT0 / 2 for i in range(8)]
            y_r16 = [(y_r32[2 * i] + y_r32[2 * i + 1]) / 2 for i in range(4)]
            y_qf  = [(y_r16[2 * i] + y_r16[2 * i + 1]) / 2 for i in range(2)]
            y_sf  = [(y_qf[0] + y_qf[1]) / 2]

            x_r32, x_r16, x_qf, x_sf = 0, RW, 2 * RW, 3 * RW

            html = '<div class="mm-half' + (' mm-mirror' if side == "R" else '') + '">'

            svg = f'<svg class="mm-svg" viewBox="0 0 582 {HALF_H}" preserveAspectRatio="none">'
            for x_prev, y_prev, x_next, y_next in (
                (x_r32, y_r32, x_r16, y_r16),
                (x_r16, y_r16, x_qf, y_qf),
                (x_qf, y_qf, x_sf, y_sf),
            ):
                xr = x_prev + MATCH_W
                xm = xr + GAP / 2
                xl = x_next
                for j in range(len(y_next)):
                    yp1, yp2 = y_prev[2 * j], y_prev[2 * j + 1]
                    yc = y_next[j]
                    svg += (
                        f'<polyline points="{xr},{yp1} {xm},{yp1} {xm},{yp2} {xr},{yp2}" '
                        'fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1.5"/>'
                    )
                    svg += (
                        f'<polyline points="{xm},{yc} {xl},{yc}" '
                        'fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1.5"/>'
                    )
            svg += '</svg>'
            html += svg

            def box(x, y, t1, t2, decide_key, win_team, extra_cls=""):
                return (
                    f'<div class="mm-box {extra_cls}" '
                    f'style="left:{x}px; top:{y - BOX_H / 2}px; width:{MATCH_W}px; height:{BOX_H}px;">'
                    + team_row(t1, decide_key, win_team) + team_row(t2, decide_key, win_team) + '</div>'
                )

            for i in range(8):
                a, b = r32_matches[i]
                html += box(x_r32, y_r32[i], a, b, "R16", r32_winners[i])
            for i in range(4):
                a, b = r16_matches[i]
                html += box(x_r16, y_r16[i], a, b, "QF", r16_winners[i])
            for i in range(2):
                a, b = qf_matches[i]
                html += box(x_qf, y_qf[i], a, b, "SF", qf_winners[i], "mm-box2")
            html += box(x_sf, y_sf[0], sf_match[0], sf_match[1], "Final", sf_winner, "mm-box2")

            html += '</div>'
            return html, sf_winner

        round_labels = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals"]
        labels_left = '<div class="mm-round-labels">' + ''.join(f'<span>{l}</span>' for l in round_labels) + '</div>'
        labels_right = '<div class="mm-round-labels mm-mirror">' + ''.join(f'<span>{l}</span>' for l in round_labels) + '</div>'
        labels_row = f'<div class="mm-labels-row">{labels_left}<div class="mm-center"></div>{labels_right}</div>'

        left_html, left_finalist = render_half("L", left16)
        right_html, right_finalist = render_half("R", right16)
        champion = winner(left_finalist, right_finalist, "Winner")
        champ_prob = all_probs.get(champion, {}).get("Winner", 0)

        center = (
            '<div class="mm-center">'
            '<div class="mm-final-label">Final</div>'
            f'<div class="mm-final-box">{team_row(left_finalist, "Winner", champion, big=True)}</div>'
            f'<div class="mm-final-box">{team_row(right_finalist, "Winner", champion, big=True)}</div>'
            '<div class="mm-champion">'
            f'🏆 {get_flag(champion)} {champion}'
            f'<div class="mm-champ-pct">Champion · {champ_prob:.0%}</div>'
            '</div>'
            '</div>'
        )

        bracket_html = labels_row + f'<div class="mm-bracket">{left_html}{center}{right_html}</div>'

        # Expose the predicted finalists/champion so the "Prediction Engine"
        # section below can run a cross-check on the Final without
        # recomputing the bracket-walk logic.
        bracket_info = {
            "left_finalist": left_finalist,
            "right_finalist": right_finalist,
            "champion": champion,
            "champ_prob": champ_prob,
        }
        return bracket_html, bracket_info

    bracket_html, bracket_info = render_mm_bracket(all_probs, get_flag, standings, best_thirds_set)
    st.markdown(bracket_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Prediction Engine: models behind this bracket ─────────────────────────
    st.markdown("#### 🧠 Prediction Engine")
    st.caption(
        "Every probability on this page comes from the same pipeline — six "
        "models working together, calibrated on real World Cup and "
        "international results."
    )

    logreg_model, logreg_trained = cached_logreg_model()
    logreg_status = "Trained on WC history" if logreg_trained else "Elo fallback (offline)"

    rf_model, rf_trained = cached_rf_model()
    rf_status = "Trained on recent-era results" if rf_trained else "Elo fallback"

    xgb_model, xgb_trained = cached_xgb_model()
    xgb_status = "Trained on recent-era results" if xgb_trained else "Elo fallback"

    model_html = (
        '<div class="model-grid">'
        '<div class="model-card">'
        '<div class="model-card-title">📐 Elo Ratings</div>'
        '<div class="model-card-body">Baseline team-strength ratings, '
        'calibrated from World Cup &amp; international results, with a '
        '+100 home-advantage boost for Mexico, USA &amp; Canada.</div>'
        '</div>'
        '<div class="model-card">'
        '<div class="model-card-title">📊 Bivariate Poisson</div>'
        '<div class="model-card-body">Converts Elo gaps into expected goals '
        'and a full scoreline distribution for every match-up — powers the '
        'Match Predictor tab.</div>'
        '</div>'
        '<div class="model-card">'
        '<div class="model-card-title">🎲 Monte Carlo (10,000 sims)</div>'
        '<div class="model-card-body">Replays the group draw, wildcard '
        'tiebreaks and every knockout round 10,000 times to produce all '
        'probabilities shown in this bracket.</div>'
        '</div>'
        '<div class="model-card">'
        '<div class="model-card-title">🤖 Logistic Regression</div>'
        '<div class="model-card-body">Multinomial classifier trained on '
        'historical World Cup matches — used below as an independent '
        'cross-check on the Final.</div>'
        f'<div class="model-card-status">{logreg_status}</div>'
        '</div>'
        '<div class="model-card">'
        '<div class="model-card-title">🌲 Random Forest</div>'
        '<div class="model-card-body">Bagged-tree ensemble trained on '
        f'international results since {RECENT_ERA_START[:4]} — captures '
        'non-linear patterns the linear model misses, used as a further '
        'cross-check on the Final.</div>'
        f'<div class="model-card-status">{rf_status}</div>'
        '</div>'
        '<div class="model-card">'
        '<div class="model-card-title">⚡ XGBoost</div>'
        '<div class="model-card-body">Gradient-boosted-tree ensemble over '
        f'the same recent-era ({RECENT_ERA_START[:4]}+) results and feature '
        'set — a final independent cross-check on the Final.</div>'
        f'<div class="model-card-status">{xgb_status}</div>'
        '</div>'
        '</div>'
    )
    st.markdown(model_html, unsafe_allow_html=True)

    # ── Final cross-check: Monte Carlo bracket vs. ML models ──────────────────
    lf, rf = bracket_info["left_finalist"], bracket_info["right_finalist"]
    champion, champ_prob = bracket_info["champion"], bracket_info["champ_prob"]
    logreg_probs = logreg_model.predict(
        home_elo=get_elo(lf), away_elo=get_elo(rf), is_neutral=True,
    )
    logreg_pick = lf if logreg_probs["home_win"] >= logreg_probs["away_win"] else rf
    agree = "✅ Agrees with Monte Carlo" if logreg_pick == champion else "⚠️ Differs from Monte Carlo"

    rf_probs = rf_model.predict(home_elo=get_elo(lf), away_elo=get_elo(rf), is_neutral=True)
    rf_pick = lf if rf_probs["home_win"] >= rf_probs["away_win"] else rf
    rf_agree = "✅ Agrees with Monte Carlo" if rf_pick == champion else "⚠️ Differs from Monte Carlo"

    xgb_probs = xgb_model.predict(home_elo=get_elo(lf), away_elo=get_elo(rf), is_neutral=True)
    xgb_pick = lf if xgb_probs["home_win"] >= xgb_probs["away_win"] else rf
    xgb_agree = "✅ Agrees with Monte Carlo" if xgb_pick == champion else "⚠️ Differs from Monte Carlo"

    st.markdown("##### 🔍 Final Cross-Check — Monte Carlo Bracket vs. ML Models")
    st.caption(
        f"The bracket's projected Final is {get_flag(lf)} {lf} vs {get_flag(rf)} {rf}. "
        "Compare the Monte Carlo simulation's overall champion pick against "
        "independent reads of that single match from the logistic regression, "
        "random forest, and XGBoost models."
    )
    cc_html = (
        '<div class="crosscheck-grid">'
        '<div class="crosscheck-card">'
        '<div class="crosscheck-label">Monte Carlo · Champion</div>'
        f'<div class="crosscheck-value">{get_flag(champion)} {champion}</div>'
        f'<div class="crosscheck-sub">{champ_prob:.0%} of 10,000 simulations</div>'
        '</div>'
        '<div class="crosscheck-card">'
        '<div class="crosscheck-label">Logistic Regression · Final</div>'
        f'<div class="crosscheck-value">{get_flag(lf)} {logreg_probs["home_win"]:.0%} '
        f'&nbsp;·&nbsp; {get_flag(rf)} {logreg_probs["away_win"]:.0%}</div>'
        f'<div class="crosscheck-sub">Draw allocation {logreg_probs["draw"]:.0%} '
        '(resolved by penalties)</div>'
        '</div>'
        '<div class="crosscheck-card">'
        '<div class="crosscheck-label">Logistic Regression · Pick</div>'
        f'<div class="crosscheck-value">{get_flag(logreg_pick)} {logreg_pick}</div>'
        f'<div class="crosscheck-sub">{agree}</div>'
        '</div>'
        '<div class="crosscheck-card">'
        '<div class="crosscheck-label">Random Forest · Pick</div>'
        f'<div class="crosscheck-value">{get_flag(rf_pick)} {rf_pick}</div>'
        f'<div class="crosscheck-sub">{rf_agree}</div>'
        '</div>'
        '<div class="crosscheck-card">'
        '<div class="crosscheck-label">XGBoost · Pick</div>'
        f'<div class="crosscheck-value">{get_flag(xgb_pick)} {xgb_pick}</div>'
        f'<div class="crosscheck-sub">{xgb_agree}</div>'
        '</div>'
        '</div>'
    )
    st.markdown(cc_html, unsafe_allow_html=True)

# ============================================================================
# TAB 6 — Model Backtest
# ============================================================================
with tab6:
    st.subheader("🧪 Model Backtest — 2018 & 2022 World Cups")
    st.caption(
        "Point-in-time evaluation: each model is retrained using only data "
        "available BEFORE that tournament's opening match, then scored "
        "against the actual results — so models aren't graded on data they "
        "were trained on."
    )

    if st.button("▶️ Run backtest", help="Trains separate point-in-time copies of "
                  "Logistic Regression, Random Forest and XGBoost — first run can "
                  "take 15-30 seconds, then it's cached for 24 hours."):
        st.session_state["backtest_results"] = cached_backtest()

    backtest_results = st.session_state.get("backtest_results")

    if backtest_results is None:
        st.info(
            "Click **Run backtest** to evaluate Elo, Elo+Poisson, Monte Carlo, "
            "Logistic Regression, Random Forest, and XGBoost against the actual "
            "2018 and 2022 World Cup results."
        )
    else:
        any_ok = False
        for year in (2018, 2022):
            res = backtest_results.get(year, {})
            if "error" in res:
                st.warning(f"{year}: {res['error']}")
                continue
            any_ok = True

            st.markdown(f"#### {year} FIFA World Cup &nbsp;·&nbsp; {res['n_matches']} matches")
            st.caption(
                f"Point-in-time training data (as of kickoff): "
                f"{res['n_train_wc']} prior World Cup matches for Logistic "
                f"Regression, {res['n_train_recent']:,} recent-era matches "
                f"(since {RECENT_ERA_START[:4]}) for Random Forest / XGBoost. "
                f"Random Forest trained: {'✅' if res['rf_trained'] else '⚠️ fallback to Elo'} "
                f"&nbsp;·&nbsp; XGBoost trained: {'✅' if res['xgb_trained'] else '⚠️ fallback to Elo'} "
                f"&nbsp;·&nbsp; Logistic Regression trained: "
                f"{'✅' if res['logreg_trained'] else '⚠️ fallback to Elo'}"
            )

            summary_rows = [
                {
                    "Model": name,
                    "Accuracy": f"{m['accuracy']:.1%}",
                    "Brier Score": f"{m['brier']:.3f}",
                    "Matches Scored": m["n"],
                }
                for name, m in res["models"].items()
            ]
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

            best_brier = min(res["models"].items(), key=lambda kv: kv[1]["brier"])
            best_acc = max(res["models"].items(), key=lambda kv: kv[1]["accuracy"])
            st.caption(
                f"Best calibration (lowest Brier): **{best_brier[0]}** "
                f"({best_brier[1]['brier']:.3f}) &nbsp;·&nbsp; "
                f"Best accuracy: **{best_acc[0]}** ({best_acc[1]['accuracy']:.1%})"
            )

            with st.expander(f"Match-by-match picks — {year}"):
                st.dataframe(
                    pd.DataFrame(res["match_details"]),
                    hide_index=True,
                    use_container_width=True,
                )
            st.divider()

        if any_ok:
            st.caption(
                "Accuracy = share of matches where the model's most-likely "
                "outcome matched the actual result. Brier score (multi-class, "
                "range 0-2, lower is better) = sum of squared errors between "
                "predicted and actual outcome probabilities — rewards "
                "well-calibrated confidence, not just correct picks. "
                "'Monte Carlo' draws 5,000 scorelines per match from the same "
                "Elo→Poisson distribution used by the live tournament simulator."
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='footer-bar'>"
    "⚽ Goal Analytics &nbsp;·&nbsp; FIFA World Cup 2026 &nbsp;·&nbsp; "
    "Elo + Bivariate Poisson + Logistic Regression + Monte Carlo + "
    "Random Forest + XGBoost &nbsp;·&nbsp; "
    "<a href='https://github.com/nithinnarla/goal-analytics' target='_blank'>GitHub ↗</a>"
    "</div>",
    unsafe_allow_html=True,
)
