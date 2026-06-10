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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GoalAnalytics | WC 2026",
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
   ANIMATED BACKGROUND LAYERS
═══════════════════════════════════════════ */
.bg-animate {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: -10;
    background: linear-gradient(
        135deg,
        #020810 0%, #041508 25%,
        #080220 50%, #041508 75%,
        #020810 100%
    );
    background-size: 400% 400%;
    animation: fieldGlow 22s ease infinite;
}

@keyframes fieldGlow {
    0%   { background-position: 0% 50%; }
    25%  { background-position: 100% 0%; }
    50%  { background-position: 100% 100%; }
    75%  { background-position: 0% 100%; }
    100% { background-position: 0% 50%; }
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
   FLOATING FOOTBALL PARTICLES
═══════════════════════════════════════════ */
.particle {
    position: fixed;
    pointer-events: none;
    z-index: -7;
    opacity: 0;
    user-select: none;
}

@keyframes floatBall {
    0%   { transform: translateY(105vh) rotate(0deg);   opacity: 0; }
    8%   { opacity: 0.18; }
    88%  { opacity: 0.12; }
    100% { transform: translateY(-8vh) rotate(540deg);  opacity: 0; }
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
    animation: logoFloat 6s ease-in-out infinite;
}

@keyframes logoFloat {
    0%, 100% { transform: translateY(0px) scale(1); }
    50%       { transform: translateY(-9px) scale(1.03); }
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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Animated background layers
# ---------------------------------------------------------------------------
st.markdown("""
<div class="bg-animate"></div>
<div class="spotlight"></div>
<div class="pitch-lines"></div>

<!-- Floating football particles -->
<span class="particle" style="left:3%;font-size:1.8rem;animation:floatBall 14s 0s linear infinite;">⚽</span>
<span class="particle" style="left:10%;font-size:1.1rem;animation:floatBall 11s 2s linear infinite;">⚽</span>
<span class="particle" style="left:18%;font-size:1.5rem;animation:floatBall 17s 4.5s linear infinite;">⚽</span>
<span class="particle" style="left:26%;font-size:0.9rem;animation:floatBall 12s 1s linear infinite;">⚽</span>
<span class="particle" style="left:35%;font-size:2rem;animation:floatBall 19s 6s linear infinite;">⚽</span>
<span class="particle" style="left:43%;font-size:1.3rem;animation:floatBall 13s 3s linear infinite;">⚽</span>
<span class="particle" style="left:51%;font-size:1.6rem;animation:floatBall 15s 8s linear infinite;">⚽</span>
<span class="particle" style="left:59%;font-size:1rem;animation:floatBall 10s 0.5s linear infinite;">⚽</span>
<span class="particle" style="left:67%;font-size:1.4rem;animation:floatBall 18s 5s linear infinite;">⚽</span>
<span class="particle" style="left:75%;font-size:0.8rem;animation:floatBall 12s 2.5s linear infinite;">⚽</span>
<span class="particle" style="left:82%;font-size:1.7rem;animation:floatBall 14s 7s linear infinite;">⚽</span>
<span class="particle" style="left:90%;font-size:1.2rem;animation:floatBall 11s 1.5s linear infinite;">⚽</span>
<span class="particle" style="left:97%;font-size:1.5rem;animation:floatBall 16s 4s linear infinite;">⚽</span>
<span class="particle" style="left:55%;font-size:0.9rem;animation:floatBall 9s 9s linear infinite;">🏆</span>
<span class="particle" style="left:30%;font-size:1.1rem;animation:floatBall 13s 11s linear infinite;">🏆</span>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <img class="wc-logo"
         src="https://upload.wikimedia.org/wikipedia/en/5/5c/2026_FIFA_World_Cup_logo.svg"
         alt="FIFA World Cup 2026"
         onerror="this.style.display='none'"/>
    <h1>⚽ GoalAnalytics</h1>
    <div class="hdr-sub">FIFA World Cup 2026 · Prediction Engine</div>
    <div class="hdr-model">Elo · Bivariate Poisson · 10,000 Monte Carlo Simulations · 48 Teams · 104 Matches</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state: actual results input by user
# ---------------------------------------------------------------------------
if "actual_results" not in st.session_state:
    st.session_state.actual_results = {}  # match_id -> (hg, ag)

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


# ---------------------------------------------------------------------------
# Build results key for cache
# ---------------------------------------------------------------------------
def results_key() -> str:
    return json.dumps(st.session_state.actual_results) if st.session_state.actual_results else ""


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Win Probabilities",
    "📊 Group Predictions",
    "📍 Match Predictor",
    "📈 Live Tracker",
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
            st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='footer-bar'>"
    "⚽ GoalAnalytics &nbsp;·&nbsp; FIFA World Cup 2026 &nbsp;·&nbsp; "
    "Elo + Bivariate Poisson + Monte Carlo &nbsp;·&nbsp; "
    "<a href='https://github.com/nithinnarla/goal-analytics' target='_blank'>GitHub ↗</a>"
    "</div>",
    unsafe_allow_html=True,
)
