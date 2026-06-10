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
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0; }
    .metric-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .team-prob { font-size: 1.4rem; font-weight: bold; color: #e94560; }
    .group-header {
        background: #0f3460;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.95rem;
    }
    .match-card {
        border-left: 4px solid #e94560;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        background: #16213e;
        border-radius: 0 8px 8px 0;
    }
    .winner-badge {
        background: #e94560;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    .accuracy-good { color: #4caf50; font-weight: bold; }
    .accuracy-bad  { color: #f44336; font-weight: bold; }
    .accuracy-ok   { color: #ff9800; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>⚽ GoalAnalytics</h1>
    <p>World Cup 2026 · Elo + Poisson + Monte Carlo · 10,000 simulations</p>
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
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;font-size:0.8rem;'>"
    "GoalAnalytics · World Cup 2026 · Built with Elo + Poisson + Monte Carlo · "
    "<a href='https://github.com/nithinnarla/goal-analytics' target='_blank'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
