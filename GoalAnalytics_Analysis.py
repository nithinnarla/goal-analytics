"""
GoalAnalytics — Full Analysis Script
=====================================
World Cup 2026 AI Prediction Engine

Matches GoalAnalytics_Analysis.ipynb cell-for-cell.
Run this script to reproduce all visualizations and predictions.

Usage
-----
    cd goal-analytics/
    python GoalAnalytics_Analysis.py

Output figures are saved to: figures/

Requirements
------------
    pip install pandas numpy requests seaborn matplotlib scikit-learn
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Setup
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("GoalAnalytics — World Cup 2026 Prediction Engine")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Fetch Historical World Cup Data (1930–2022)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/8] Fetching historical World Cup data...")

from data.historical import (
    fetch_results,
    get_wc_matches,
    get_wc_stats,
    compute_elo_ratings,
    build_form_lookup,
    WC_WINNERS,
    WC_HOSTS,
)

# Attempt to fetch live data; fall back gracefully if offline
results_df = fetch_results()
wc_df      = get_wc_matches(results_df)
wc_stats   = get_wc_stats(wc_df)

print(f"  WC tournaments covered: {len(wc_stats['matches_by_year'])} (1930–2022)")
print(f"  Total WC matches: {sum(wc_stats['matches_by_year'].values())}")
print(f"  Most titles: {max(wc_stats['titles'], key=wc_stats['titles'].get)} "
      f"({max(wc_stats['titles'].values())})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Compute Elo Ratings from History
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/8] Computing historical Elo ratings...")

from data.teams import TEAMS, get_elo as preset_get_elo

if results_df is not None:
    historical_elos = compute_elo_ratings(results_df)
    print(f"  Elo computed for {len(historical_elos)} teams from {len(results_df):,} matches")
    # Blend: preset Elo (hand-calibrated) + historical derivation (60/40)
    blended_elos = {}
    for team in TEAMS:
        preset  = preset_get_elo(team)
        derived = historical_elos.get(team, preset)
        blended_elos[team] = 0.6 * preset + 0.4 * derived
else:
    print("  (Offline) Using preset Elo ratings from teams.py")
    blended_elos = {t: preset_get_elo(t) for t in TEAMS}

# Show top 10
top10 = sorted(blended_elos.items(), key=lambda x: -x[1])[:10]
print("\n  Top 10 Elo Ratings:")
for rank, (team, elo) in enumerate(top10, 1):
    flag = TEAMS.get(team, {}).get("flag", "")
    print(f"    {rank:2}. {flag} {team:<25} {elo:.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Feature Engineering & Logistic Regression Training
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/8] Training logistic regression model...")

from models.logistic import LogisticMatchPredictor
from models.features import build_match_features_df

logistic_model = LogisticMatchPredictor()

if wc_df is not None and results_df is not None:
    form_lookup = build_form_lookup(results_df, as_of="2026-06-01", n=15)
    X, y = build_match_features_df(wc_df, blended_elos, form_lookup)
    logistic_model.train(X, y, cv=5)
    print(logistic_model.summary())
else:
    print("  (Offline) Logistic model will use Elo fallback (no training data)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — All 72 Group Stage Match Predictions
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/8] Predicting all 72 group stage matches...")

from data.fixtures import FIXTURES, get_group_fixtures
from data import GROUPS
from models.elo import match_probabilities
from models.poisson import most_likely_score, expected_goals

predictions = {}
for fixture in FIXTURES:
    elo_h = blended_elos.get(fixture.home, preset_get_elo(fixture.home))
    elo_a = blended_elos.get(fixture.away, preset_get_elo(fixture.away))

    elo_probs = match_probabilities(elo_h, elo_a)
    xg_h, xg_a = expected_goals(elo_h, elo_a)
    score_h, score_a = most_likely_score(xg_h, xg_a)

    # Logistic regression prediction
    home_form = form_lookup.get(fixture.home, {}) if results_df else {}
    away_form = form_lookup.get(fixture.away, {}) if results_df else {}
    logistic_probs = logistic_model.predict(
        home_elo=elo_h,
        away_elo=elo_a,
        home_form=home_form.get("form", 1.5),
        away_form=away_form.get("form", 1.5),
        home_scored=home_form.get("avg_scored", 1.3),
        away_scored=away_form.get("avg_scored", 1.3),
        home_conceded=home_form.get("avg_conceded", 1.1),
        away_conceded=away_form.get("avg_conceded", 1.1),
    )

    predictions[fixture.match_id] = {
        "home": fixture.home,
        "away": fixture.away,
        "group": fixture.group,
        "venue": fixture.venue,
        "elo_home_win": elo_probs["home_win"],
        "elo_draw":     elo_probs["draw"],
        "elo_away_win": elo_probs["away_win"],
        "log_home_win": logistic_probs["home_win"],
        "log_draw":     logistic_probs["draw"],
        "log_away_win": logistic_probs["away_win"],
        "xg_home":      round(xg_h, 2),
        "xg_away":      round(xg_a, 2),
        "predicted_score": f"{score_h}-{score_a}",
    }

preds_df = pd.DataFrame.from_dict(predictions, orient="index")
print(f"  Predicted {len(preds_df)} matches")
print("\n  Sample predictions (Group A):")
group_a_preds = preds_df[preds_df["group"] == "A"][
    ["home", "away", "predicted_score", "elo_home_win", "elo_draw", "elo_away_win"]
].head(3)
print(group_a_preds.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Monte Carlo Simulation (Win Probabilities)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/8] Running 10,000-simulation Monte Carlo tournament...")

from models.monte_carlo import win_probabilities, simulate_group

sim_n = 10_000
n_group_sims = 2_000
print(f"  Running {sim_n:,} tournament simulations...")
win_probs = win_probabilities(n=sim_n, known_results={}, seed=42)

# Group finish distributions — simulate each group independently
print(f"  Computing group finish distributions ({n_group_sims} sims per group)...")
group_finish: dict[str, dict] = {}
for grp in sorted(GROUPS.keys()):
    for _ in range(n_group_sims):
        ranked = simulate_group(grp, {})
        for pos, rec in enumerate(ranked, 1):
            group_finish.setdefault(rec.team, {})
            group_finish[rec.team][pos] = group_finish[rec.team].get(pos, 0) + 1

for team in group_finish:
    total = sum(group_finish[team].values())
    for pos in group_finish[team]:
        group_finish[team][pos] /= total

print("\n  Top 10 Tournament Win Probabilities:")
top10_win = sorted(win_probs.items(), key=lambda x: -x[1])[:10]
for i, (team, p) in enumerate(top10_win, 1):
    flag = TEAMS.get(team, {}).get("flag", "")
    print(f"    {i:2}. {flag} {team:<22} {p*100:5.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Generate & Save All Visualizations
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/8] Generating visualizations...")

from analysis.visualizations import (
    plot_wc_winners_history,
    plot_goals_per_tournament,
    plot_group_standings,
    plot_win_probabilities,
    plot_scoreline_heatmap,
    plot_model_comparison,
    plot_knockout_bracket,
    plot_host_advantage,
    build_predicted_bracket,
    save_fig,
)

figures_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(figures_dir, exist_ok=True)

# Fig 01 — WC winners history
print("  01/08 WC winners history...")
fig01 = plot_wc_winners_history(wc_stats)
save_fig(fig01, "01_wc_winners_history.png")
plt.close(fig01)

# Fig 02 — Goals per tournament
print("  02/08 Goals per tournament...")
fig02 = plot_goals_per_tournament(wc_stats)
save_fig(fig02, "02_goals_per_tournament.png")
plt.close(fig02)

# Fig 03 — All 12 group standings
print("  03/08 Group stage standings (all 12 groups)...")
fig03 = plot_group_standings(group_finish)
save_fig(fig03, "03_all_group_standings.png")
plt.close(fig03)

# Figs 03a-03l — Individual group charts
for grp in sorted(GROUPS.keys()):
    fig = plot_group_standings(group_finish, group=grp)
    save_fig(fig, f"03_{grp}_group_standings.png")
    plt.close(fig)

# Fig 04 — Win probabilities
print("  04/08 Tournament win probabilities...")
fig04 = plot_win_probabilities(win_probs, top_n=20)
save_fig(fig04, "04_win_probabilities.png")
plt.close(fig04)

# Fig 05 — Scoreline heatmap (Argentina vs France showcase)
print("  05/08 Argentina vs France scoreline heatmap...")
elo_arg = blended_elos.get("Argentina", 2100)
elo_fra = blended_elos.get("France",    2050)
fig05 = plot_scoreline_heatmap("Argentina", "France", elo_arg, elo_fra)
save_fig(fig05, "05_argentina_vs_france_scoreline.png")
plt.close(fig05)

# Fig 06 — Model comparison
print("  06/08 Model comparison (Elo vs Logistic)...")
elo_probs_arg_fra = match_probabilities(elo_arg, elo_fra)
log_probs_arg_fra = logistic_model.predict(
    home_elo=elo_arg, away_elo=elo_fra,
    home_form=form_lookup.get("Argentina", {}).get("form", 1.5) if results_df else 1.5,
    away_form=form_lookup.get("France",    {}).get("form", 1.5) if results_df else 1.5,
)
fig06 = plot_model_comparison(
    "Argentina", "France", elo_probs_arg_fra, log_probs_arg_fra,
    logistic_trained=logistic_model.trained,
)
save_fig(fig06, "06_model_comparison_arg_fra.png")
plt.close(fig06)

# Fig 07 — Knockout bracket
print("  07/08 Predicted knockout bracket...")
bracket = build_predicted_bracket(win_probs)
fig07 = plot_knockout_bracket(bracket)
save_fig(fig07, "07_knockout_bracket.png")
plt.close(fig07)

# Fig 08 — Host advantage
print("  08/08 Host nation advantage...")
fig08 = plot_host_advantage()
save_fig(fig08, "08_host_advantage.png")
plt.close(fig08)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Full Knockout Stage Predictions (Predicted bracket path)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/8] Knockout stage predictions (by win probability seed order)...")

print("\n  Predicted R32 →")
for i in range(0, 32, 2):
    h = bracket["r32"][i]
    a = bracket["r32"][i+1] if i+1 < len(bracket["r32"]) else "BYE"
    phw = win_probs.get(h, 0)
    paw = win_probs.get(a, 0)
    total = phw + paw
    ph = phw / total if total > 0 else 0.5
    pa = paw / total if total > 0 else 0.5
    winner = h if phw >= paw else a
    print(f"    {h:<22} vs {a:<22} → ★ {winner}  ({ph*100:.0f}%/{pa*100:.0f}%)")

rounds = [("r16", "R16"), ("qf", "Quarterfinals"), ("sf", "Semifinals"),
           ("final", "Final"), ("winner", "🏆 CHAMPION")]
for key, label in rounds:
    teams = bracket.get(key, [])
    print(f"\n  Predicted {label}:")
    for t in teams:
        print(f"    ★ {t}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Summary Report
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/8] Summary Report")
print("-" * 60)

champion = bracket["winner"][0] if bracket["winner"] else "N/A"
runner_up = bracket["final"][0] if len(bracket.get("final", [])) > 1 else (
            bracket["final"][1] if len(bracket.get("final", [])) > 1 else bracket["sf"][0]
            if bracket.get("sf") else "N/A")

print(f"  Predicted Champion:    {TEAMS.get(champion, {}).get('flag', '🏆')} {champion}")
print(f"  Champion Win Prob:     {win_probs.get(champion, 0)*100:.1f}%")
print(f"\n  Logistic model status: {'Trained' if logistic_model.trained else 'Fallback (offline)'}")
if logistic_model.cv_accuracy:
    print(f"  Logistic CV accuracy:  {logistic_model.cv_accuracy:.1%}")
print(f"\n  Figures saved to:      {os.path.abspath(figures_dir)}/")
print(f"  Total figures:         {len(os.listdir(figures_dir))}")

print("\n" + "=" * 60)
print("  Analysis complete.")
print("=" * 60)
