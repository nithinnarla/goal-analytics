"""
All visualization functions for GoalAnalytics.

Every function returns a matplotlib Figure so it can be:
  - Displayed inline in Jupyter:  display(fig) or just fig
  - Saved to disk in scripts:     fig.savefig("figures/name.png", ...)
  - Shown interactively:          plt.show()

Charts produced:
  01. wc_winners_history     – Historical WC title counts (bar)
  02. goals_per_tournament   – Avg goals per game by year (line)
  03. group_standings        – All 12 groups predicted standings (faceted bar)
  04. win_probabilities      – Top-N teams by tournament win % (horizontal bar)
  04b. round_progression      – P(reach R32/R16/QF/SF/Final/Win) line chart for
                                the top-N teams by win probability — shows
                                where each contender's run is most likely to
                                end, from models.monte_carlo.run_simulations().
  05. scoreline_heatmap      – Expected score grid for a match (seaborn heatmap)
  06. model_comparison       – Grouped bar chart of model win-probs for a sample
                                match. elo_probs + logistic_probs are required;
                                rf_probs, xgb_probs, mc_probs (Random Forest,
                                XGBoost, Monte Carlo) are optional extra bars.
  06b. backtest_results      – plot_backtest_results(): per-model accuracy bars
                                from models.backtest.run_full_backtest()'s
                                {2018, 2022} output, with graceful handling of
                                its {"error": ...} case (e.g. offline sandbox).
  07. knockout_bracket       – Bracket tree. Build with build_bracket_from_simulation()
                                (real WC2026 Annex-C bracket, from
                                models.monte_carlo.simulate_full_tournament_detailed() —
                                matches the dashboard) or, for a quick illustrative
                                figure only, build_predicted_bracket() (simplified
                                "higher seed always wins" seeding — see its docstring
                                for why it does NOT match the live app).
  08. host_advantage         – USA/Mexico/Canada home xG boost
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, Tuple

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # non-interactive backend (safe for scripts + notebooks)

# ─── Style ───────────────────────────────────────────────────────────────────
DARK_BG    = "#0a0a0a"
PANEL_BG   = "#16213e"
ACCENT     = "#e94560"
ACCENT2    = "#0f3460"
TEXT       = "#e0e0e0"
SUBTEXT    = "#a0a0a0"
GOLD       = "#f4c430"
SILVER     = "#c0c0c0"
BRONZE     = "#cd7f32"

PALETTE = [ACCENT, "#4db8ff", "#7fff7f", "#ffbf00", "#c87fff", "#ff8c00"]

CONF_COLORS = {
    "UEFA":     "#4db8ff",
    "CONMEBOL": "#7fff7f",
    "CAF":      "#ffbf00",
    "CONCACAF": "#e94560",
    "AFC":      "#c87fff",
    "OFC":      "#ff8c00",
}

def _apply_dark(fig: plt.Figure, ax_list=None):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_list or fig.axes):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(SUBTEXT)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def save_fig(fig: plt.Figure, name: str, dpi: int = 150):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓ Saved → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 01. Historical WC title counts
# ═══════════════════════════════════════════════════════════════════════════════

def plot_wc_winners_history(wc_stats: dict) -> plt.Figure:
    """Bar chart: countries ranked by number of World Cup titles."""
    titles = wc_stats.get("titles", {})
    df = pd.Series(titles).sort_values(ascending=False).reset_index()
    df.columns = ["Country", "Titles"]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [GOLD if t == df["Titles"].max() else ACCENT2 for t in df["Titles"]]
    bars = ax.bar(df["Country"], df["Titles"], color=colors, edgecolor=SUBTEXT, linewidth=0.5)

    # Labels
    for bar, val in zip(bars, df["Titles"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(int(val)), ha="center", va="bottom", color=TEXT, fontsize=11, fontweight="bold")

    ax.set_title("🏆 FIFA World Cup Titles by Nation (1930–2022)", fontsize=14,
                 fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("Country", labelpad=8)
    ax.set_ylabel("Titles", labelpad=8)
    ax.set_ylim(0, df["Titles"].max() + 0.8)
    _apply_dark(fig)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 02. Goals per tournament
# ═══════════════════════════════════════════════════════════════════════════════

def plot_goals_per_tournament(wc_stats: dict) -> plt.Figure:
    """Line chart: avg goals per game per World Cup year."""
    avg = wc_stats.get("avg_goals_by_year", {})
    if not avg:
        # Hard-coded fallback (representative values)
        avg = {
            1930:3.89, 1934:4.12, 1938:4.67, 1950:4.00, 1954:5.38,
            1958:3.60, 1962:2.78, 1966:2.34, 1970:2.97, 1974:2.55,
            1978:2.68, 1982:2.81, 1986:2.54, 1990:2.21, 1994:2.71,
            1998:2.67, 2002:2.52, 2006:2.30, 2010:2.27, 2014:2.67,
            2018:2.64, 2022:2.70,
        }

    years = sorted(avg.keys())
    vals  = [avg[y] for y in years]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(years, vals, color=ACCENT, linewidth=2.5, marker="o",
            markersize=6, markerfacecolor=GOLD)
    ax.fill_between(years, vals, alpha=0.15, color=ACCENT)

    # Annotate notable tournaments
    notable = {1954: "5.38 — GIF record", 1990: "2.21 — Lowest ever"}
    for yr, label in notable.items():
        if yr in avg:
            ax.annotate(label, xy=(yr, avg[yr]), xytext=(yr + 1, avg[yr] + 0.3),
                        color=GOLD, fontsize=8, arrowprops=dict(arrowstyle="->", color=SUBTEXT))

    ax.axhline(np.mean(vals), color=SUBTEXT, linestyle="--", linewidth=1, alpha=0.6)
    ax.text(years[-1] + 0.5, np.mean(vals), f"Mean: {np.mean(vals):.2f}",
            color=SUBTEXT, fontsize=8, va="center")

    ax.set_title("⚽ Avg Goals per Game by World Cup Tournament (1930–2022)",
                 fontsize=14, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("Year", labelpad=8)
    ax.set_ylabel("Goals per Game", labelpad=8)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    _apply_dark(fig)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 03. All 12 group predicted standings
# ═══════════════════════════════════════════════════════════════════════════════

def plot_group_standings(sim_results: dict, group: str | None = None) -> plt.Figure:
    """
    Bar chart of predicted finish probabilities for a group.
    sim_results: from monte_carlo.run_simulations() → {team: {1st:%, 2nd:%, ...}}
    group: if None, plot all 12 groups in a 3×4 grid.
    """
    from data import GROUPS, TEAM_GROUP

    if group:
        groups_to_plot = [group]
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        axes = [axes]
    else:
        groups_to_plot = sorted(GROUPS.keys())
        fig, axes = plt.subplots(3, 4, figsize=(20, 14))
        axes = axes.flatten()

    positions = [1, 2, 3, 4]
    pos_colors = [GOLD, SILVER, BRONZE, "#555555"]

    for idx, grp in enumerate(groups_to_plot):
        ax = axes[idx]
        teams = GROUPS.get(grp, [])

        if not teams:
            ax.set_visible(False)
            continue

        x = np.arange(len(teams))
        width = 0.2

        for pi, pos in enumerate(positions):
            vals = [sim_results.get(t, {}).get(pos, 0) for t in teams]
            ax.bar(x + pi * width, vals, width, label=f"{pos}{'st' if pos==1 else 'nd' if pos==2 else 'rd' if pos==3 else 'th'}",
                   color=pos_colors[pi], alpha=0.85)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(teams, rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Group {grp}", color=TEXT, fontsize=10, fontweight="bold")
        ax.set_ylabel("Probability", fontsize=7)
        ax.tick_params(labelsize=7)

    # Legend + title
    legend_patches = [mpatches.Patch(color=pos_colors[i], label=f"{p}{'st' if p==1 else 'nd' if p==2 else 'rd' if p==3 else 'th'}")
                      for i, p in enumerate(positions)]
    fig.legend(handles=legend_patches, loc="lower right", fontsize=9,
               framealpha=0.3, labelcolor=TEXT, facecolor=PANEL_BG)
    fig.suptitle("🏟  Predicted Group Stage Final Standings — WC 2026",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.01)
    _apply_dark(fig)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 04. Tournament win probabilities
# ═══════════════════════════════════════════════════════════════════════════════

def plot_win_probabilities(win_probs: dict, top_n: int = 20) -> plt.Figure:
    """
    Horizontal bar chart of tournament win probabilities for top-N teams.
    win_probs: {team: probability}
    """
    from data import TEAMS

    df = (pd.Series(win_probs)
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index())
    df.columns = ["Team", "P(Win)"]
    df["Pct"] = (df["P(Win)"] * 100).round(1)

    # Confederation colour
    conf_col = [CONF_COLORS.get(TEAMS.get(t, {}).get("confederation", ""), ACCENT)
                for t in df["Team"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df["Team"][::-1], df["Pct"][::-1],
                   color=conf_col[::-1], edgecolor=SUBTEXT, linewidth=0.4, height=0.7)

    for bar, val in zip(bars, df["Pct"][::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", ha="left", color=TEXT, fontsize=9)

    ax.set_xlim(0, df["Pct"].max() * 1.18)
    ax.set_title(f"🏆 Tournament Win Probability — Top {top_n} Teams\n(10,000 Monte Carlo simulations)",
                 fontsize=13, fontweight="bold", color=TEXT, pad=10)
    ax.set_xlabel("Win Probability (%)", labelpad=8)

    # Confederation legend
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in CONF_COLORS.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.3, labelcolor=TEXT, facecolor=PANEL_BG)

    _apply_dark(fig)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 04b. Round-by-round progression (Top-N teams)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_round_progression(probs: dict, top_n: int = 8) -> plt.Figure:
    """
    Line chart of P(reach round) through the knockout stages for the
    top-N teams by tournament win probability.

    probs: output of models.monte_carlo.run_simulations() —
           {team: {round: probability, ...}, ...} where round is one of
           ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"].

    "Group" is omitted (every team reaches it with probability 1.0 by
    construction, so it carries no information). Each remaining round is
    a strictly-decreasing "survival" probability — this view answers
    "where does each contender's run most likely end?" at a glance,
    which a single aggregate win-probability bar (Figure 04) cannot show.
    """
    from data import TEAMS

    rounds = ["R32", "R16", "QF", "SF", "Final", "Winner"]
    labels = ["Reach\nR32", "Reach\nR16", "Reach\nQF", "Reach\nSF", "Reach\nFinal", "Win\nTitle"]
    x = np.arange(len(rounds))

    ranked_teams = sorted(probs.keys(), key=lambda t: probs[t].get("Winner", 0.0), reverse=True)[:top_n]
    finals = [probs[t].get("Winner", 0.0) * 100 for t in ranked_teams]

    # De-overlap the end-of-line labels: push them apart in ascending order
    # of final value so closely-bunched teams (e.g. 3-9%) get readable labels.
    min_gap = 5.5
    order = sorted(range(len(ranked_teams)), key=lambda i: finals[i])
    label_y = [0.0] * len(ranked_teams)
    prev = -999.0
    for i in order:
        y = max(finals[i], prev + min_gap)
        label_y[i] = y
        prev = y

    fig, ax = plt.subplots(figsize=(11, 7))

    for team, y_final, y_label in zip(ranked_teams, finals, label_y):
        y = [probs[team].get(r, 0.0) * 100 for r in rounds]
        conf = TEAMS.get(team, {}).get("confederation", "")
        color = CONF_COLORS.get(conf, ACCENT)
        ax.plot(x, y, marker="o", linewidth=2.2, markersize=5.5, color=color,
                alpha=0.9, label=team)
        # Leader line from the actual data point to the (de-overlapped) label row
        if abs(y_label - y_final) > 0.05:
            ax.plot([x[-1], x[-1] + 0.25], [y_final, y_label],
                    color=color, linewidth=0.7, alpha=0.5)
        ax.text(x[-1] + 0.3, y_label, f"{team}  {y_final:.1f}%",
                va="center", ha="left", color=color,
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.4, len(rounds) - 1 + 2.6)
    ax.set_ylim(0, max(105, max(label_y) + 6))
    ax.set_ylabel("Probability (%)", labelpad=8)
    ax.set_title(f"Round-by-Round Progression — Top {top_n} Teams\n"
                 "(P(reach ≥ round), 10,000 Monte Carlo simulations)",
                 fontsize=13, fontweight="bold", color=TEXT, pad=10)
    ax.grid(axis="y", color=SUBTEXT, alpha=0.15)

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in CONF_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              framealpha=0.3, labelcolor=TEXT, facecolor=PANEL_BG)

    _apply_dark(fig)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 05. Scoreline heatmap (seaborn)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_scoreline_heatmap(
    home_team: str,
    away_team: str,
    home_elo: float,
    away_elo: float,
    max_goals: int = 6,
) -> plt.Figure:
    """
    Seaborn heatmap of scoreline probabilities.
    Uses Bivariate Poisson from models.poisson.
    """
    from models.poisson import scoreline_distribution

    xg_home = max(0.3, 1.35 * 10 ** ((home_elo - away_elo) / 800))
    xg_away = max(0.3, 1.35 * 10 ** ((away_elo - home_elo) / 800))
    dist = scoreline_distribution(round(xg_home, 2), round(xg_away, 2))

    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for (h, a), p in dist.items():
        if h <= max_goals and a <= max_goals:
            matrix[h][a] = p

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix * 100,
        annot=True, fmt=".1f",
        cmap=sns.color_palette("YlOrRd", as_cmap=True),
        linewidths=0.5, linecolor="#333",
        xticklabels=range(max_goals + 1),
        yticklabels=range(max_goals + 1),
        ax=ax,
        cbar_kws={"label": "Probability (%)"},
    )

    ax.set_xlabel(f"⬆ {away_team} goals", fontsize=11, color=TEXT, labelpad=8)
    ax.set_ylabel(f"⬅ {home_team} goals", fontsize=11, color=TEXT, labelpad=8)
    ax.set_title(
        f"📊 Scoreline Probability Heatmap\n{home_team} vs {away_team}\n"
        f"xG: {home_team} {xg_home:.2f} — {away_team} {xg_away:.2f}",
        fontsize=12, fontweight="bold", color=TEXT, pad=10,
    )
    ax.tick_params(colors=TEXT)
    ax.set_facecolor(PANEL_BG)
    fig.patch.set_facecolor(DARK_BG)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 06. Model comparison (Elo vs Logistic vs RF/XGBoost/Monte Carlo)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    home_team: str,
    away_team: str,
    elo_probs: dict,
    logistic_probs: dict,
    rf_probs: dict | None = None,
    xgb_probs: dict | None = None,
    mc_probs: dict | None = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing model win-probability predictions for one match.

    Each probs dict has keys {home_win, draw, away_win} (fractions, ~sum to 1):
      elo_probs       – models.poisson.match_result_probs_poisson()        (required)
      logistic_probs  – models.logistic.LogisticMatchPredictor.predict()   (required)
      rf_probs        – models.random_forest.RandomForestMatchPredictor.predict()  (optional)
      xgb_probs       – models.xgboost_model.XGBoostMatchPredictor.predict()       (optional)
      mc_probs        – empirical Monte Carlo win/draw/loss frequencies, e.g.
                         models.backtest._monte_carlo_match_probs() or your own
                         sampling over models.monte_carlo's scoreline distribution (optional)

    Pass None (or omit) for any optional model you don't have — e.g. if
    scikit-learn/xgboost aren't installed, or you didn't run a Monte Carlo
    pass for this match. Note RandomForestMatchPredictor / XGBoostMatchPredictor
    silently fall back to an Elo-based estimate when untrained (check their
    .trained attribute) — callers may prefer to omit rf_probs/xgb_probs in
    that case rather than show a near-duplicate Elo bar.
    """
    outcomes = ["Home Win", "Draw", "Away Win"]
    keys     = ["home_win", "draw", "away_win"]

    candidates = [
        ("Elo+Poisson",   elo_probs,      PALETTE[0]),
        ("Logistic Reg",  logistic_probs, PALETTE[1]),
        ("Random Forest", rf_probs,       PALETTE[2]),
        ("XGBoost",       xgb_probs,      PALETTE[3]),
        ("Monte Carlo",   mc_probs,       PALETTE[4]),
    ]
    models = [(name, probs, color) for name, probs, color in candidates if probs is not None]
    n = len(models)

    x = np.arange(3)
    w = 0.8 / n
    offsets = (np.arange(n) - (n - 1) / 2) * w

    fig, ax = plt.subplots(figsize=(8 + 1.3 * max(n - 2, 0), 5))
    all_vals = []
    for (name, probs, color), off in zip(models, offsets):
        vals = [probs.get(k, 0) * 100 for k in keys]
        all_vals += vals
        bars = ax.bar(x + off, vals, w * 0.92, label=name, color=color, edgecolor=SUBTEXT, lw=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom",
                    color=TEXT, fontsize=8 if n > 2 else 9)

    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, fontsize=11)
    ax.set_ylim(0, max(all_vals) * 1.2)
    ax.set_ylabel("Probability (%)", labelpad=8)
    ax.set_title(
        f"⚙ Model Comparison: {home_team} vs {away_team}",
        fontsize=13, fontweight="bold", color=TEXT, pad=10,
    )
    ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT, framealpha=0.5, ncol=min(n, 3))
    _apply_dark(fig)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 06b. Backtest accuracy (2018 & 2022 World Cups)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_backtest_results(backtest_results: dict) -> plt.Figure:
    """
    Bar chart of per-model prediction accuracy on the 2018 and 2022 World Cups,
    from models.backtest.run_full_backtest()'s output:
        {2018: {..., "models": {name: {"accuracy", "brier", "n"}, ...}},
         2022: {...}}

    Each year's entry may instead be {"year": y, "error": "..."} if historical
    match data couldn't be fetched (e.g. no network access in a sandbox).
    Handled gracefully:
      - if EVERY year errored, returns a figure showing the error message(s)
        in place of a chart (no crash)
      - if SOME years errored, charts the years that succeeded and adds a
        small annotation noting which years are missing and why
    """
    years = sorted(backtest_results.keys())
    valid = {y: r for y, r in backtest_results.items() if "models" in r}

    fig, ax = plt.subplots(figsize=(10, 5.5))

    if not valid:
        msg = "\n".join(
            f"WC {y}: {backtest_results[y].get('error', 'unknown error')}"
            for y in years
        )
        ax.text(0.5, 0.5, f"Backtest unavailable\n\n{msg}",
                ha="center", va="center", fontsize=11, color=TEXT,
                transform=ax.transAxes, wrap=True)
        ax.axis("off")
        _apply_dark(fig)
        plt.tight_layout()
        return fig

    model_names = list(next(iter(valid.values()))["models"].keys())
    x = np.arange(len(model_names))
    w = 0.8 / len(valid)

    for i, y in enumerate(sorted(valid)):
        accs = [valid[y]["models"][m]["accuracy"] * 100 for m in model_names]
        off = (i - (len(valid) - 1) / 2) * w
        bars = ax.bar(x + off, accs, w * 0.92, label=f"WC {y}",
                       color=PALETTE[i % len(PALETTE)], edgecolor=SUBTEXT, lw=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom",
                    color=TEXT, fontsize=8)

    errored = [y for y in years if y not in valid]
    if errored:
        msg = "; ".join(
            f"WC {y}: {backtest_results[y].get('error', 'unknown error')}"
            for y in errored
        )
        ax.text(0.5, -0.22, f"⚠ Not shown — {msg}", ha="center", va="top",
                transform=ax.transAxes, fontsize=8.5, color=SUBTEXT, wrap=True)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", labelpad=8)
    ax.set_title(
        "🎯 Backtest Accuracy — Point-in-Time Predictions\n"
        "(each model trained only on data available before that tournament)",
        fontsize=12, fontweight="bold", color=TEXT, pad=10,
    )
    ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT, framealpha=0.5)
    _apply_dark(fig)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 07. Knockout bracket
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_bracket_node(ax, x, y, text, color=PANEL_BG, width=2.2, height=0.5,
                        fontsize=7.5, text_color=TEXT):
    rect = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.05", linewidth=1,
        edgecolor=ACCENT, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold")


def plot_knockout_bracket(
    bracket: dict,
    title: str = "Predicted Knockout Bracket — WC 2026",
) -> plt.Figure:
    """
    Bracket tree visualization.
    bracket: dict with keys 'r32', 'r16', 'qf', 'sf', 'final', 'winner'
             each being a list of predicted winners (team names)
    e.g. bracket['r32'] = [32 team names in seeded order]

    Optional key 'third_place': (winner, loser) tuple for the 3rd-place
    playoff (match 103). If present, it's drawn as an extra annotation
    near the Final — build_bracket_from_simulation() sets this key.
    """
    fig, ax = plt.subplots(figsize=(22, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 33)
    ax.axis("off")
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Round x positions
    round_x = {"r32": 1.3, "r16": 3.8, "qf": 6.3, "sf": 8.8, "final": 11.3, "winner": 13.0}
    round_labels = {"r32": "Round of 32", "r16": "Round of 16", "qf": "Quarterfinals",
                    "sf": "Semifinals", "final": "Final", "winner": "🏆 Champion"}
    round_order = ["r32", "r16", "qf", "sf", "final", "winner"]
    for rnd, x in round_x.items():
        ax.text(x, 32.5, round_labels[rnd], ha="center", va="center",
                fontsize=9, color=ACCENT, fontweight="bold")

    def y_positions(n):
        return [1 + (32 / n) * i + (16 / n) - 0.5 for i in range(n)]

    # Draw each round (iterate over round_order, not bracket.items(), so extra
    # keys such as 'third_place' aren't treated as a round of team boxes)
    for rnd in round_order:
        teams = bracket.get(rnd)
        if not teams:
            continue
        x = round_x[rnd]
        ys = y_positions(len(teams))
        for team, y in zip(teams, ys):
            color = GOLD if rnd == "winner" else PANEL_BG
            text_c = DARK_BG if rnd == "winner" else TEXT
            _draw_bracket_node(ax, x, y, team, color=color, text_color=text_c,
                                fontsize=7.0 if len(teams) > 8 else 8.0)

    # Connect lines between rounds
    for ri in range(len(round_order) - 1):
        r_from = round_order[ri]
        r_to   = round_order[ri + 1]
        if not bracket.get(r_from) or not bracket.get(r_to):
            continue
        ys_from = y_positions(len(bracket[r_from]))
        ys_to   = y_positions(len(bracket[r_to]))
        x_from  = round_x[r_from] + 1.15
        x_to    = round_x[r_to]   - 1.15

        for i, y_to in enumerate(ys_to):
            # Connect pairs
            pair = [ys_from[i*2], ys_from[i*2 + 1]] if i*2+1 < len(ys_from) else [ys_from[i*2]]
            mid_y = np.mean(pair)
            for y_from in pair:
                ax.plot([x_from, x_from + 0.2, x_from + 0.2, x_to - 0.2, x_to - 0.2, x_to],
                        [y_from, y_from, mid_y, mid_y, y_to, y_to],
                        color=SUBTEXT, linewidth=0.6, alpha=0.5)

    # Optional 3rd-place playoff annotation (match 103) — not part of the
    # winner-advancement tree, so it's drawn as a standalone box rather than
    # via round_x/round_order.
    third_place = bracket.get("third_place")
    if third_place:
        tp_winner, tp_loser = third_place
        _draw_bracket_node(
            ax, round_x["final"], -0.5,
            f"🥉 3rd Place: {tp_winner} def. {tp_loser}",
            color=PANEL_BG, width=4.8, height=0.6, fontsize=8.5, text_color=GOLD,
        )

    ax.set_title(title, fontsize=15, fontweight="bold", color=TEXT,
                 y=1.02, x=0.5, transform=ax.transAxes)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 08. Host advantage analysis
# ═══════════════════════════════════════════════════════════════════════════════

def plot_host_advantage() -> plt.Figure:
    """
    Visualize the +100 Elo home advantage for USA, Mexico, Canada.
    Shows effective Elo boost and expected xG increase vs average opponent.
    """
    hosts = {
        "USA":    {"elo_base": 1820, "confederation": "CONCACAF"},
        "Mexico": {"elo_base": 1850, "confederation": "CONCACAF"},
        "Canada": {"elo_base": 1760, "confederation": "CONCACAF"},
    }
    avg_opp_elo = 1800  # typical WC group opponent
    boost = 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Elo comparison
    ax = axes[0]
    for i, (team, info) in enumerate(hosts.items()):
        base = info["elo_base"]
        adj  = base + boost
        ax.bar(i - 0.2, base, 0.35, color=ACCENT2, label="Base Elo" if i == 0 else "")
        ax.bar(i + 0.2, adj,  0.35, color=ACCENT,  label="With Home Boost" if i == 0 else "")
        ax.text(i, adj + 10, f"+{boost}", ha="center", color=GOLD, fontsize=9, fontweight="bold")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(list(hosts.keys()))
    ax.set_ylim(1700, 1980)
    ax.set_title("Elo Rating — Base vs Home-Adjusted", color=TEXT, fontsize=11)
    ax.set_ylabel("Elo Rating")
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT, framealpha=0.5)

    # Right: Win probability vs avg opponent
    ax = axes[1]
    for i, (team, info) in enumerate(hosts.items()):
        base = info["elo_base"]
        p_base = 1 / (1 + 10 ** ((avg_opp_elo - (base + 0))   / 400))
        p_adj  = 1 / (1 + 10 ** ((avg_opp_elo - (base + boost)) / 400))
        ax.bar(i - 0.2, p_base * 100, 0.35, color=ACCENT2, label="Without boost" if i == 0 else "")
        ax.bar(i + 0.2, p_adj  * 100, 0.35, color=ACCENT,  label="With +100 boost" if i == 0 else "")
        ax.text(i, p_adj * 100 + 0.5,
                f"+{(p_adj - p_base)*100:.1f}pp", ha="center", color=GOLD, fontsize=8)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(list(hosts.keys()))
    ax.set_title(f"Win Prob vs Avg Opponent (Elo {avg_opp_elo})", color=TEXT, fontsize=11)
    ax.set_ylabel("Win Probability (%)")
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT, framealpha=0.5)

    _apply_dark(fig)
    fig.suptitle("🏟  Host Nation Home Advantage — USA, Mexico, Canada",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.03)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: Generate predicted knockout bracket from sim results
# ═══════════════════════════════════════════════════════════════════════════════

def build_bracket_from_simulation(match_results: Dict[int, Tuple[str, str]]) -> dict:
    """
    Build a plot_knockout_bracket()-compatible dict from the REAL WC2026 Annex-C
    bracket simulation.

    Parameters
    ----------
    match_results : dict
        The second element returned by
        models.monte_carlo.simulate_full_tournament_detailed() — a
        {match_number: (winner, loser)} mapping for every knockout match,
        matches 73-104 (16 Round-of-32 matches, 8 Round-of-16, 4 quarterfinals,
        2 semifinals, the 3rd-place playoff (103), and the Final (104)).

    Returns
    -------
    dict with keys:
      'r32'    : 32 team names — match_results[m] (winner, loser) for each R32
                 match m, in data.knockout_fixtures' LEFT_R32_ORDER +
                 RIGHT_R32_ORDER order.
      'r16'    : 16 team names — winners of the 16 R32 matches, same order
                 (these are the entrants to the Round of 16).
      'qf'     : 8 team names — winners of matches 89,90,93,94,91,92,95,96.
      'sf'     : 4 team names — winners of matches 97,98,99,100.
      'final'  : 2 team names — winners of matches 101,102.
      'winner' : 1 team name — winner of match 104 (the Final).
      'third_place' : (winner, loser) of match 103, if present in
                 match_results — used by plot_knockout_bracket() to draw the
                 3rd-place playoff annotation.

    The list orderings above are derived from data.knockout_fixtures.BRACKET_TREE
    so that plot_knockout_bracket()'s adjacent-pair connecting lines (round i
    positions 2k/2k+1 -> round i+1 position k) reconstruct the real WC2026
    bracket tree — this is THE bracket that matches the dashboard's Bracket tab,
    unlike the simplified seeding from build_predicted_bracket() below.
    """
    from data.knockout_fixtures import LEFT_R32_ORDER, RIGHT_R32_ORDER

    r32_order = list(LEFT_R32_ORDER) + list(RIGHT_R32_ORDER)

    r32 = []
    for m in r32_order:
        r32.extend(match_results[m])  # (winner, loser) for this R32 match

    r16 = [match_results[m][0] for m in r32_order]
    qf  = [match_results[m][0] for m in (89, 90, 93, 94, 91, 92, 95, 96)]
    sf  = [match_results[m][0] for m in (97, 98, 99, 100)]
    final = [match_results[101][0], match_results[102][0]]
    winner = [match_results[104][0]]

    bracket = {
        "r32": r32,
        "r16": r16,
        "qf": qf,
        "sf": sf,
        "final": final,
        "winner": winner,
    }
    if 103 in match_results:
        bracket["third_place"] = match_results[103]
    return bracket


def build_predicted_bracket(win_probs: dict) -> dict:
    """
    Construct a SIMPLIFIED, ILLUSTRATIVE bracket by taking the top 32 teams by
    win probability and seeding them into a single-elimination tree where the
    higher seed always advances.

    ⚠️ THIS IS NOT THE DASHBOARD'S BRACKET. The live app's Bracket tab and its
    win probabilities come from models.monte_carlo.simulate_full_tournament_detailed(),
    which resolves the *real* WC2026 Round-of-32 pairings via
    data.knockout_fixtures (the official Annex C "best third-placed team" table,
    resolve_r32_pairing(), resolve_third_place_assignment(), walk_bracket()) and
    simulates every knockout match — including penalty shootouts — rather than
    always advancing the higher seed.

    Use this function only for a quick illustrative bracket figure (feeds
    plot_knockout_bracket) in notebooks/scripts that don't want to run a full
    tournament simulation. If you need a bracket that matches what the
    dashboard shows, call models.monte_carlo.simulate_full_tournament_detailed()
    and pass its match_results to build_bracket_from_simulation() instead.

    Returns dict suitable for plot_knockout_bracket().
    """
    sorted_teams = sorted(win_probs, key=win_probs.get, reverse=True)
    r32 = sorted_teams[:32]
    r16, qf, sf, final, winner = [], [], [], [], []

    def simulate_round(teams):
        """Pick winner of each pair by higher win probability (simplified —
        not a simulation, just 'higher seed always wins')."""
        winners = []
        for i in range(0, len(teams) - 1, 2):
            winners.append(teams[i])  # higher seed wins (simplified)
        return winners

    r16     = simulate_round(r32)
    qf      = simulate_round(r16)
    sf      = simulate_round(qf)
    final   = simulate_round(sf)
    winner  = [final[0]] if final else [r32[0]]

    return {
        "r32":    r32,
        "r16":    r16,
        "qf":     qf,
        "sf":     sf,
        "final":  final,
        "winner": winner,
    }
