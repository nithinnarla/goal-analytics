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
  05. scoreline_heatmap      – Expected score grid for a match (seaborn heatmap)
  06. model_comparison       – Elo vs LogReg win probs for a sample match
  07. knockout_bracket       – Predicted R32→Final bracket tree
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
# 06. Model comparison (Elo vs Logistic)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    home_team: str,
    away_team: str,
    elo_probs: dict,
    logistic_probs: dict,
) -> plt.Figure:
    """
    Side-by-side grouped bar chart comparing Elo and Logistic regression predictions.
    probs: {home_win, draw, away_win}
    """
    outcomes = ["Home Win", "Draw", "Away Win"]
    keys     = ["home_win", "draw", "away_win"]
    elo_vals = [elo_probs.get(k, 0) * 100 for k in keys]
    log_vals = [logistic_probs.get(k, 0) * 100 for k in keys]

    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, elo_vals, w, label="Elo+Poisson", color=ACCENT,  edgecolor=SUBTEXT, lw=0.5)
    b2 = ax.bar(x + w/2, log_vals, w, label="Logistic Reg", color="#4db8ff", edgecolor=SUBTEXT, lw=0.5)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", color=TEXT, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, fontsize=11)
    ax.set_ylim(0, max(elo_vals + log_vals) * 1.2)
    ax.set_ylabel("Probability (%)", labelpad=8)
    ax.set_title(
        f"⚙ Model Comparison: {home_team} vs {away_team}",
        fontsize=13, fontweight="bold", color=TEXT, pad=10,
    )
    ax.legend(fontsize=10, facecolor=PANEL_BG, labelcolor=TEXT, framealpha=0.5)
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
    for rnd, x in round_x.items():
        ax.text(x, 32.5, round_labels[rnd], ha="center", va="center",
                fontsize=9, color=ACCENT, fontweight="bold")

    def y_positions(n):
        return [1 + (32 / n) * i + (16 / n) - 0.5 for i in range(n)]

    # Draw each round
    for rnd, teams in bracket.items():
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
    round_order = ["r32", "r16", "qf", "sf", "final", "winner"]
    for ri in range(len(round_order) - 1):
        r_from = round_order[ri]
        r_to   = round_order[ri + 1]
        if r_from not in bracket or r_to not in bracket:
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

def build_predicted_bracket(win_probs: dict) -> dict:
    """
    Construct a simplified predicted bracket by taking the top 32 teams
    by win probability and seeding them into a bracket in Elo order.

    Returns dict suitable for plot_knockout_bracket().
    """
    sorted_teams = sorted(win_probs, key=win_probs.get, reverse=True)
    r32 = sorted_teams[:32]
    r16, qf, sf, final, winner = [], [], [], [], []

    def simulate_round(teams):
        """Pick winner of each pair by higher win probability."""
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
