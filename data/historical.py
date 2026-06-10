"""
Historical World Cup data fetcher.
Source: martj42/international_results (public GitHub, no auth needed)
  - All international results 1872-present
  - Used for: Elo calibration, recent form, historical analysis

Note: World Cup started in 1930 (Uruguay). We include all data from 1930 onwards.
There was no 1932 World Cup.
"""
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─── Public data sources (no authentication required) ────────────────────────
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
)

# ─── World Cup winners (hard-coded for reliability) ──────────────────────────
WC_WINNERS = {
    1930: "Uruguay",   1934: "Italy",     1938: "Italy",     1950: "Uruguay",
    1954: "Germany",   1958: "Brazil",    1962: "Brazil",    1966: "England",
    1970: "Brazil",    1974: "Germany",   1978: "Argentina", 1982: "Italy",
    1986: "Argentina", 1990: "Germany",   1994: "Brazil",    1998: "France",
    2002: "Brazil",    2006: "Italy",     2010: "Spain",     2014: "Germany",
    2018: "France",    2022: "Argentina",
}

WC_HOSTS = {
    1930: "Uruguay",   1934: "Italy",     1938: "France",    1950: "Brazil",
    1954: "Switzerland", 1958: "Sweden",  1962: "Chile",     1966: "England",
    1970: "Mexico",    1974: "Germany",   1978: "Argentina", 1982: "Spain",
    1986: "Mexico",    1990: "Italy",     1994: "USA",       1998: "France",
    2002: "South Korea/Japan", 2006: "Germany", 2010: "South Africa",
    2014: "Brazil",    2018: "Russia",    2022: "Qatar",     2026: "USA/Mexico/Canada",
}

# Name normalization: martj42 → our internal names
TEAM_NAME_MAP = {
    "United States":              "USA",
    "IR Iran":                    "Iran",
    "Korea Republic":             "South Korea",
    "China PR":                   "China",
    "DR Congo":                   "DR Congo",
    "Congo DR":                   "DR Congo",
    "Ivory Coast":                "Ivory Coast",
    "Cape Verde Islands":         "Cape Verde",
    "Trinidad and Tobago":        "Trinidad and Tobago",
    "Bosnia and Herzegovina":     "Bosnia and Herzegovina",
    "North Macedonia":            "North Macedonia",
    "Czech Republic":             "Czechia",
    "Kyrgyz Republic":            "Kyrgyzstan",
    "São Tomé e Príncipe":        "São Tomé & Príncipe",
    "St. Kitts & Nevis":          "St. Kitts and Nevis",
    "St. Lucia":                  "Saint Lucia",
    "St. Vincent & the Grenadines": "St. Vincent and the Grenadines",
}

# Cache
_results_cache = None


def fetch_results(force_refresh: bool = False) -> pd.DataFrame | None:
    """
    Fetch all international football results from martj42/international_results.
    Returns DataFrame with columns: date, home_team, away_team, home_score,
    away_score, tournament, city, country, neutral.
    Returns None if network unavailable.
    """
    global _results_cache
    if _results_cache is not None and not force_refresh:
        return _results_cache

    print("Fetching international results from GitHub...")
    try:
        resp = requests.get(RESULTS_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df["date"] = pd.to_datetime(df["date"])
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
        df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)
        df = df.dropna(subset=["home_score", "away_score"])
        _results_cache = df
        print(f"  ✓ {len(df):,} matches loaded ({df['date'].min().year}–{df['date'].max().year})")
        return df
    except Exception as e:
        print(f"  ✗ Network unavailable: {e}")
        print("  Using offline fallback (limited historical data).")
        return None


def get_wc_matches(df: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Filter to World Cup final-tournament matches only (not qualification)."""
    if df is None:
        df = fetch_results()
    if df is None:
        return None
    mask = df["tournament"] == "FIFA World Cup"
    wc = df[mask].copy().sort_values("date").reset_index(drop=True)
    print(f"  WC matches: {len(wc)} ({wc['date'].dt.year.min()}–{wc['date'].dt.year.max()})")
    return wc


def get_wc_stats(wc: pd.DataFrame | None = None) -> dict:
    """
    Return summary statistics for historical WC data.
    Returns dict with: matches_by_year, goals_by_year, avg_goals_by_year,
    wins_by_country, titles_by_country.
    """
    if wc is None:
        wc = get_wc_matches()

    stats = {
        "winners": WC_WINNERS,
        "titles": {},
        "matches_by_year": {},
        "goals_by_year": {},
        "avg_goals_by_year": {},
    }

    # Title counts
    for year, winner in WC_WINNERS.items():
        stats["titles"][winner] = stats["titles"].get(winner, 0) + 1

    if wc is None:
        return stats

    wc = wc.copy()
    wc["year"] = wc["date"].dt.year
    wc["total_goals"] = wc["home_score"] + wc["away_score"]

    by_year = wc.groupby("year")
    stats["matches_by_year"] = by_year.size().to_dict()
    stats["goals_by_year"] = by_year["total_goals"].sum().to_dict()
    stats["avg_goals_by_year"] = by_year["total_goals"].mean().round(2).to_dict()

    return stats


def compute_elo_ratings(
    df: pd.DataFrame | None = None,
    k_wc: float = 40.0,
    k_competitive: float = 25.0,
    k_friendly: float = 10.0,
    home_advantage: int = 100,
    initial_elo: int = 1500,
) -> dict[str, float]:
    """
    Compute Elo ratings for all teams by replaying historical results.

    K factors:
      - WC matches: 40
      - Other competitive (qualifiers, continental): 25
      - Friendlies: 10

    Goal margin factor: 1-goal = 1.0×K, 2-goal = 1.5×K, 3+ = 1.75×K
    """
    if df is None:
        df = fetch_results()
    if df is None:
        print("  Cannot compute Elo from history — using preset values from teams.py")
        return {}

    # Sort chronologically, exclude friendlies for speed if needed
    df = df.sort_values("date").reset_index(drop=True)
    elos: dict[str, float] = {}

    def get_e(team: str) -> float:
        return elos.get(team, float(initial_elo))

    def exp_score(elo_a: float, elo_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        neutral = bool(row.get("neutral", False))
        tournament = str(row.get("tournament", ""))

        # K factor
        if "World Cup" in tournament and "qualification" not in tournament.lower():
            k = k_wc
        elif "Friendly" in tournament or "friendly" in tournament:
            k = k_friendly
        else:
            k = k_competitive

        eh, ea = get_e(home), get_e(away)
        eh_adj = eh + (0 if neutral else home_advantage)

        exp_h = exp_score(eh_adj, ea)
        exp_a = 1.0 - exp_h

        if hs > as_:
            act_h, act_a = 1.0, 0.0
        elif hs < as_:
            act_h, act_a = 0.0, 1.0
        else:
            act_h, act_a = 0.5, 0.5

        # Goal margin multiplier
        gd = abs(hs - as_)
        m = 1.0 if gd <= 1 else (1.5 if gd == 2 else 1.75)

        elos[home] = eh + k * m * (act_h - exp_h)
        elos[away] = ea + k * m * (act_a - exp_a)

    return elos


def get_recent_form(
    team: str,
    as_of_date: datetime | None = None,
    n_matches: int = 10,
    df: pd.DataFrame | None = None,
) -> dict:
    """
    Return recent form for a team.
    Returns: {form (avg pts/game), avg_scored, avg_conceded, n_matches}
    """
    if df is None:
        df = fetch_results()
    if df is None:
        return {"form": 1.0, "avg_scored": 1.3, "avg_conceded": 1.1, "n_matches": 0}

    if as_of_date is None:
        as_of_date = datetime.now()
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date)

    mask = (df["home_team"] == team) | (df["away_team"] == team)
    matches = df[mask & (df["date"] < as_of_date)].copy()
    matches = matches.sort_values("date", ascending=False).head(n_matches)

    if len(matches) == 0:
        return {"form": 1.0, "avg_scored": 1.3, "avg_conceded": 1.1, "n_matches": 0}

    pts, scored, conceded = [], [], []
    for _, row in matches.iterrows():
        if row["home_team"] == team:
            gs, gc = int(row["home_score"]), int(row["away_score"])
        else:
            gs, gc = int(row["away_score"]), int(row["home_score"])
        scored.append(gs)
        conceded.append(gc)
        pts.append(3 if gs > gc else (1 if gs == gc else 0))

    return {
        "form":          float(np.mean(pts)),
        "avg_scored":    float(np.mean(scored)),
        "avg_conceded":  float(np.mean(conceded)),
        "n_matches":     len(matches),
    }


def build_form_lookup(
    df: pd.DataFrame | None = None,
    as_of: str = "2026-06-01",
    n: int = 10,
) -> dict[str, dict]:
    """
    Pre-compute recent form for all teams in the dataset (expensive but cached).
    Returns dict: team_name → form_dict
    """
    if df is None:
        df = fetch_results()
    if df is None:
        return {}

    cutoff = pd.to_datetime(as_of)
    teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    print(f"Computing recent form for {len(teams)} teams (last {n} matches before {as_of})...")
    lookup = {}
    for t in teams:
        lookup[t] = get_recent_form(t, as_of_date=cutoff, n_matches=n, df=df)
    print(f"  ✓ Done.")
    return lookup
