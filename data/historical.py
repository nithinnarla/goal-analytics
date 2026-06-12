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
import unicodedata
from io import StringIO
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from data.teams import all_teams as _teams_py_all_teams

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

# ---------------------------------------------------------------------------
# Canonicalize this module's internal team names (the right-hand side of
# TEAM_NAME_MAP, or the raw martj42 name if unmapped) to the names used in
# data/teams.py — and therefore by the Elo presets, fixtures, and dashboard.
#
# Most of the ~46 WC2026 teams already line up. A couple don't:
#   - this module:  "USA"          <->  data/teams.py: "United States"
#   - this module:  "South Korea"  <->  data/teams.py: "Korea Republic"
#
# Without this, anything that blends preset Elo (keyed by data/teams.py
# names) with compute_elo_ratings()/build_form_lookup() output (keyed by
# this module's names) — e.g. GoalAnalytics_Analysis.py's blended_elos
# step — silently misses "United States" and "Korea Republic" and falls
# back to the preset value only, defeating the blend for those teams.
#
# compute_elo_ratings() and build_form_lookup() add an alias entry under
# the data/teams.py name alongside the existing (unchanged) entry — purely
# additive, nothing is renamed or removed.
# ---------------------------------------------------------------------------
TEAMS_PY_NAME_MAP = {
    # Confirmed: this module's TEAM_NAME_MAP target -> data/teams.py name
    "USA": "United States",
    "South Korea": "Korea Republic",
    # Defensive: raw source-name variants TEAM_NAME_MAP doesn't normalize,
    # in case they ever appear unmapped in the upstream CSV.
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}

_TEAMS_PY_TEAMS = set(_teams_py_all_teams())


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# Accent-insensitive fallback lookup (e.g. "Curaçao" -> "Curacao").
_TEAMS_PY_FOLDED = {_strip_accents(t).lower(): t for t in _TEAMS_PY_TEAMS}


def _canonical_team_name(name: str) -> str:
    """
    Map an internal team name (as produced by fetch_results(), after
    TEAM_NAME_MAP) to the corresponding name in data/teams.py, where one
    is known. Falls back to an accent-insensitive match against
    data/teams.py's 48 WC2026 team names, and finally to `name` unchanged.
    """
    if name in TEAMS_PY_NAME_MAP:
        return TEAMS_PY_NAME_MAP[name]
    if name in _TEAMS_PY_TEAMS:
        return name
    return _TEAMS_PY_FOLDED.get(_strip_accents(name).lower(), name)


def _add_teams_py_aliases(d: dict) -> dict:
    """
    Additively alias entries in `d` (keyed by this module's team names) to
    their data/teams.py equivalents, where that differs and isn't already
    present. Mutates and returns `d`.
    """
    for internal_name, value in list(d.items()):
        canon = _canonical_team_name(internal_name)
        if canon != internal_name and canon not in d:
            d[canon] = value
    return d


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

    # itertuples() is positional and has no Series.get() fallback, so make
    # sure the optional columns the loop relies on actually exist first
    # (mirrors the old row.get("neutral", False) / row.get("tournament", "")
    # defaults for sources that omit these columns).
    fill_cols = {}
    if "neutral" not in df.columns:
        fill_cols["neutral"] = False
    if "tournament" not in df.columns:
        fill_cols["tournament"] = ""
    if fill_cols:
        df = df.assign(**fill_cols)

    cols = ["home_team", "away_team", "home_score", "away_score", "neutral", "tournament"]
    elos: dict[str, float] = {}

    def get_e(team: str) -> float:
        return elos.get(team, float(initial_elo))

    def exp_score(elo_a: float, elo_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    # itertuples(name=None) yields plain tuples — roughly an order of
    # magnitude faster than iterrows() over ~25k+ rows.
    for home, away, home_score, away_score, neutral, tournament in df[cols].itertuples(index=False, name=None):
        hs, as_ = int(home_score), int(away_score)
        neutral = bool(neutral)
        tournament = str(tournament)

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

    # Additive aliases for data/teams.py naming (e.g. "USA" -> "United
    # States"), so callers that look Elo up by data/teams.py names — like
    # GoalAnalytics_Analysis.py's preset/historical blend — get a real
    # historical value instead of silently falling back to the preset.
    _add_teams_py_aliases(elos)

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


def _build_long_form(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Reshape `df` (one row per match, home/away columns) into long format —
    one row per team per match — restricted to matches strictly before
    `cutoff`, sorted chronologically. Used by build_form_lookup() to
    compute recent-form stats for every team in a couple of vectorized
    groupby passes instead of one get_recent_form() full-dataframe scan
    per team (O(n_teams × n_matches) -> O(n_matches)).
    """
    pre = df[df["date"] < cutoff]

    home = pre[["date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "scored", "away_score": "conceded"}
    )
    away = pre[["date", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "scored", "home_score": "conceded"}
    )
    long_df = pd.concat([home, away], ignore_index=True)
    long_df["pts"] = np.where(
        long_df["scored"] > long_df["conceded"], 3,
        np.where(long_df["scored"] == long_df["conceded"], 1, 0),
    )
    # Descending + later groupby().head(n) mirrors get_recent_form()'s
    # sort_values("date", ascending=False).head(n_matches) exactly,
    # including stable tie-breaking for same-date matches.
    return long_df.sort_values("date", ascending=False)


def build_form_lookup(
    df: pd.DataFrame | None = None,
    as_of: str = "2026-06-01",
    n: int = 10,
) -> dict[str, dict]:
    """
    Pre-compute recent form for all teams in the dataset (vectorized).
    Returns dict: team_name → form_dict
    """
    if df is None:
        df = fetch_results()
    if df is None:
        return {}

    cutoff = pd.to_datetime(as_of)
    teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    print(f"Computing recent form for {len(teams)} teams (last {n} matches before {as_of})...")

    default = {"form": 1.0, "avg_scored": 1.3, "avg_conceded": 1.1, "n_matches": 0}

    long_df = _build_long_form(df, cutoff)
    recent = long_df.groupby("team", sort=False).head(n)
    agg = recent.groupby("team")[["pts", "scored", "conceded"]].mean()
    counts = recent.groupby("team").size()

    lookup: dict[str, dict] = {}
    for t in teams:
        if t in agg.index:
            lookup[t] = {
                "form":         float(agg.loc[t, "pts"]),
                "avg_scored":   float(agg.loc[t, "scored"]),
                "avg_conceded": float(agg.loc[t, "conceded"]),
                "n_matches":    int(counts.loc[t]),
            }
        else:
            lookup[t] = default.copy()

    # Additive aliases for data/teams.py naming (e.g. "South Korea" ->
    # "Korea Republic"), matching compute_elo_ratings()'s aliasing so
    # callers can look form up by either naming convention.
    _add_teams_py_aliases(lookup)

    print(f"  ✓ Done.")
    return lookup
