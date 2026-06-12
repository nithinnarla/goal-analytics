"""
Live FIFA World Ranking lookup for World Cup 2026 teams.

Scrapes the public FIFA World Ranking table published at
https://www.whereig.com/football/fifa-world-rankings.html (a freely
accessible mirror of the official FIFA/Coca-Cola World Ranking, ranks
1-211, updated each FIFA ranking cycle).

This module is purely ADDITIVE: it supplements (never replaces) the
static `TEAMS[team]["fifa_rank"]` values in data/teams.py, which were
hand-set alongside the Elo ratings used by the Poisson/Monte Carlo
simulation. The live numbers are surfaced for informational/comparison
purposes only (e.g. an info panel showing "model rank vs. live rank")
and do NOT feed back into Elo, expected goals, or simulation outcomes.

Design:
  - get_live_rankings()       -> {team: (rank, points)} for every WC2026
                                  team that could be matched on the page.
  - get_fifa_rankings_table()  -> per-team dict comparing static vs. live
                                  rank, for the dashboard's comparison view.
  - get_fifa_rank(team)        -> single convenience lookup, live rank if
                                  available else the static fifa_rank.

Caching: results are cached to a JSON file in the system temp directory
for CACHE_TTL_HOURS hours (default 24h — FIFA rankings update roughly
monthly, so this is generous headroom while keeping the dashboard snappy).
On any failure (network, parsing, missing team), falls back to a stale
cache if one exists, and ultimately to data/teams.py's static fifa_rank
for any team that can't be resolved. The scraper never raises — every
public function degrades gracefully to {} / static values.

Note: this module makes outbound HTTP requests, so live fetches only
work where the host has internet access (e.g. Streamlit Cloud). In
network-restricted environments, get_live_rankings() simply returns {}
(or a stale cache) and callers fall back to static ranks automatically.
"""

import json
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

from data.teams import TEAMS, all_teams

RANKINGS_URL = "https://www.whereig.com/football/fifa-world-rankings.html"
CACHE_PATH = Path(tempfile.gettempdir()) / "goalanalytics_fifa_rankings_cache.json"
CACHE_TTL_HOURS = 24
REQUEST_TIMEOUT = 15

# ---------------------------------------------------------------------------
# Name normalization: whereig.com naming -> data/teams.py naming, for the
# WC2026 teams whose names commonly differ between sources.
# ---------------------------------------------------------------------------
NAME_MAP: Dict[str, str] = {
    "USA": "United States",
    "US": "United States",
    "United States of America": "United States",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "Congo, DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Korea Republic": "Korea Republic",
    "South Korea": "Korea Republic",
    "Republic of Korea": "Korea Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
    "Curaçao": "Curacao",
    "Curacao": "Curacao",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}

_OUR_TEAMS = set(all_teams())


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# Accent/diacritic-insensitive lookup for our 48 team names, used as a
# fallback when an exact / NAME_MAP match isn't found (e.g. "Curaçao").
_OUR_TEAMS_FOLDED: Dict[str, str] = {_strip_accents(t).lower(): t for t in _OUR_TEAMS}


def _normalize_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", raw).strip()
    if name in NAME_MAP:
        return NAME_MAP[name]
    if name in _OUR_TEAMS:
        return name
    return _OUR_TEAMS_FOLDED.get(_strip_accents(name).lower(), name)


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _parse_html_tables(html: str) -> list:
    """
    Parse every <table> in `html` into {team: (rank, points)}, using a
    simple regex-based row scanner (no lxml/bs4 dependency required).

    A row qualifies if its first cell is an integer rank (1-300) and at
    least one later cell is a numeric points value. The team name is
    taken from the first non-numeric cell after the rank.
    """
    tables: list = []

    for table_html in re.findall(r"<table[^>]*>(.*?)</table>", html, re.S | re.I):
        parsed: Dict[str, Tuple[int, float]] = {}
        for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S | re.I):
            cells_raw = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.S | re.I)
            if len(cells_raw) < 3:
                continue

            cells = []
            for c in cells_raw:
                text = re.sub(r"<[^>]+>", " ", c)          # strip nested tags
                text = re.sub(r"&nbsp;|&#160;", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                cells.append(text)

            rank_str = cells[0]
            if not re.fullmatch(r"\d{1,3}", rank_str):
                continue
            rank = int(rank_str)
            if not (1 <= rank <= 300):
                continue

            # Team name: first cell after the rank that isn't purely numeric
            name_raw = None
            for c in cells[1:]:
                if c and not re.fullmatch(r"[\d.,+\-]+", c):
                    name_raw = c
                    break
            if not name_raw:
                continue

            # Points: last cell that looks numeric
            points = None
            for c in reversed(cells):
                m = re.fullmatch(r"[\d]{2,5}(?:\.\d+)?", c.replace(",", ""))
                if m:
                    points = float(m.group())
                    break
            if points is None:
                continue

            name = _normalize_name(name_raw)
            if name not in parsed or rank < parsed[name][0]:
                parsed[name] = (rank, points)

        if parsed:
            tables.append(parsed)

    return tables


def _fetch_live() -> Dict[str, Tuple[int, float]]:
    resp = requests.get(
        RANKINGS_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; GoalAnalytics/1.0; +https://goal-analytics-wc2026.streamlit.app)"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    tables = _parse_html_tables(resp.text)
    if not tables:
        return {}

    # Pick the table that covers the most of our 48 WC2026 teams.
    best = max(tables, key=lambda d: sum(1 for t in _OUR_TEAMS if t in d))
    return best if any(t in best for t in _OUR_TEAMS) else {}


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _load_cache(allow_stale: bool = False) -> Optional[Dict[str, Tuple[int, float]]]:
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text())
        fresh = (time.time() - data.get("fetched_at", 0)) <= CACHE_TTL_HOURS * 3600
        if fresh or allow_stale:
            return {k: (v[0], v[1]) for k, v in data.get("rankings", {}).items()}
    except (json.JSONDecodeError, OSError, KeyError, IndexError, TypeError):
        pass
    return None


def _save_cache(rankings: Dict[str, Tuple[int, float]]) -> None:
    try:
        CACHE_PATH.write_text(json.dumps({
            "fetched_at": time.time(),
            "rankings": {k: list(v) for k, v in rankings.items()},
        }))
    except OSError:
        pass  # best-effort cache; never fail the caller over this


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_live_rankings(force_refresh: bool = False) -> Dict[str, Tuple[int, float]]:
    """
    Return {team_name: (rank, points)} for as many WC2026 teams as could
    be scraped/cached, using data/teams.py naming. Returns {} if the
    source is unreachable/unparseable and no cache (fresh or stale)
    exists -- callers should treat that as "use static fifa_rank".
    """
    if not force_refresh:
        cached = _load_cache(allow_stale=False)
        if cached is not None:
            return cached

    try:
        live = _fetch_live()
    except Exception:
        live = {}

    if live:
        _save_cache(live)
        return live

    stale = _load_cache(allow_stale=True)
    return stale if stale is not None else {}


def get_fifa_rankings_table() -> Dict[str, dict]:
    """
    Per-team comparison table for all 48 WC2026 teams:
        {team: {"static_rank": int, "live_rank": int | None,
                "live_points": float | None, "delta": int | None}}

    `delta` = static_rank - live_rank (positive => live ranking is
    BETTER/higher than the model's static rank, i.e. the model may be
    underrating that team relative to the current live FIFA ranking).
    """
    live = get_live_rankings()
    table: Dict[str, dict] = {}
    for team in all_teams():
        static_rank = TEAMS[team]["fifa_rank"]
        if team in live:
            live_rank, live_points = live[team]
            table[team] = {
                "static_rank": static_rank,
                "live_rank": live_rank,
                "live_points": live_points,
                "delta": static_rank - live_rank,
            }
        else:
            table[team] = {
                "static_rank": static_rank,
                "live_rank": None,
                "live_points": None,
                "delta": None,
            }
    return table


def get_fifa_rank(team: str) -> int:
    """
    Convenience lookup: live FIFA rank if available, else the static
    fifa_rank from data/teams.py. Always returns an int.
    """
    live = get_live_rankings()
    if team in live:
        return live[team][0]
    return TEAMS[team]["fifa_rank"]
