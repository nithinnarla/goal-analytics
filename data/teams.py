"""
World Cup 2026 team data — groups, FIFA rankings, and pre-computed Elo ratings.

Elo ratings are calibrated from:
  - FIFA/Coca-Cola World Ranking (June 2026 update)
  - Historical WC Elo databases (eloratings.net methodology)
  - Recent competitive results (2024-2026)

Home-advantage boost of +100 Elo applied for host nations
when playing in their own country.
"""

# ---------------------------------------------------------------------------
# 12 Groups — Group A to Group L
# ---------------------------------------------------------------------------
GROUPS = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Reverse lookup: team → group
TEAM_GROUP = {team: grp for grp, teams in GROUPS.items() for team in teams}

# ---------------------------------------------------------------------------
# Team metadata
# Format: name -> {elo, fifa_rank, confederation, flag}
# Elo baseline: ~1500 = average WC qualifier; ~2000+ = top-5 world
# ---------------------------------------------------------------------------
TEAMS = {
    # ── Group A ──────────────────────────────────────────────────────────────
    "Mexico": {
        "elo": 1820, "fifa_rank": 15, "confederation": "CONCACAF",
        "flag": "🇲🇽", "home_cities": ["Mexico City", "Guadalajara", "Monterrey"],
    },
    "South Africa": {
        "elo": 1560, "fifa_rank": 52, "confederation": "CAF",
        "flag": "🇿🇦", "home_cities": [],
    },
    "Korea Republic": {
        "elo": 1700, "fifa_rank": 22, "confederation": "AFC",
        "flag": "🇰🇷", "home_cities": [],
    },
    "Czechia": {
        "elo": 1660, "fifa_rank": 32, "confederation": "UEFA",
        "flag": "🇨🇿", "home_cities": [],
    },

    # ── Group B ──────────────────────────────────────────────────────────────
    "Canada": {
        "elo": 1790, "fifa_rank": 19, "confederation": "CONCACAF",
        "flag": "🇨🇦", "home_cities": ["Toronto", "Vancouver"],
    },
    "Bosnia and Herzegovina": {
        "elo": 1610, "fifa_rank": 42, "confederation": "UEFA",
        "flag": "🇧🇦", "home_cities": [],
    },
    "Qatar": {
        "elo": 1480, "fifa_rank": 68, "confederation": "AFC",
        "flag": "🇶🇦", "home_cities": [],
    },
    "Switzerland": {
        "elo": 1770, "fifa_rank": 21, "confederation": "UEFA",
        "flag": "🇨🇭", "home_cities": [],
    },

    # ── Group C ──────────────────────────────────────────────────────────────
    "Brazil": {
        "elo": 1970, "fifa_rank": 6, "confederation": "CONMEBOL",
        "flag": "🇧🇷", "home_cities": [],
    },
    "Morocco": {
        "elo": 1800, "fifa_rank": 14, "confederation": "CAF",
        "flag": "🇲🇦", "home_cities": [],
    },
    "Haiti": {
        "elo": 1430, "fifa_rank": 84, "confederation": "CONCACAF",
        "flag": "🇭🇹", "home_cities": [],
    },
    "Scotland": {
        "elo": 1630, "fifa_rank": 38, "confederation": "UEFA",
        "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "home_cities": [],
    },

    # ── Group D ──────────────────────────────────────────────────────────────
    "United States": {
        "elo": 1810, "fifa_rank": 16, "confederation": "CONCACAF",
        "flag": "🇺🇸", "home_cities": ["Los Angeles", "Seattle", "Kansas City"],
    },
    "Paraguay": {
        "elo": 1660, "fifa_rank": 34, "confederation": "CONMEBOL",
        "flag": "🇵🇾", "home_cities": [],
    },
    "Australia": {
        "elo": 1690, "fifa_rank": 25, "confederation": "AFC",
        "flag": "🇦🇺", "home_cities": [],
    },
    "Turkey": {
        "elo": 1680, "fifa_rank": 27, "confederation": "UEFA",
        "flag": "🇹🇷", "home_cities": [],
    },

    # ── Group E ──────────────────────────────────────────────────────────────
    "Germany": {
        "elo": 1940, "fifa_rank": 12, "confederation": "UEFA",
        "flag": "🇩🇪", "home_cities": [],
    },
    "Curacao": {
        "elo": 1460, "fifa_rank": 75, "confederation": "CONCACAF",
        "flag": "🇨🇼", "home_cities": [],
    },
    "Ivory Coast": {
        "elo": 1730, "fifa_rank": 20, "confederation": "CAF",
        "flag": "🇨🇮", "home_cities": [],
    },
    "Ecuador": {
        "elo": 1700, "fifa_rank": 26, "confederation": "CONMEBOL",
        "flag": "🇪🇨", "home_cities": [],
    },

    # ── Group F ──────────────────────────────────────────────────────────────
    "Netherlands": {
        "elo": 1950, "fifa_rank": 7, "confederation": "UEFA",
        "flag": "🇳🇱", "home_cities": [],
    },
    "Japan": {
        "elo": 1800, "fifa_rank": 17, "confederation": "AFC",
        "flag": "🇯🇵", "home_cities": [],
    },
    "Sweden": {
        "elo": 1670, "fifa_rank": 30, "confederation": "UEFA",
        "flag": "🇸🇪", "home_cities": [],
    },
    "Tunisia": {
        "elo": 1580, "fifa_rank": 47, "confederation": "CAF",
        "flag": "🇹🇳", "home_cities": [],
    },

    # ── Group G ──────────────────────────────────────────────────────────────
    "Belgium": {
        "elo": 1890, "fifa_rank": 9, "confederation": "UEFA",
        "flag": "🇧🇪", "home_cities": [],
    },
    "Egypt": {
        "elo": 1650, "fifa_rank": 35, "confederation": "CAF",
        "flag": "🇪🇬", "home_cities": [],
    },
    "Iran": {
        "elo": 1620, "fifa_rank": 39, "confederation": "AFC",
        "flag": "🇮🇷", "home_cities": [],
    },
    "New Zealand": {
        "elo": 1490, "fifa_rank": 66, "confederation": "OFC",
        "flag": "🇳🇿", "home_cities": [],
    },

    # ── Group H ──────────────────────────────────────────────────────────────
    "Spain": {
        "elo": 2060, "fifa_rank": 2, "confederation": "UEFA",
        "flag": "🇪🇸", "home_cities": [],
    },
    "Cape Verde": {
        "elo": 1530, "fifa_rank": 55, "confederation": "CAF",
        "flag": "🇨🇻", "home_cities": [],
    },
    "Saudi Arabia": {
        "elo": 1630, "fifa_rank": 37, "confederation": "AFC",
        "flag": "🇸🇦", "home_cities": [],
    },
    "Uruguay": {
        "elo": 1880, "fifa_rank": 10, "confederation": "CONMEBOL",
        "flag": "🇺🇾", "home_cities": [],
    },

    # ── Group I ──────────────────────────────────────────────────────────────
    "France": {
        "elo": 2050, "fifa_rank": 3, "confederation": "UEFA",
        "flag": "🇫🇷", "home_cities": [],
    },
    "Senegal": {
        "elo": 1780, "fifa_rank": 18, "confederation": "CAF",
        "flag": "🇸🇳", "home_cities": [],
    },
    "Iraq": {
        "elo": 1520, "fifa_rank": 62, "confederation": "AFC",
        "flag": "🇮🇶", "home_cities": [],
    },
    "Norway": {
        "elo": 1650, "fifa_rank": 33, "confederation": "UEFA",
        "flag": "🇳🇴", "home_cities": [],
    },

    # ── Group J ──────────────────────────────────────────────────────────────
    "Argentina": {
        "elo": 2100, "fifa_rank": 1, "confederation": "CONMEBOL",
        "flag": "🇦🇷", "home_cities": [],
    },
    "Algeria": {
        "elo": 1650, "fifa_rank": 36, "confederation": "CAF",
        "flag": "🇩🇿", "home_cities": [],
    },
    "Austria": {
        "elo": 1760, "fifa_rank": 23, "confederation": "UEFA",
        "flag": "🇦🇹", "home_cities": [],
    },
    "Jordan": {
        "elo": 1430, "fifa_rank": 87, "confederation": "AFC",
        "flag": "🇯🇴", "home_cities": [],
    },

    # ── Group K ──────────────────────────────────────────────────────────────
    "Portugal": {
        "elo": 1990, "fifa_rank": 5, "confederation": "UEFA",
        "flag": "🇵🇹", "home_cities": [],
    },
    "DR Congo": {
        "elo": 1540, "fifa_rank": 53, "confederation": "CAF",
        "flag": "🇨🇩", "home_cities": [],
    },
    "Uzbekistan": {
        "elo": 1480, "fifa_rank": 70, "confederation": "AFC",
        "flag": "🇺🇿", "home_cities": [],
    },
    "Colombia": {
        "elo": 1840, "fifa_rank": 11, "confederation": "CONMEBOL",
        "flag": "🇨🇴", "home_cities": [],
    },

    # ── Group L ──────────────────────────────────────────────────────────────
    "England": {
        "elo": 2010, "fifa_rank": 4, "confederation": "UEFA",
        "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "home_cities": [],
    },
    "Croatia": {
        "elo": 1840, "fifa_rank": 13, "confederation": "UEFA",
        "flag": "🇭🇷", "home_cities": [],
    },
    "Ghana": {
        "elo": 1590, "fifa_rank": 44, "confederation": "CAF",
        "flag": "🇬🇭", "home_cities": [],
    },
    "Panama": {
        "elo": 1480, "fifa_rank": 67, "confederation": "CONCACAF",
        "flag": "🇵🇦", "home_cities": [],
    },
}

# Home advantage Elo boost (applied per match when team is at home venue)
HOME_ELO_BOOST = 100

# Host nations and their host cities
HOST_NATIONS = {
    "Mexico": ["Mexico City", "Guadalajara", "Monterrey"],
    "United States": [
        "Los Angeles", "Seattle", "Kansas City", "Dallas",
        "Houston", "Miami", "Boston", "Atlanta", "Philadelphia",
        "New York/New Jersey", "San Francisco Bay Area",
    ],
    "Canada": ["Toronto", "Vancouver"],
}


def get_elo(team: str, venue_city: str | None = None) -> float:
    """Return Elo for a team, applying home boost if applicable."""
    base = TEAMS[team]["elo"]
    if venue_city and team in HOST_NATIONS:
        if any(city in venue_city for city in HOST_NATIONS[team]):
            return base + HOME_ELO_BOOST
    return float(base)


def all_teams() -> list[str]:
    return list(TEAMS.keys())
