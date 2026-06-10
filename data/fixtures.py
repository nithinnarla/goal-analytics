"""
All 72 World Cup 2026 group-stage fixtures.
Dates are in US Eastern Time.  Kickoff stored as (date_str, time_et).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Fixture:
    match_id: str          # e.g. "A1", "B3"
    group: str
    home: str
    away: str
    date: str              # "2026-06-11"
    time_et: str           # "15:00"
    venue: str
    city: str
    # Results (filled in after the match)
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    @property
    def result(self) -> Optional[str]:
        if not self.played:
            return None
        if self.home_goals > self.away_goals:
            return "home"
        if self.away_goals > self.home_goals:
            return "away"
        return "draw"

    def label(self) -> str:
        return f"{self.home} vs {self.away}"


# ---------------------------------------------------------------------------
# All 72 group-stage fixtures (source: Yahoo Sports / simbye.com June 2026)
# ---------------------------------------------------------------------------
FIXTURES: list[Fixture] = [

    # ── GROUP A ──────────────────────────────────────────────────────────────
    Fixture("A1", "A", "Mexico", "South Africa",
            "2026-06-11", "15:00", "Estadio Azteca", "Mexico City"),
    Fixture("A2", "A", "Korea Republic", "Czechia",
            "2026-06-11", "22:00", "Estadio Akron", "Guadalajara"),
    Fixture("A3", "A", "Czechia", "South Africa",
            "2026-06-18", "12:00", "Mercedes-Benz Stadium", "Atlanta"),
    Fixture("A4", "A", "Mexico", "Korea Republic",
            "2026-06-18", "23:00", "Estadio Akron", "Guadalajara"),
    Fixture("A5", "A", "Czechia", "Mexico",
            "2026-06-24", "21:00", "Estadio Azteca", "Mexico City"),
    Fixture("A6", "A", "South Africa", "Korea Republic",
            "2026-06-24", "21:00", "Estadio BBVA", "Monterrey"),

    # ── GROUP B ──────────────────────────────────────────────────────────────
    Fixture("B1", "B", "Canada", "Bosnia and Herzegovina",
            "2026-06-12", "15:00", "BMO Field", "Toronto"),
    Fixture("B2", "B", "Qatar", "Switzerland",
            "2026-06-13", "15:00", "Levi's Stadium", "San Francisco Bay Area"),
    Fixture("B3", "B", "Switzerland", "Bosnia and Herzegovina",
            "2026-06-18", "15:00", "SoFi Stadium", "Los Angeles"),
    Fixture("B4", "B", "Canada", "Qatar",
            "2026-06-18", "18:00", "BC Place", "Vancouver"),
    Fixture("B5", "B", "Switzerland", "Canada",
            "2026-06-24", "15:00", "BC Place", "Vancouver"),
    Fixture("B6", "B", "Bosnia and Herzegovina", "Qatar",
            "2026-06-24", "15:00", "Lumen Field", "Seattle"),

    # ── GROUP C ──────────────────────────────────────────────────────────────
    Fixture("C1", "C", "Brazil", "Morocco",
            "2026-06-13", "18:00", "MetLife Stadium", "New York/New Jersey"),
    Fixture("C2", "C", "Haiti", "Scotland",
            "2026-06-13", "21:00", "Gillette Stadium", "Boston"),
    Fixture("C3", "C", "Scotland", "Morocco",
            "2026-06-19", "18:00", "Gillette Stadium", "Boston"),
    Fixture("C4", "C", "Brazil", "Haiti",
            "2026-06-19", "21:00", "Lincoln Financial Field", "Philadelphia"),
    Fixture("C5", "C", "Scotland", "Brazil",
            "2026-06-24", "18:00", "Hard Rock Stadium", "Miami"),
    Fixture("C6", "C", "Morocco", "Haiti",
            "2026-06-24", "18:00", "Mercedes-Benz Stadium", "Atlanta"),

    # ── GROUP D ──────────────────────────────────────────────────────────────
    Fixture("D1", "D", "United States", "Paraguay",
            "2026-06-12", "21:00", "SoFi Stadium", "Los Angeles"),
    Fixture("D2", "D", "Australia", "Turkey",
            "2026-06-13", "00:00", "BC Place", "Vancouver"),
    Fixture("D3", "D", "United States", "Australia",
            "2026-06-19", "15:00", "Lumen Field", "Seattle"),
    Fixture("D4", "D", "Turkey", "Paraguay",
            "2026-06-19", "23:00", "Levi's Stadium", "San Francisco Bay Area"),
    Fixture("D5", "D", "Turkey", "United States",
            "2026-06-25", "22:00", "SoFi Stadium", "Los Angeles"),
    Fixture("D6", "D", "Paraguay", "Australia",
            "2026-06-25", "22:00", "Levi's Stadium", "San Francisco Bay Area"),

    # ── GROUP E ──────────────────────────────────────────────────────────────
    Fixture("E1", "E", "Germany", "Curacao",
            "2026-06-14", "13:00", "NRG Stadium", "Houston"),
    Fixture("E2", "E", "Ivory Coast", "Ecuador",
            "2026-06-14", "19:00", "Lincoln Financial Field", "Philadelphia"),
    Fixture("E3", "E", "Germany", "Ivory Coast",
            "2026-06-20", "16:00", "BMO Field", "Toronto"),
    Fixture("E4", "E", "Ecuador", "Curacao",
            "2026-06-20", "20:00", "Arrowhead Stadium", "Kansas City"),
    Fixture("E5", "E", "Ecuador", "Germany",
            "2026-06-25", "16:00", "MetLife Stadium", "New York/New Jersey"),
    Fixture("E6", "E", "Curacao", "Ivory Coast",
            "2026-06-25", "16:00", "Lincoln Financial Field", "Philadelphia"),

    # ── GROUP F ──────────────────────────────────────────────────────────────
    Fixture("F1", "F", "Netherlands", "Japan",
            "2026-06-14", "16:00", "AT&T Stadium", "Dallas"),
    Fixture("F2", "F", "Sweden", "Tunisia",
            "2026-06-14", "22:00", "Estadio BBVA", "Monterrey"),
    Fixture("F3", "F", "Netherlands", "Sweden",
            "2026-06-20", "13:00", "NRG Stadium", "Houston"),
    Fixture("F4", "F", "Tunisia", "Japan",
            "2026-06-20", "00:00", "Estadio BBVA", "Monterrey"),
    Fixture("F5", "F", "Japan", "Sweden",
            "2026-06-25", "19:00", "AT&T Stadium", "Dallas"),
    Fixture("F6", "F", "Tunisia", "Netherlands",
            "2026-06-25", "19:00", "Arrowhead Stadium", "Kansas City"),

    # ── GROUP G ──────────────────────────────────────────────────────────────
    Fixture("G1", "G", "Belgium", "Egypt",
            "2026-06-15", "15:00", "Lumen Field", "Seattle"),
    Fixture("G2", "G", "Iran", "New Zealand",
            "2026-06-15", "21:00", "SoFi Stadium", "Los Angeles"),
    Fixture("G3", "G", "Belgium", "Iran",
            "2026-06-21", "15:00", "SoFi Stadium", "Los Angeles"),
    Fixture("G4", "G", "New Zealand", "Egypt",
            "2026-06-21", "21:00", "BC Place", "Vancouver"),
    Fixture("G5", "G", "Egypt", "Iran",
            "2026-06-26", "23:00", "Lumen Field", "Seattle"),
    Fixture("G6", "G", "New Zealand", "Belgium",
            "2026-06-26", "23:00", "BC Place", "Vancouver"),

    # ── GROUP H ──────────────────────────────────────────────────────────────
    Fixture("H1", "H", "Spain", "Cape Verde",
            "2026-06-15", "13:00", "Mercedes-Benz Stadium", "Atlanta"),
    Fixture("H2", "H", "Saudi Arabia", "Uruguay",
            "2026-06-15", "18:00", "Hard Rock Stadium", "Miami"),
    Fixture("H3", "H", "Spain", "Saudi Arabia",
            "2026-06-21", "12:00", "Mercedes-Benz Stadium", "Atlanta"),
    Fixture("H4", "H", "Uruguay", "Cape Verde",
            "2026-06-21", "18:00", "Hard Rock Stadium", "Miami"),
    Fixture("H5", "H", "Cape Verde", "Saudi Arabia",
            "2026-06-26", "20:00", "NRG Stadium", "Houston"),
    Fixture("H6", "H", "Uruguay", "Spain",
            "2026-06-26", "20:00", "Estadio Akron", "Guadalajara"),

    # ── GROUP I ──────────────────────────────────────────────────────────────
    Fixture("I1", "I", "France", "Senegal",
            "2026-06-16", "15:00", "MetLife Stadium", "New York/New Jersey"),
    Fixture("I2", "I", "Iraq", "Norway",
            "2026-06-16", "18:00", "Gillette Stadium", "Boston"),
    Fixture("I3", "I", "France", "Iraq",
            "2026-06-22", "17:00", "Lincoln Financial Field", "Philadelphia"),
    Fixture("I4", "I", "Norway", "Senegal",
            "2026-06-22", "20:00", "MetLife Stadium", "New York/New Jersey"),
    Fixture("I5", "I", "Norway", "France",
            "2026-06-26", "15:00", "Gillette Stadium", "Boston"),
    Fixture("I6", "I", "Senegal", "Iraq",
            "2026-06-26", "15:00", "BMO Field", "Toronto"),

    # ── GROUP J ──────────────────────────────────────────────────────────────
    Fixture("J1", "J", "Argentina", "Algeria",
            "2026-06-16", "21:00", "Arrowhead Stadium", "Kansas City"),
    Fixture("J2", "J", "Austria", "Jordan",
            "2026-06-16", "00:00", "Levi's Stadium", "San Francisco Bay Area"),
    Fixture("J3", "J", "Argentina", "Austria",
            "2026-06-22", "13:00", "AT&T Stadium", "Dallas"),
    Fixture("J4", "J", "Jordan", "Algeria",
            "2026-06-22", "23:00", "Levi's Stadium", "San Francisco Bay Area"),
    Fixture("J5", "J", "Algeria", "Austria",
            "2026-06-27", "22:00", "Arrowhead Stadium", "Kansas City"),
    Fixture("J6", "J", "Jordan", "Argentina",
            "2026-06-27", "22:00", "AT&T Stadium", "Dallas"),

    # ── GROUP K ──────────────────────────────────────────────────────────────
    Fixture("K1", "K", "Portugal", "DR Congo",
            "2026-06-17", "13:00", "NRG Stadium", "Houston"),
    Fixture("K2", "K", "Uzbekistan", "Colombia",
            "2026-06-17", "22:00", "Estadio Azteca", "Mexico City"),
    Fixture("K3", "K", "Portugal", "Uzbekistan",
            "2026-06-23", "13:00", "NRG Stadium", "Houston"),
    Fixture("K4", "K", "Colombia", "DR Congo",
            "2026-06-23", "22:00", "Estadio Akron", "Guadalajara"),
    Fixture("K5", "K", "Colombia", "Portugal",
            "2026-06-27", "19:30", "Hard Rock Stadium", "Miami"),
    Fixture("K6", "K", "DR Congo", "Uzbekistan",
            "2026-06-27", "19:30", "Mercedes-Benz Stadium", "Atlanta"),

    # ── GROUP L ──────────────────────────────────────────────────────────────
    Fixture("L1", "L", "England", "Croatia",
            "2026-06-17", "16:00", "AT&T Stadium", "Dallas"),
    Fixture("L2", "L", "Ghana", "Panama",
            "2026-06-17", "19:00", "BMO Field", "Toronto"),
    Fixture("L3", "L", "England", "Ghana",
            "2026-06-23", "16:00", "Gillette Stadium", "Boston"),
    Fixture("L4", "L", "Panama", "Croatia",
            "2026-06-23", "19:00", "BMO Field", "Toronto"),
    Fixture("L5", "L", "Panama", "England",
            "2026-06-27", "17:00", "MetLife Stadium", "New York/New Jersey"),
    Fixture("L6", "L", "Croatia", "Ghana",
            "2026-06-27", "17:00", "Lincoln Financial Field", "Philadelphia"),
]

# Quick lookups
FIXTURE_BY_ID = {f.match_id: f for f in FIXTURES}

def get_group_fixtures(group: str) -> list[Fixture]:
    return [f for f in FIXTURES if f.group == group]

def get_team_fixtures(team: str) -> list[Fixture]:
    return [f for f in FIXTURES if f.home == team or f.away == team]
