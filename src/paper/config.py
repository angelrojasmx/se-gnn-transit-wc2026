"""
Central path and parameter configuration for the WC2026 transit paper pipeline.

All paths are derived from the repository root using Path(__file__).resolve()
so scripts run correctly from any working directory without hardcoded paths.
"""

from pathlib import Path

# Repository root and top-level directories
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs" / "paper"

# NYC inputs
NYC_HOURLY   = DATA_DIR / "nyc" / "raw" / "mta_manhattan_2022_2024.parquet"
NYC_LOOKUP   = DATA_DIR / "nyc" / "mta_manhattan_lookup.csv"

# Trained GNN backbone checkpoint
GNN_MODEL = ROOT / "outputs" / "nyc" / "manhattan_model.pt"

# Output directories
OUT_CDMX = OUTPUTS_DIR / "cdmx"
OUT_NYC  = OUTPUTS_DIR / "nyc"
OUT_VAN  = OUTPUTS_DIR / "van"
OUT_COMP = OUTPUTS_DIR / "comparative"
OUT_FIGS = OUTPUTS_DIR / "comparative" / "figures"


# WC2026 confirmed match schedule
# Source: FIFA / official host-city communication

WC2026_CDMX = [
    # (date, kickoff_local, matchup, stage)
    ("2026-06-11", "15:00", "Mexico vs South Africa",   "Group A - Opening"),
    ("2026-06-17", "22:00", "Uzbekistan vs Colombia",   "Group H"),
    ("2026-06-24", "21:00", "Mexico vs UEFA-D",         "Group A"),
    ("2026-06-30", "21:00", "R16 (Group A winner)",     "Round of 16"),
]

WC2026_NYC = [
    # (date, kickoff_ET, matchup, stage)
    ("2026-06-13", "18:00", "Brazil vs Morocco",  "Group E"),
    ("2026-06-16", "15:00", "France vs Senegal",  "Group B"),
    ("2026-06-22", "20:00", "Norway vs Senegal",  "Group B"),
    ("2026-06-25", "16:00", "Ecuador vs Germany", "Group F"),
    ("2026-06-27", "17:00", "Panama vs England",  "Group C"),
    ("2026-07-04", "TBD",   "Round of 32",        "Round of 32"),
    ("2026-07-11", "TBD",   "Round of 16",        "Round of 16"),
    ("2026-07-19", "15:00", "FINAL",              "Final"),
]

WC2026_VAN = [
    # (date, kickoff_PT, matchup, stage)
    ("2026-06-13", "21:00", "Australia vs Turkey",       "Group D"),
    ("2026-06-18", "15:00", "Canada vs Qatar",           "Group D"),
    ("2026-06-20", "TBD",   "New Zealand vs Egypt",      "Group G"),
    ("2026-06-24", "12:00", "Switzerland vs Canada",     "Group D"),
    ("2026-06-26", "TBD",   "New Zealand vs Belgium",    "Group G"),
    ("2026-07-02", "TBD",   "Round of 32",               "Round of 32"),
    ("2026-07-09", "TBD",   "Round of 16",               "Round of 16"),
]


# Historical analogs

# NFL MetLife (Giants/Jets) — secondary NYC analogs
# ~8 home games per team per season; listed dates are high-demand matches
# cross-referenced against MTA hourly ridership data
NFL_METLIFE_ANALOGS = [
    "2022-09-12",   # Giants Week 1
    "2022-10-30",   # Jets vs Patriots
    "2023-09-11",   # Giants opener
    "2023-10-29",   # Jets Halloween game
    "2024-09-08",   # Giants opener 2024
    "2024-11-03",   # Jets late season
]

# CDMX: Liga MX high-shock days (Estadio Ciudad de México / alternate venues)
# Used as training events for the CDMX spatial GNN
CDMX_LIGAMX_ANALOGS = [
    "2025-11-29",   # Saturday, z=5.30 STC Metro
    "2025-11-22",   # Saturday, z=3.84
    "2025-12-06",   # Saturday, z=3.86
]


# Station / corridor definitions

# NYC: Penn Station feeders to MetLife via NJ Transit
PENN_STATION_SIDS = [164, 318]   # 34 St-Penn Station (A,C,E) and (1,2,3)
HERALD_SQ_SID     = [607]        # 34 St-Herald Sq (B,D,F,M,N,Q,R,W)
PENN_CORRIDOR     = PENN_STATION_SIDS + HERALD_SQ_SID

HUDSON_YARDS_SID  = [471]        # 34 St-Hudson Yards (7) — secondary feeder

# Vancouver: primary SkyTrain gateway to BC Place
STADIUM_CHINATOWN_NAME = "Stadium"

# CDMX: modes included in multimodal ridership analysis
CDMX_MODES_KEY = [
    "STC_Metro", "Metrobus", "STE_Trolebus",
    "STE_Cablebus", "STE_TrenLigero", "RTP",
]

# CDMX: Metro and Metrobús lines with direct Estadio Azteca coverage
AZTECA_METRO_LINES    = ["Linea 2", "Línea 2", "Linea 3", "Línea 3",
                          "Linea 12", "Línea 12"]
AZTECA_METROBUS_LINES = ["Línea 2", "linea 2", "Línea 7", "linea 7"]


# Methodological parameters

POST_COVID_START = "2022-01-01"    # start of valid post-COVID baseline period

TOP_N_SHOCK_DAYS = 30              # number of top-demand days for uplift distribution

UPLIFT_PERCENTILES = (25, 50, 75)  # low / mid / high scenario percentiles

# NYC event travel window (local time):
# typical kickoffs 19:00-21:00 → outbound from Penn 17:00-23:00
NYC_EVENT_HOURS = list(range(14, 24))

# CDMX event windows (afternoon/evening kickoffs)
CDMX_EVENT_HOURS_AFTERNOON = list(range(12, 18))
CDMX_EVENT_HOURS_NIGHT     = list(range(18, 24))
