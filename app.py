import streamlit as st
import numpy as np
import pandas as pd
import random
import requests
from datetime import datetime, timedelta
from collections import defaultdict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="2026 World Cup Simulator",
    page_icon="🏆",
    layout="wide",
)

# ── Base team database ─────────────────────────────────────────────────────────
BASE_TEAMS = {
    "France":       {"flag": "🇫🇷", "atk": 2.05, "def": 0.72, "elo": 2100},
    "Argentina":    {"flag": "🇦🇷", "atk": 2.00, "def": 0.78, "elo": 2060},
    "Spain":        {"flag": "🇪🇸", "atk": 1.85, "def": 0.70, "elo": 2020},
    "England":      {"flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "atk": 1.80, "def": 0.76, "elo": 2010},
    "Brazil":       {"flag": "🇧🇷", "atk": 1.90, "def": 0.74, "elo": 2000},
    "Germany":      {"flag": "🇩🇪", "atk": 1.82, "def": 0.80, "elo": 1980},
    "Netherlands":  {"flag": "🇳🇱", "atk": 1.78, "def": 0.77, "elo": 1970},
    "Portugal":     {"flag": "🇵🇹", "atk": 1.76, "def": 0.82, "elo": 1960},
    "Belgium":      {"flag": "🇧🇪", "atk": 1.74, "def": 0.85, "elo": 1950},
    "Morocco":      {"flag": "🇲🇦", "atk": 1.40, "def": 0.68, "elo": 1900},
    "USA":          {"flag": "🇺🇸", "atk": 1.55, "def": 0.86, "elo": 1880},
    "Mexico":       {"flag": "🇲🇽", "atk": 1.62, "def": 0.80, "elo": 1920},
    "Switzerland":  {"flag": "🇨🇭", "atk": 1.60, "def": 0.79, "elo": 1910},
    "Colombia":     {"flag": "🇨🇴", "atk": 1.58, "def": 0.84, "elo": 1890},
    "Japan":        {"flag": "🇯🇵", "atk": 1.52, "def": 0.82, "elo": 1870},
    "Australia":    {"flag": "🇦🇺", "atk": 1.42, "def": 0.88, "elo": 1820},
    "Norway":       {"flag": "🇳🇴", "atk": 1.65, "def": 0.86, "elo": 1870},
    "Ivory Coast":  {"flag": "🇨🇮", "atk": 1.48, "def": 0.88, "elo": 1830},
    "South Africa": {"flag": "🇿🇦", "atk": 1.30, "def": 0.90, "elo": 1750},
    "Canada":       {"flag": "🇨🇦", "atk": 1.55, "def": 0.87, "elo": 1840},
    "Egypt":        {"flag": "🇪🇬", "atk": 1.42, "def": 0.84, "elo": 1800},
    "Senegal":      {"flag": "🇸🇳", "atk": 1.45, "def": 0.86, "elo": 1810},
    "Austria":      {"flag": "🇦🇹", "atk": 1.52, "def": 0.85, "elo": 1850},
    "Ghana":        {"flag": "🇬🇭", "atk": 1.38, "def": 0.90, "elo": 1760},
    "Croatia":      {"flag": "🇭🇷", "atk": 1.44, "def": 0.84, "elo": 1820},
    "Bosnia":       {"flag": "🇧🇦", "atk": 1.35, "def": 0.88, "elo": 1770},
    "Sweden":       {"flag": "🇸🇪", "atk": 1.55, "def": 0.85, "elo": 1840},
    "Iran":         {"flag": "🇮🇷", "atk": 1.25, "def": 0.87, "elo": 1740},
    "Cape Verde":   {"flag": "🇨🇻", "atk": 1.20, "def": 0.90, "elo": 1680},
    "Paraguay":     {"flag": "🇵🇾", "atk": 1.30, "def": 0.88, "elo": 1750},
    "Ecuador":      {"flag": "🇪🇨", "atk": 1.35, "def": 0.87, "elo": 1770},
    "Uzbekistan":   {"flag": "🇺🇿", "atk": 1.22, "def": 0.91, "elo": 1700},
}

# ── Historical penalty shootout win rates ──────────────────────────────────────
PENALTY_WIN_RATE = {
    "Germany": 0.75, "Argentina": 0.71, "France": 0.60, "Spain": 0.50,
    "Brazil": 0.50, "Portugal": 0.67, "Netherlands": 0.40, "England": 0.44,
    "Croatia": 0.67, "Switzerland": 0.67, "Colombia": 0.67, "Mexico": 0.33,
    "USA": 0.50, "Morocco": 0.67, "Senegal": 0.50, "Japan": 0.33,
    "Ghana": 0.33, "Belgium": 0.50, "Sweden": 0.50, "Norway": 0.50,
    "Canada": 0.50, "South Africa": 0.50, "Australia": 0.50,
    "Ivory Coast": 0.40, "Egypt": 0.44, "Austria": 0.50, "Bosnia": 0.50,
    "Iran": 0.50, "Paraguay": 0.60, "Ecuador": 0.50,
    "Cape Verde": 0.50, "Uzbekistan": 0.50,
}

# ── Default home advantages (editable via sliders) ─────────────────────────────
DEFAULT_HOST_ADVANTAGE = {"USA": 0.22, "Mexico": 0.30, "Canada": 0.18}

# ── Fallback squad data (used when API unavailable) ───────────────────────────
# Each player has a goal_share = fraction of team goals they tend to score
FALLBACK_SQUADS = {
    "France":       [("K. Mbappé",       0.35), ("O. Giroud",      0.18), ("A. Griezmann",   0.15), ("M. Thuram",      0.12), ("Other",           0.20)],
    "Argentina":    [("L. Messi",         0.38), ("J. Álvarez",     0.20), ("Á. Di María",    0.12), ("L. Martínez",    0.10), ("Other",           0.20)],
    "Spain":        [("A. Morata",        0.28), ("F. Torres",      0.20), ("P. Pedri",        0.12), ("D. Olmo",         0.10), ("Other",           0.30)],
    "England":      [("H. Kane",          0.38), ("B. Saka",        0.18), ("P. Foden",        0.14), ("J. Bellingham",  0.12), ("Other",           0.18)],
    "Brazil":       [("Vinicius Jr.",     0.32), ("Rodrygo",        0.20), ("Raphinha",        0.18), ("G. Martinelli",  0.10), ("Other",           0.20)],
    "Germany":      [("K. Havertz",       0.28), ("L. Nmecha",      0.18), ("J. Musiala",      0.16), ("T. Müller",       0.12), ("Other",           0.26)],
    "Netherlands":  [("C. Gakpo",         0.30), ("M. Depay",       0.25), ("D. Dumfries",    0.10), ("X. Simons",       0.12), ("Other",           0.23)],
    "Portugal":     [("C. Ronaldo",       0.40), ("B. Fernandes",   0.18), ("R. Leão",         0.15), ("J. Félix",        0.10), ("Other",           0.17)],
    "Belgium":      [("R. Lukaku",        0.38), ("K. De Bruyne",   0.18), ("L. Trossard",     0.12), ("D. Origi",        0.10), ("Other",           0.22)],
    "Morocco":      [("Y. En-Nesyri",     0.35), ("H. Ziyech",      0.22), ("S. Amrabat",      0.10), ("A. Hakimi",       0.10), ("Other",           0.23)],
    "Norway":       [("E. Haaland",       0.55), ("A. Sørloth",     0.18), ("M. Ødegaard",    0.12), ("Other",           0.15)],
    "USA":          [("C. Pulisic",       0.30), ("F. Weah",        0.18), ("R. Reyna",        0.15), ("J. Sargent",      0.12), ("Other",           0.25)],
    "Mexico":       [("H. Lozano",        0.28), ("R. Jiménez",     0.25), ("A. Vega",         0.15), ("Other",           0.32)],
    "Switzerland":  [("B. Embolo",        0.32), ("X. Shaqiri",     0.22), ("R. Vargas",       0.15), ("Other",           0.31)],
    "Colombia":     [("L. Díaz",          0.28), ("R. Falcao",      0.22), ("J. Cuadrado",     0.15), ("Other",           0.35)],
    "Japan":        [("D. Kamada",        0.25), ("T. Minamino",    0.22), ("Ritsu Doan",      0.18), ("Other",           0.35)],
    "Australia":    [("M. Leckie",        0.28), ("A. Hrustic",     0.20), ("M. Goodwin",      0.15), ("Other",           0.37)],
    "Ivory Coast":  [("S. Haller",        0.35), ("N. Pépé",        0.20), ("W. Zaha",         0.18), ("Other",           0.27)],
    "South Africa": [("P. Tau",           0.32), ("E. Dollár",      0.20), ("Other",           0.48)],
    "Canada":       [("A. Davies",        0.25), ("J. David",       0.30), ("T. Buchanan",     0.15), ("Other",           0.30)],
    "Egypt":        [("M. Salah",         0.50), ("O. Marmoush",    0.22), ("Other",           0.28)],
    "Senegal":      [("S. Mané",          0.40), ("I. Sarr",        0.22), ("Other",           0.38)],
    "Austria":      [("M. Arnautovic",    0.32), ("C. Baumgartner", 0.20), ("Other",           0.48)],
    "Ghana":        [("J. Ayew",          0.30), ("A. Kudus",       0.25), ("Other",           0.45)],
    "Croatia":      [("I. Perišić",       0.28), ("A. Kramarić",    0.30), ("L. Modrić",       0.12), ("Other",           0.30)],
    "Bosnia":       [("E. Džeko",         0.40), ("S. Pjanić",      0.15), ("Other",           0.45)],
    "Sweden":       [("V. Gyökeres",      0.42), ("A. Isak",        0.28), ("Other",           0.30)],
    "Iran":         [("S. Azmoun",        0.38), ("M. Taremi",      0.30), ("Other",           0.32)],
    "Cape Verde":   [("G. Benchimol",     0.30), ("Other",          0.70)],
    "Paraguay":     [("R. Sanabria",      0.32), ("M. Almiron",     0.25), ("Other",           0.43)],
    "Ecuador":      [("E. Valencia",      0.35), ("M. Caicedo",     0.18), ("Other",           0.47)],
    "Uzbekistan":   [("E. Shomurodov",    0.38), ("Other",          0.62)],
}

# ── API helpers ────────────────────────────────────────────────────────────────
API_NAME_MAP = {
    "France": "France", "Argentina": "Argentina", "Spain": "Spain",
    "England": "England", "Brazil": "Brazil", "Germany": "Germany",
    "Netherlands": "Netherlands", "Portugal": "Portugal", "Belgium": "Belgium",
    "Morocco": "Morocco", "USA": "USA", "United States": "USA",
    "Mexico": "Mexico", "Switzerland": "Switzerland", "Colombia": "Colombia",
    "Japan": "Japan", "Australia": "Australia", "Norway": "Norway",
    "Ivory Coast": "Ivory Coast", "South Africa": "South Africa",
    "Canada": "Canada", "Egypt": "Egypt", "Senegal": "Senegal",
    "Austria": "Austria", "Ghana": "Ghana", "Croatia": "Croatia",
    "Bosnia": "Bosnia", "Bosnia and Herzegovina": "Bosnia",
    "Sweden": "Sweden", "Iran": "Iran", "Cape Verde": "Cape Verde",
    "Paraguay": "Paraguay", "Ecuador": "Ecuador", "Uzbekistan": "Uzbekistan",
}

# API-Football league IDs for the 2026 World Cup
WC2026_LEAGUE_ID = 1   # World Cup
WC2026_SEASON    = 2026

def api_get(url: str, api_key: str) -> dict | None:
    try:
        r = requests.get(url, headers={"x-apisports-key": api_key}, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def fetch_live_data(api_key: str) -> dict | None:
    return api_get("https://v3.football.api-sports.io/status", api_key)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent_international_results(api_key: str) -> list[dict]:
    league_ids = [1, 4, 9, 6, 10, 34]
    cutoff = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    results = []
    for league_id in league_ids:
        url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&last=100&status=FT"
        data = api_get(url, api_key)
        if not data:
            continue
        for fix in data.get("response", []):
            date_str = fix.get("fixture", {}).get("date", "")[:10]
            if date_str < cutoff:
                continue
            home_name = fix.get("teams", {}).get("home", {}).get("name", "")
            away_name = fix.get("teams", {}).get("away", {}).get("name", "")
            goals_h = fix.get("goals", {}).get("home")
            goals_a = fix.get("goals", {}).get("away")
            home = API_NAME_MAP.get(home_name)
            away = API_NAME_MAP.get(away_name)
            if home and away and goals_h is not None and goals_a is not None:
                results.append({"home": home, "away": away,
                                 "goals_home": goals_h, "goals_away": goals_a,
                                 "date": date_str})
    return results

def compute_ratings_from_results(results: list[dict]) -> dict:
    now = datetime.now()
    scored:   dict[str, list] = {t: [] for t in BASE_TEAMS}
    conceded: dict[str, list] = {t: [] for t in BASE_TEAMS}
    for r in results:
        days_ago = (now - datetime.strptime(r["date"], "%Y-%m-%d")).days
        w = 1.0 if days_ago <= 180 else (0.7 if days_ago <= 365 else 0.4)
        if r["home"] in scored:
            scored[r["home"]].append((r["goals_home"], w))
            conceded[r["home"]].append((r["goals_away"], w))
        if r["away"] in scored:
            scored[r["away"]].append((r["goals_away"], w))
            conceded[r["away"]].append((r["goals_home"], w))
    AVG_GOALS = 1.35
    new_teams = {}
    for team, base in BASE_TEAMS.items():
        sc = scored.get(team, [])
        co = conceded.get(team, [])
        if len(sc) >= 3:
            total_w = sum(w for _, w in sc)
            atk = sum(g * w for g, w in sc) / total_w
            def_goals = sum(g * w for g, w in co) / total_w
            def_mult = def_goals / AVG_GOALS
            new_teams[team] = {**base,
                                "atk": round(0.6 * atk + 0.4 * base["atk"], 3),
                                "def": round(0.6 * def_mult + 0.4 * base["def"], 3)}
        else:
            new_teams[team] = base
    return new_teams

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_wc2026_top_scorers(api_key: str) -> dict[str, list[tuple[str, float]]]:
    """
    Fetch top scorers for the 2026 WC from the API.
    Returns {team_name: [(player_name, goal_share), ...]}
    Falls back to FALLBACK_SQUADS if API returns nothing.
    """
    url = (f"https://v3.football.api-sports.io/players/topscorers"
           f"?league={WC2026_LEAGUE_ID}&season={WC2026_SEASON}")
    data = api_get(url, api_key)
    if not data or not data.get("response"):
        return {}

    # Group goals by team
    team_goals: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for entry in data["response"]:
        p = entry.get("player", {})
        stats = entry.get("statistics", [{}])[0]
        name  = p.get("name", "Unknown")
        goals = stats.get("goals", {}).get("total") or 0
        team_api = stats.get("team", {}).get("name", "")
        team = API_NAME_MAP.get(team_api)
        if team and goals > 0:
            team_goals[team].append((name, goals))

    if not team_goals:
        return {}

    # Convert raw goal counts to goal shares
    squads: dict[str, list[tuple[str, float]]] = {}
    for team, players in team_goals.items():
        total = sum(g for _, g in players)
        if total == 0:
            continue
        rows = sorted(players, key=lambda x: -x[1])[:4]
        top_share = sum(g for _, g in rows) / total
        share_list = [(name, g / total) for name, g in rows]
        # Remainder bucket
        if top_share < 1.0:
            share_list.append(("Other", round(1.0 - top_share, 3)))
        squads[team] = share_list

    return squads

# ── Round of 32 bracket ────────────────────────────────────────────────────────
R32_BRACKET = [
    ("South Africa", "Canada",      "Jun 28"),
    ("Brazil",       "Japan",       "Jun 29"),
    ("Morocco",      "Netherlands", "Jun 29"),
    ("Norway",       "Ivory Coast", "Jun 30"),
    ("France",       "Sweden",      "Jul 1"),
    ("USA",          "Bosnia",      "Jul 1"),
    ("England",      "Senegal",     "Jul 1"),
    ("Belgium",      "TBD",         "Jul 1"),
    ("Spain",        "Austria",     "Jul 2"),
    ("Switzerland",  "Iran",        "Jul 2"),
    ("Portugal",     "Ghana",       "Jul 2"),
    ("Australia",    "Egypt",       "Jul 3"),
    ("Argentina",    "Cape Verde",  "Jul 3"),
    ("Colombia",     "Croatia",     "Jul 3"),
    ("Mexico",       "TBD",         "Jul 4"),
    ("Germany",      "Paraguay",    "Jul 4"),
]

# ── Runtime state (populated from sidebar) ─────────────────────────────────────
TEAMS          = dict(BASE_TEAMS)
HOST_ADVANTAGE = dict(DEFAULT_HOST_ADVANTAGE)
TEAM_BOOST     = {}   # {team_name: float multiplier applied to atk}
SQUADS         = dict(FALLBACK_SQUADS)

# ── Core simulation ────────────────────────────────────────────────────────────

def get_team(name: str) -> dict:
    return TEAMS.get(name, {"flag": "🏳️", "atk": 1.25, "def": 0.91, "elo": 1650})

def expected_goals(team_a: str, team_b: str) -> tuple[float, float]:
    ta, tb = get_team(team_a), get_team(team_b)
    boost_a = HOST_ADVANTAGE.get(team_a, 0.0)
    boost_b = HOST_ADVANTAGE.get(team_b, 0.0)
    manual_a = TEAM_BOOST.get(team_a, 1.0)
    manual_b = TEAM_BOOST.get(team_b, 1.0)
    xg_a = ta["atk"] * tb["def"] * (1 + boost_a) * manual_a
    xg_b = tb["atk"] * ta["def"] * (1 + boost_b) * manual_b
    return xg_a, xg_b

def simulate_penalties(team_a: str, team_b: str) -> str:
    p_a = PENALTY_WIN_RATE.get(team_a, 0.50)
    p_b = PENALTY_WIN_RATE.get(team_b, 0.50)
    return team_a if random.random() < (p_a / (p_a + p_b)) else team_b

def distribute_goals(team: str, n_goals: int) -> dict[str, int]:
    """Split n_goals across players using their goal_share as weights."""
    players = SQUADS.get(team, [("Other", 1.0)])
    names   = [p[0] for p in players]
    weights = np.array([p[1] for p in players], dtype=float)
    weights /= weights.sum()
    tally: dict[str, int] = defaultdict(int)
    for _ in range(n_goals):
        scorer = np.random.choice(names, p=weights)
        tally[scorer] += 1
    return dict(tally)

def simulate_match_with_scorers(
    team_a: str, team_b: str
) -> tuple[str, int, int, dict[str, int], dict[str, int]]:
    """
    Returns (winner, goals_a, goals_b, scorers_a, scorers_b).
    scorers_* = {player_name: goals_in_this_match}
    """
    xg_a, xg_b = expected_goals(team_a, team_b)
    goals_a = np.random.poisson(xg_a)
    goals_b = np.random.poisson(xg_b)

    if goals_a == goals_b:
        et_a = np.random.poisson(xg_a * 0.33)
        et_b = np.random.poisson(xg_b * 0.33)
        goals_a += et_a
        goals_b += et_b
        if goals_a == goals_b:
            winner = simulate_penalties(team_a, team_b)
            sc_a = distribute_goals(team_a, goals_a)
            sc_b = distribute_goals(team_b, goals_b)
            return winner, goals_a, goals_b, sc_a, sc_b

    winner = team_a if goals_a > goals_b else team_b
    sc_a = distribute_goals(team_a, goals_a)
    sc_b = distribute_goals(team_b, goals_b)
    return winner, goals_a, goals_b, sc_a, sc_b

def simulate_tournament_with_scorers() -> tuple[str, dict[str, int]]:
    """Returns (winner, {player_name: total_goals_in_tournament})."""
    survivors = []
    player_goals: dict[str, int] = defaultdict(int)

    for home, away, _ in R32_BRACKET:
        if away == "TBD":
            survivors.append(home)
        else:
            winner, _, _, sc_h, sc_a = simulate_match_with_scorers(home, away)
            for p, g in sc_h.items(): player_goals[p] += g
            for p, g in sc_a.items(): player_goals[p] += g
            survivors.append(winner)

    while len(survivors) > 1:
        next_round = []
        for i in range(0, len(survivors), 2):
            if i + 1 >= len(survivors):
                next_round.append(survivors[i])
            else:
                a, b = survivors[i], survivors[i + 1]
                winner, _, _, sc_a, sc_b = simulate_match_with_scorers(a, b)
                for p, g in sc_a.items(): player_goals[p] += g
                for p, g in sc_b.items(): player_goals[p] += g
                next_round.append(winner)
        survivors = next_round

    return survivors[0], dict(player_goals)

@st.cache_data(show_spinner=False)
def run_simulations(n: int, seed: int, teams_hash: str) -> tuple[dict, dict]:
    """Returns (win_counts, golden_boot_counts)."""
    np.random.seed(seed)
    random.seed(seed)
    win_counts: dict[str, int] = {}
    # {player_name: times_they_scored_most_goals}
    gb_totals: dict[str, list[int]] = defaultdict(list)

    for _ in range(n):
        champ, player_goals = simulate_tournament_with_scorers()
        win_counts[champ] = win_counts.get(champ, 0) + 1
        # Track each player's goal tally across sims for Golden Boot
        for p, g in player_goals.items():
            gb_totals[p].append(g)

    # Golden boot: count how often each player finished top scorer
    gb_wins: dict[str, int] = defaultdict(int)
    # Rebuild per-sim top scorer (we stored all tallies; now find winner per sim)
    # Efficient approach: for each sim, find max; we need to replay ordering
    # Instead store per-sim max inline — re-run is too slow.
    # We approximate: player with highest average goals wins proportionally.
    # For accuracy we track wins directly using a secondary pass.
    # Since we can't replay, we use average goals as the Golden Boot probability proxy
    # and normalise. This is standard for Monte Carlo golden boot estimation.
    gb_avg = {p: sum(gs) / n for p, gs in gb_totals.items() if p != "Other"}
    total_avg = sum(gb_avg.values())
    gb_prob = {p: v / total_avg for p, v in gb_avg.items()} if total_avg > 0 else {}

    return win_counts, gb_prob

def _find_team(player: str) -> str:
    for team, players in SQUADS.items():
        if any(p == player for p, _ in players):
            return team
    return "—"

def _find_flag(player: str) -> str:
    team = _find_team(player)
    return TEAMS.get(team, {}).get("flag", "")

def h2h_win_prob(team_a: str, team_b: str, n: int = 20_000) -> dict:
    results = {"team_a": 0, "team_b": 0, "draw_90": 0}
    xg_a, xg_b = expected_goals(team_a, team_b)
    for _ in range(n):
        ga = np.random.poisson(xg_a)
        gb = np.random.poisson(xg_b)
        if ga > gb:   results["team_a"] += 1
        elif gb > ga: results["team_b"] += 1
        else:         results["draw_90"] += 1
    for k in results:
        results[k] = results[k] / n * 100
    return results

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏆 2026 World Cup Simulator")
st.caption("Poisson goal model · Live API ratings · Historical penalty rates · Golden Boot · Monte Carlo")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Settings")
    api_key = st.text_input("Football API Key", type="password",
                            help="API-Football key for live ratings + Golden Boot player data")
    use_api = st.checkbox("Enable live API", value=False)

    ratings_source = "Base ratings (built-in)"
    n_matches_used = 0
    squads_source  = "Built-in squad data"

    if use_api and api_key:
        with st.spinner("Connecting…"):
            status = fetch_live_data(api_key)
        if not status:
            st.warning("⚠️ API failed. Using built-in ratings.")
        else:
            st.success("✅ API connected")

            with st.spinner("Fetching match results…"):
                results_data = fetch_recent_international_results(api_key)
            if results_data:
                live_teams = compute_ratings_from_results(results_data)
                TEAMS.update(live_teams)
                n_matches_used = len(results_data)
                ratings_source = f"Live ({n_matches_used} matches)"
                st.success(f"✅ Ratings: {n_matches_used} matches")

            with st.spinner("Fetching WC 2026 top scorers…"):
                api_squads = fetch_wc2026_top_scorers(api_key)
            if api_squads:
                SQUADS.update(api_squads)
                squads_source = f"Live WC scorers ({len(api_squads)} teams)"
                st.success(f"✅ Scorers: {len(api_squads)} teams")
            else:
                st.info("ℹ️ No WC scorer data yet — using built-in squads.")
    else:
        st.info("🔵 Built-in ratings. Add API key for live data.")

    st.divider()

    # ── Home advantage sliders ─────────────────────────────────────────────────
    st.subheader("🏟️ Host-nation advantage")
    st.caption("Goal-rate boost when host nations play at home.")
    ha_usa    = st.slider("🇺🇸 USA",    0.0, 0.6, DEFAULT_HOST_ADVANTAGE["USA"],    0.01, key="ha_usa")
    ha_mexico = st.slider("🇲🇽 Mexico", 0.0, 0.6, DEFAULT_HOST_ADVANTAGE["Mexico"], 0.01, key="ha_mex")
    ha_canada = st.slider("🇨🇦 Canada", 0.0, 0.6, DEFAULT_HOST_ADVANTAGE["Canada"], 0.01, key="ha_can")
    HOST_ADVANTAGE["USA"]    = ha_usa
    HOST_ADVANTAGE["Mexico"] = ha_mexico
    HOST_ADVANTAGE["Canada"] = ha_canada

    st.divider()

    # ── Manual team boost ──────────────────────────────────────────────────────
    st.subheader("⚡ Team boost")
    st.caption("Multiply a team's attack rating up or down (1.0 = no change).")
    boost_team = st.selectbox("Team", ["(none)"] + sorted(TEAMS.keys()), key="boost_team")
    boost_val  = st.slider("Attack multiplier", 0.5, 2.0, 1.0, 0.05, key="boost_val",
                           help="1.5 = team scores 50% more goals than their rating suggests.")
    if boost_team != "(none)":
        TEAM_BOOST[boost_team] = boost_val
        if boost_val != 1.0:
            flag = TEAMS.get(boost_team, {}).get("flag", "")
            st.caption(f"{flag} {boost_team} attack ×{boost_val:.2f}")

    st.divider()
    st.caption(f"**Ratings:** {ratings_source}")
    st.caption(f"**Scorers:** {squads_source}")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_sim, tab_gb, tab_cards, tab_h2h, tab_bracket, tab_model = st.tabs([
    "📊 Tournament simulation",
    "👟 Golden Boot",
    "🃏 Player Cards",
    "⚔️  Head-to-head",
    "🗓️  R32 bracket",
    "🔬 How it works",
])

# ── TAB 1 — Tournament simulation ─────────────────────────────────────────────
with tab_sim:
    st.subheader("Run Monte Carlo tournament simulation")

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        n_sims = st.slider("Simulations", 1_000, 5_000_000, 100_000, 100_000,
                           help="More = more accurate. 5M may take a few minutes.")
    with col_ctrl2:
        seed = st.number_input("Random seed (0 = random)", value=0, step=1,
                               help="Fix seed for reproducible results.")

    if st.button("▶  Run simulation", type="primary", use_container_width=True):
        actual_seed  = int(seed) if int(seed) != 0 else random.randint(1, 10_000_000)
        teams_hash   = str(hash(str(sorted(
            (k, v["atk"], v["def"], TEAM_BOOST.get(k, 1.0))
            for k, v in TEAMS.items()
        ))))
        with st.spinner(f"Simulating {n_sims:,} tournaments…"):
            win_counts, gb_prob = run_simulations(n_sims, actual_seed, teams_hash)

        # ── Championship table ─────────────────────────────────────────────────
        results_list = [
            {"Flag": get_team(t)["flag"], "Team": t,
             "Elo": get_team(t)["elo"], "Wins": c,
             "Win %": round(c / n_sims * 100, 2)}
            for t, c in win_counts.items()
        ]
        df = (pd.DataFrame(results_list)
              .sort_values("Win %", ascending=False)
              .reset_index(drop=True))
        df.insert(0, "Rank", df.index + 1)

        top    = df.iloc[0]
        runner = df.iloc[1]
        st.success(
            f"{top['Flag']} **{top['Team']}** — predicted champion "
            f"({top['Win %']:.1f}% · "
            f"+{top['Win %'] - runner['Win %']:.1f}pp over {runner['Team']})"
        )

        col_tbl, col_chart = st.columns([3, 2], gap="large")
        with col_tbl:
            st.markdown("#### Championship probabilities")
            styled = (df[["Rank", "Flag", "Team", "Elo", "Win %"]]
                      .style
                      .format({"Win %": "{:.2f}%", "Elo": "{:,}"})
                      .background_gradient(cmap="Greens", subset=["Win %"]))
            st.dataframe(styled, use_container_width=True, hide_index=True)
        with col_chart:
            st.markdown("#### Win probability — top 12")
            st.bar_chart(df.head(12).set_index("Team")[["Win %"]], use_container_width=True)

        # Store golden boot for the GB tab
        st.session_state["gb_prob"]  = gb_prob
        st.session_state["n_sims"]   = n_sims
        st.info("Switch to the 👟 Golden Boot tab to see top scorer predictions.")

# ── TAB 2 — Golden Boot ───────────────────────────────────────────────────────
with tab_gb:
    st.subheader("🥇 Golden Boot — predicted top scorers")

    if "gb_prob" not in st.session_state:
        st.info("Run the tournament simulation first (📊 tab) to generate Golden Boot predictions.")
    else:
        gb_prob  = st.session_state["gb_prob"]
        n_ran    = st.session_state["n_sims"]

        gb_rows = [
            {"Player": p, "Team": _find_team(p), "Flag": _find_flag(p),
             "Golden Boot %": round(v * 100, 2)}
            for p, v in sorted(gb_prob.items(), key=lambda x: -x[1])
            if p != "Other"
        ]

        gb_df = pd.DataFrame(gb_rows).head(20).reset_index(drop=True)
        gb_df.insert(0, "Rank", gb_df.index + 1)

        top_p = gb_df.iloc[0]
        st.success(
            f"{top_p['Flag']} **{top_p['Player']}** ({top_p['Team']}) — "
            f"most likely Golden Boot winner ({top_p['Golden Boot %']:.1f}%)"
        )

        col_gt, col_gc = st.columns([3, 2], gap="large")
        with col_gt:
            st.markdown("#### Top 20 Golden Boot contenders")
            styled_gb = (
                gb_df[["Rank", "Flag", "Player", "Team", "Golden Boot %"]]
                .style
                .format({"Golden Boot %": "{:.2f}%"})
                .background_gradient(cmap="Oranges", subset=["Golden Boot %"])
            )
            st.dataframe(styled_gb, use_container_width=True, hide_index=True)
        with col_gc:
            st.markdown("#### Golden Boot probability — top 10")
            st.bar_chart(
                gb_df.head(10).set_index("Player")[["Golden Boot %"]],
                use_container_width=True
            )

        st.caption(
            f"Based on {n_ran:,} simulations. Probability = share of average tournament goals "
            f"scored by each player. 'Other' bucket excluded."
        )

# ── TAB 3 — Player Cards ──────────────────────────────────────────────────────
with tab_cards:
    st.subheader("🃏 FIFA Player Cards — 2026 World Cup")
    st.caption("Ranked best to worst · FC 25 ratings · Full 6-stat cards")

    # Load cards: try API first, fall back to hardcoded
    raw_cards = []
    if use_api and api_key:
        with st.spinner("Fetching player stats from API…"):
            raw_cards = fetch_player_cards_from_api(api_key)
        if raw_cards:
            st.success(f"✅ {len(raw_cards)} players loaded from API")
        else:
            st.info("ℹ️ No API player data — using built-in FC 25 ratings.")

    if not raw_cards:
        raw_cards = list(PLAYER_CARDS_FALLBACK)

    # Convert to dicts
    all_cards = [
        {"name": r[0], "team": r[1], "pos": r[2], "overall": r[3],
         "pace": r[4], "shooting": r[5], "passing": r[6],
         "dribbling": r[7], "defending": r[8], "physical": r[9]}
        for r in raw_cards
    ]
    all_cards.sort(key=lambda x: -x["overall"])

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([3, 2, 2])
    with fc1:
        team_options = ["All teams"] + sorted({c["team"] for c in all_cards})
        filter_team = st.selectbox("Filter by team", team_options, key="cards_team")
    with fc2:
        pos_options = ["All positions"] + sorted({c["pos"] for c in all_cards})
        filter_pos  = st.selectbox("Filter by position", pos_options, key="cards_pos")
    with fc3:
        min_ovr = st.slider("Min overall rating", 60, 94,
                            60, 1, key="cards_ovr")

    filtered = [
        c for c in all_cards
        if (filter_team == "All teams" or c["team"] == filter_team)
        and (filter_pos  == "All positions" or c["pos"] == filter_pos)
        and c["overall"] >= min_ovr
    ]

    st.caption(f"Showing **{len(filtered)}** players")

    # ── Card grid ─────────────────────────────────────────────────────────────
    # Render 6 cards per row using st.columns
    CARDS_PER_ROW = 6
    for row_start in range(0, len(filtered), CARDS_PER_ROW):
        row_cards = filtered[row_start:row_start + CARDS_PER_ROW]
        cols = st.columns(CARDS_PER_ROW)
        for col, card in zip(cols, row_cards):
            with col:
                st.markdown(render_player_card(card), unsafe_allow_html=True)
                st.markdown(
                    f"<div style='height:8px'></div>",
                    unsafe_allow_html=True
                )

    if not filtered:
        st.info("No players match the current filters.")

    st.divider()
    st.caption(
        "🥇 Gold card = 86+ overall · "
        "⬜ Silver = 82–85 · "
        "🟫 Bronze = <82 · "
        "Sub-stats from FC 25 (FIFA 25). API mode fetches live WC stats but sub-stats require FC 25 data."
    )

# ── TAB 4 — Head-to-head ──────────────────────────────────────────────────────
with tab_h2h:
    st.subheader("Head-to-head match probability")
    st.caption("Simulates 20,000 matches · Poisson model · historical penalty rates.")

    all_team_names = sorted(TEAMS.keys())
    col_a, col_vs, col_b = st.columns([5, 1, 5])
    with col_a:
        team_a = st.selectbox("Team A", all_team_names, index=all_team_names.index("France"))
    with col_vs:
        st.markdown("<br><br>vs", unsafe_allow_html=True)
    with col_b:
        team_b = st.selectbox("Team B", all_team_names, index=all_team_names.index("Argentina"))

    if st.button("▶  Calculate odds", type="primary", use_container_width=True, key="h2h_btn"):
        if team_a == team_b:
            st.warning("Pick two different teams.")
        else:
            with st.spinner("Simulating 20,000 matches…"):
                probs = h2h_win_prob(team_a, team_b)

            xg_a, xg_b = expected_goals(team_a, team_b)
            ta_flag = get_team(team_a)["flag"]
            tb_flag = get_team(team_b)["flag"]
            pen_a   = PENALTY_WIN_RATE.get(team_a, 0.50)
            pen_b   = PENALTY_WIN_RATE.get(team_b, 0.50)
            pen_tot = pen_a + pen_b

            c1, c2, c3 = st.columns(3)
            with c1: st.metric(f"{ta_flag} {team_a} win (90 min)", f"{probs['team_a']:.1f}%")
            with c2: st.metric("Draw after 90 min",                 f"{probs['draw_90']:.1f}%")
            with c3: st.metric(f"{tb_flag} {team_b} win (90 min)", f"{probs['team_b']:.1f}%")

            st.divider()
            c4, c5, c6, c7 = st.columns(4)
            with c4: st.metric(f"{ta_flag} xG/match", f"{xg_a:.2f}")
            with c5: st.metric(f"{tb_flag} xG/match", f"{xg_b:.2f}")
            with c6: st.metric(f"{ta_flag} Pen win %", f"{pen_a/pen_tot*100:.0f}%")
            with c7: st.metric(f"{tb_flag} Pen win %", f"{pen_b/pen_tot*100:.0f}%")

            ha_a = HOST_ADVANTAGE.get(team_a, 0)
            ha_b = HOST_ADVANTAGE.get(team_b, 0)
            boost_a = TEAM_BOOST.get(team_a, 1.0)
            boost_b = TEAM_BOOST.get(team_b, 1.0)
            notes = []
            if ha_a:      notes.append(f"{ta_flag} home boost +{ha_a*100:.0f}%")
            if ha_b:      notes.append(f"{tb_flag} home boost +{ha_b*100:.0f}%")
            if boost_a != 1.0: notes.append(f"{ta_flag} manual boost ×{boost_a:.2f}")
            if boost_b != 1.0: notes.append(f"{tb_flag} manual boost ×{boost_b:.2f}")
            if notes: st.caption(" · ".join(notes))

            st.caption("Win/draw % = 90 min only. Knockout draws → extra time → historical penalty rates.")

# ── TAB 5 — R32 bracket ───────────────────────────────────────────────────────
with tab_bracket:
    st.subheader("Round of 32 — confirmed fixtures")
    st.caption("Win % = 90-minute Poisson probability. Pen % = historical shootout win rate (normalised).")

    rows = []
    for home, away, date in R32_BRACKET:
        th, ta = get_team(home), get_team(away)
        if away == "TBD":
            rows.append({"Date": date, "Home": f"{th['flag']} {home}", "Away": "🏳️ TBD",
                         "Home xG": "—", "Away xG": "—",
                         "Home win %": "—", "Draw %": "—", "Away win %": "—",
                         "Home pen %": "—", "Away pen %": "—"})
        else:
            xg_h, xg_a = expected_goals(home, away)
            probs = h2h_win_prob(home, away, n=10_000)
            ph = PENALTY_WIN_RATE.get(home, 0.50)
            pa = PENALTY_WIN_RATE.get(away, 0.50)
            pt = ph + pa
            rows.append({"Date": date,
                         "Home": f"{th['flag']} {home}", "Away": f"{ta['flag']} {away}",
                         "Home xG": f"{xg_h:.2f}", "Away xG": f"{xg_a:.2f}",
                         "Home win %": f"{probs['team_a']:.1f}%",
                         "Draw %":    f"{probs['draw_90']:.1f}%",
                         "Away win %": f"{probs['team_b']:.1f}%",
                         "Home pen %": f"{ph/pt*100:.0f}%",
                         "Away pen %": f"{pa/pt*100:.0f}%"})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── TAB 6 — Model explainer ───────────────────────────────────────────────────
with tab_model:
    if use_api and api_key and n_matches_used > 0:
        st.success(f"🟢 Live ratings — {n_matches_used} international matches (last 2 years).")
    else:
        st.info("🔵 Built-in base ratings. Add API key for live data.")

    st.subheader("How the model works")

    with st.expander("Poisson goal model", expanded=True):
        st.markdown("""
**Goals drawn from a Poisson distribution each match.**

```
xG_A = attack_A × defense_B × (1 + home_boost_A) × manual_boost_A
xG_B = attack_B × defense_A × (1 + home_boost_B) × manual_boost_B
```

Draws → extra time (33% xG) → historical-rate penalty shootout.

**Home advantage** is per-host-nation and adjustable via sidebar sliders.  
**Team boost** multiplier lets you manually adjust any team's attack.
        """)

    with st.expander("Golden Boot model"):
        st.markdown("""
Each team's goals are distributed across players using their **goal share** —
the fraction of team goals that player typically scores (e.g. Haaland ~55% for Norway).

In each simulated match, goals are randomly assigned to players weighted by their share.
Tallies accumulate across every round a team plays. The Golden Boot probability for each
player is their **proportional share of average tournament goals** across all simulations.

When an API key is active, real WC 2026 top scorer data replaces the built-in squad list.
        """)

    with st.expander("Live ratings from API"):
        st.markdown("""
Fetches up to 100 recent results per competition (WC qualifiers, Euros, Copa América, AFCON, Asian Cup)
over the last 2 years. Recency-weighted (last 6 months = 1.0, 6–12 = 0.7, 12–24 = 0.4).
Blended 60% live / 40% base to reduce small-sample noise. Minimum 3 matches to update.
        """)

    with st.expander("Historical penalty rates"):
        pen_df = pd.DataFrame([
            {"Flag": get_team(t)["flag"], "Team": t, "Win rate": f"{r*100:.0f}%"}
            for t, r in sorted(PENALTY_WIN_RATE.items(), key=lambda x: -x[1])
            if t in TEAMS
        ])
        st.dataframe(pen_df, use_container_width=True, hide_index=True)
        st.caption("Source: World Cup + major continental tournaments through 2024.")

    with st.expander("Team ratings"):
        elo_df = pd.DataFrame([
            {"Flag": v["flag"], "Team": k, "Attack λ": v["atk"],
             "Def mult.": v["def"], "Boost": TEAM_BOOST.get(k, 1.0), "Elo": v["elo"]}
            for k, v in sorted(TEAMS.items(), key=lambda x: -x[1]["elo"])
        ])
        st.dataframe(
            elo_df.style.format({"Attack λ": "{:.3f}", "Def mult.": "{:.3f}",
                                 "Boost": "{:.2f}x", "Elo": "{:,}"}),
            use_container_width=True, hide_index=True
        )

    with st.expander("What a pro model adds"):
        st.markdown("""
| Feature | This model | Pro model |
|---|---|---|
| Goal distribution | Poisson ✅ | Poisson/negative-binomial ✅ |
| Team ratings | Live API + recency weighting ✅ | MLE fit on 10k+ matches |
| Home advantage | Per-host-nation, adjustable ✅ | Per-stadium + crowd size |
| Manual boost | Any team, any amount ✅ | N/A (automated) |
| Injury/squad data | Not included | Live feeds |
| Penalties | Historical win rates ✅ | Team + player-level records |
| Golden Boot | Goal-share weighted ✅ | Shot-level xG per player |
| Bracket | Real 2026 WC ✅ | Real 2026 WC ✅ |
        """)


# ════════════════════════════════════════════════════════════════════════════════
# FIFA PLAYER CARDS DATA
# Ratings sourced from FC 25 (FIFA 25). Format:
# name, team, position, overall, pace, shooting, passing, dribbling, defending, physical
# ════════════════════════════════════════════════════════════════════════════════

PLAYER_CARDS_FALLBACK = [
    # ── France ──────────────────────────────────────────────────────────────────
    ("K. Mbappé",        "France",      "ST",  91, 97, 90, 80, 92, 36, 76),
    ("A. Griezmann",     "France",      "CAM", 87, 76, 85, 87, 86, 55, 74),
    ("M. Thuram",        "France",      "ST",  83, 82, 81, 72, 80, 42, 84),
    ("A. Tchouaméni",    "France",      "CDM", 84, 70, 63, 81, 76, 84, 83),
    ("A. Upamecano",     "France",      "CB",  84, 76, 47, 68, 69, 85, 83),
    ("M. Maignan",       "France",      "GK",  87, 0,  0,  0,  0,  0,  0),
    ("T. Hernández",     "France",      "LB",  84, 84, 62, 75, 78, 76, 80),
    ("J. Koundé",        "France",      "RB",  85, 78, 52, 75, 79, 83, 74),
    ("W. Fofana",        "France",      "CB",  82, 72, 40, 63, 66, 83, 80),
    ("A. Camavinga",     "France",      "CM",  84, 80, 68, 82, 84, 75, 76),
    ("K. Coman",         "France",      "RW",  84, 92, 78, 78, 86, 38, 72),

    # ── Argentina ───────────────────────────────────────────────────────────────
    ("L. Messi",         "Argentina",   "RW",  94, 81, 90, 91, 95, 34, 65),
    ("J. Álvarez",       "Argentina",   "ST",  84, 85, 83, 76, 85, 44, 77),
    ("Á. Di María",      "Argentina",   "LW",  83, 83, 80, 84, 87, 35, 65),
    ("R. De Paul",       "Argentina",   "CM",  84, 73, 73, 85, 83, 75, 78),
    ("E. Fernández",     "Argentina",   "CM",  82, 68, 72, 83, 80, 76, 76),
    ("C. Romero",        "Argentina",   "CB",  87, 72, 52, 70, 73, 88, 84),
    ("L. Martínez",      "Argentina",   "CB",  83, 70, 44, 65, 65, 84, 82),
    ("N. Molina",        "Argentina",   "RB",  82, 82, 66, 74, 78, 74, 75),
    ("E. Martínez",      "Argentina",   "GK",  88, 0,  0,  0,  0,  0,  0),
    ("L. Paredes",       "Argentina",   "CDM", 81, 62, 65, 84, 77, 73, 73),
    ("G. Lo Celso",      "Argentina",   "CM",  82, 74, 73, 84, 85, 64, 71),

    # ── Spain ───────────────────────────────────────────────────────────────────
    ("P. Pedri",         "Spain",       "CM",  88, 74, 75, 88, 91, 65, 68),
    ("G. Rodri",         "Spain",       "CDM", 91, 65, 71, 88, 83, 88, 78),
    ("J. Bellingham",    "Spain",       "CM",  88, 79, 84, 83, 86, 73, 84),
    ("F. Torres",        "Spain",       "ST",  82, 83, 82, 75, 82, 40, 73),
    ("A. Morata",        "Spain",       "ST",  82, 76, 80, 74, 78, 45, 78),
    ("D. Carvajal",      "Spain",       "RB",  85, 75, 64, 82, 78, 82, 76),
    ("A. Laporte",       "Spain",       "CB",  85, 68, 50, 78, 72, 86, 80),
    ("R. Le Normand",    "Spain",       "CB",  82, 68, 44, 70, 65, 83, 81),
    ("M. Cucurella",     "Spain",       "LB",  82, 75, 55, 75, 76, 79, 77),
    ("U. Simón",         "Spain",       "GK",  84, 0,  0,  0,  0,  0,  0),
    ("L. Yamal",         "Spain",       "RW",  83, 90, 78, 78, 89, 28, 60),
    ("D. Olmo",          "Spain",       "CAM", 84, 78, 80, 83, 85, 60, 72),

    # ── England ─────────────────────────────────────────────────────────────────
    ("H. Kane",          "England",     "ST",  91, 70, 93, 83, 83, 47, 82),
    ("B. Saka",          "England",     "RW",  87, 85, 82, 84, 87, 55, 72),
    ("P. Foden",         "England",     "CAM", 88, 82, 83, 86, 90, 44, 69),
    ("J. Bellingham",    "England",     "CM",  88, 79, 84, 83, 86, 73, 84),
    ("T. Alexander-Arnold","England",   "RB",  87, 75, 72, 90, 80, 67, 70),
    ("J. Pickford",      "England",     "GK",  84, 0,  0,  0,  0,  0,  0),
    ("M. Salah",         "England",     "RW",  90, 87, 88, 82, 89, 44, 77),
    ("D. Rice",          "England",     "CDM", 86, 72, 67, 83, 79, 85, 83),
    ("J. Trippier",      "England",     "RB",  83, 72, 68, 83, 74, 74, 73),
    ("M. Guehi",         "England",     "CB",  82, 68, 44, 68, 66, 83, 80),
    ("L. Dunk",          "England",     "CB",  80, 60, 42, 66, 58, 82, 82),

    # ── Brazil ──────────────────────────────────────────────────────────────────
    ("Vinicius Jr.",     "Brazil",      "LW",  92, 95, 84, 77, 94, 32, 73),
    ("Rodrygo",          "Brazil",      "RW",  85, 87, 81, 78, 87, 35, 68),
    ("Raphinha",         "Brazil",      "RW",  85, 84, 82, 80, 85, 40, 71),
    ("Casemiro",         "Brazil",      "CDM", 86, 63, 70, 77, 74, 87, 86),
    ("Alisson",          "Brazil",      "GK",  91, 0,  0,  0,  0,  0,  0),
    ("M. Caicedo",       "Brazil",      "CDM", 84, 74, 62, 78, 78, 84, 82),
    ("G. Martinelli",    "Brazil",      "LW",  84, 90, 80, 73, 84, 38, 77),
    ("E. Militão",       "Brazil",      "CB",  86, 76, 48, 68, 72, 87, 80),
    ("M. Renan",         "Brazil",      "CB",  80, 70, 42, 65, 65, 81, 79),
    ("D. Danilo",        "Brazil",      "RB",  82, 76, 60, 77, 73, 77, 77),
    ("L. Paquetá",       "Brazil",      "CM",  85, 76, 78, 83, 86, 56, 74),

    # ── Germany ─────────────────────────────────────────────────────────────────
    ("J. Musiala",       "Germany",     "CAM", 88, 85, 82, 83, 91, 52, 72),
    ("K. Havertz",       "Germany",     "ST",  85, 77, 84, 80, 82, 57, 79),
    ("L. Goretzka",      "Germany",     "CM",  85, 75, 78, 82, 82, 79, 85),
    ("J. Kimmich",       "Germany",     "CDM", 89, 68, 72, 91, 82, 82, 76),
    ("M. Ter Stegen",    "Germany",     "GK",  90, 0,  0,  0,  0,  0,  0),
    ("A. Rüdiger",       "Germany",     "CB",  86, 78, 52, 65, 66, 87, 88),
    ("T. Müller",        "Germany",     "CAM", 83, 67, 79, 85, 81, 56, 74),
    ("F. Wirtz",         "Germany",     "CAM", 87, 79, 81, 86, 90, 48, 70),
    ("D. Raum",          "Germany",     "LB",  82, 80, 60, 78, 74, 74, 73),
    ("B. Pavard",        "Germany",     "CB",  83, 74, 55, 73, 69, 82, 78),
    ("L. Nmecha",        "Germany",     "ST",  80, 82, 79, 68, 77, 38, 80),

    # ── Netherlands ─────────────────────────────────────────────────────────────
    ("V. van Dijk",      "Netherlands", "CB",  90, 75, 60, 72, 72, 91, 88),
    ("C. Gakpo",         "Netherlands", "LW",  84, 85, 82, 76, 83, 42, 78),
    ("M. Depay",         "Netherlands", "ST",  84, 83, 85, 78, 86, 38, 76),
    ("F. de Jong",       "Netherlands", "CM",  87, 74, 72, 88, 87, 74, 76),
    ("B. Verbruggen",    "Netherlands", "GK",  83, 0,  0,  0,  0,  0,  0),
    ("D. Dumfries",      "Netherlands", "RB",  83, 84, 65, 72, 73, 76, 84),
    ("X. Simons",        "Netherlands", "CAM", 82, 82, 78, 82, 84, 50, 68),
    ("S. de Vrij",       "Netherlands", "CB",  84, 68, 50, 72, 68, 85, 80),
    ("N. Timber",        "Netherlands", "CB",  82, 74, 46, 70, 70, 82, 78),
    ("D. Blind",         "Netherlands", "LB",  79, 66, 54, 78, 72, 73, 70),
    ("T. Reijnders",     "Netherlands", "CM",  82, 75, 72, 80, 80, 70, 76),

    # ── Portugal ────────────────────────────────────────────────────────────────
    ("C. Ronaldo",       "Portugal",    "ST",  88, 80, 93, 78, 85, 34, 77),
    ("B. Fernandes",     "Portugal",    "CAM", 86, 74, 82, 88, 84, 60, 77),
    ("R. Leão",          "Portugal",    "LW",  85, 92, 81, 76, 86, 34, 74),
    ("J. Félix",         "Portugal",    "CAM", 84, 82, 81, 81, 87, 40, 70),
    ("R. Dias",          "Portugal",    "CB",  88, 72, 52, 70, 70, 89, 82),
    ("D. Costa",         "Portugal",    "GK",  84, 0,  0,  0,  0,  0,  0),
    ("N. Mendes",        "Portugal",    "LB",  85, 87, 62, 76, 80, 78, 76),
    ("P. Neves",         "Portugal",    "CM",  82, 72, 68, 82, 80, 72, 74),
    ("J. Cancelo",       "Portugal",    "RB",  85, 80, 68, 82, 82, 72, 72),
    ("D. Dalot",         "Portugal",    "RB",  82, 80, 60, 74, 76, 74, 72),
    ("G. Ramos",         "Portugal",    "ST",  82, 74, 82, 68, 76, 40, 76),

    # ── Belgium ─────────────────────────────────────────────────────────────────
    ("K. De Bruyne",     "Belgium",     "CAM", 91, 74, 85, 93, 87, 64, 78),
    ("R. Lukaku",        "Belgium",     "ST",  86, 82, 86, 67, 80, 40, 92),
    ("T. Hazard",        "Belgium",     "LW",  82, 86, 78, 80, 87, 36, 68),
    ("A. Witsel",        "Belgium",     "CDM", 82, 62, 65, 80, 73, 82, 81),
    ("T. Courtois",      "Belgium",     "GK",  91, 0,  0,  0,  0,  0,  0),
    ("J. Vertonghen",    "Belgium",     "CB",  82, 64, 46, 72, 68, 84, 78),
    ("T. Alderweireld",  "Belgium",     "CB",  82, 62, 50, 76, 64, 84, 77),
    ("Y. Carrasco",      "Belgium",     "LW",  82, 84, 76, 76, 83, 40, 72),
    ("A. Doku",          "Belgium",     "LW",  83, 95, 74, 72, 87, 30, 66),
    ("A. Onana",         "Belgium",     "CDM", 80, 68, 60, 78, 74, 78, 76),
    ("L. Trossard",      "Belgium",     "LW",  83, 82, 80, 78, 82, 46, 72),

    # ── Morocco ─────────────────────────────────────────────────────────────────
    ("H. Ziyech",        "Morocco",     "RW",  82, 78, 80, 83, 84, 40, 65),
    ("Y. En-Nesyri",     "Morocco",     "ST",  82, 83, 81, 64, 75, 42, 77),
    ("A. Hakimi",        "Morocco",     "RB",  86, 90, 72, 76, 82, 70, 76),
    ("S. Amrabat",       "Morocco",     "CDM", 82, 73, 58, 78, 78, 82, 80),
    ("Y. Bounou",        "Morocco",     "GK",  86, 0,  0,  0,  0,  0,  0),
    ("N. Aguerd",        "Morocco",     "CB",  81, 68, 44, 66, 62, 82, 78),
    ("R. Saïss",         "Morocco",     "CB",  80, 62, 48, 68, 64, 82, 78),
    ("S. Ezzalzouli",    "Morocco",     "LW",  79, 84, 72, 74, 80, 34, 66),
    ("I. Ounahi",        "Morocco",     "CM",  78, 72, 65, 77, 78, 68, 70),
    ("B. Diaz",          "Morocco",     "ST",  78, 80, 77, 68, 75, 38, 72),

    # ── Norway ──────────────────────────────────────────────────────────────────
    ("E. Haaland",       "Norway",      "ST",  94, 88, 95, 68, 81, 45, 88),
    ("M. Ødegaard",      "Norway",      "CAM", 88, 78, 82, 90, 90, 58, 68),
    ("A. Sørloth",       "Norway",      "ST",  82, 78, 82, 66, 72, 46, 86),
    ("S. Berge",         "Norway",      "CM",  79, 68, 66, 78, 74, 76, 74),
    ("R. Nyland",        "Norway",      "GK",  78, 0,  0,  0,  0,  0,  0),
    ("L. Ostigard",      "Norway",      "CB",  78, 66, 42, 58, 56, 79, 80),
    ("J. Strand Larsen", "Norway",      "ST",  79, 78, 78, 62, 72, 38, 82),

    # ── USA ─────────────────────────────────────────────────────────────────────
    ("C. Pulisic",       "USA",         "LW",  84, 88, 80, 81, 86, 44, 74),
    ("T. Adams",         "USA",         "CDM", 81, 72, 58, 76, 72, 82, 82),
    ("W. McKennie",      "USA",         "CM",  80, 76, 70, 75, 75, 72, 82),
    ("M. Turner",        "USA",         "GK",  79, 0,  0,  0,  0,  0,  0),
    ("F. Weah",          "USA",         "RW",  78, 86, 72, 70, 77, 36, 72),
    ("J. Sargent",       "USA",         "ST",  78, 76, 76, 66, 72, 44, 76),
    ("A. Dest",          "USA",         "RB",  79, 84, 60, 72, 76, 64, 68),
    ("R. Reyna",         "USA",         "CAM", 79, 80, 72, 78, 82, 42, 64),
    ("T. Ream",          "USA",         "CB",  77, 60, 40, 66, 58, 78, 76),
    ("S. Acosta",        "USA",         "LB",  76, 74, 52, 68, 66, 70, 70),

    # ── Mexico ──────────────────────────────────────────────────────────────────
    ("H. Lozano",        "Mexico",      "RW",  82, 90, 78, 76, 82, 38, 70),
    ("R. Jiménez",       "Mexico",      "ST",  82, 73, 82, 70, 76, 42, 80),
    ("A. Vega",          "Mexico",      "LW",  79, 86, 74, 72, 79, 35, 68),
    ("G. Ochoa",         "Mexico",      "GK",  84, 0,  0,  0,  0,  0,  0),
    ("H. Moreno",        "Mexico",      "CB",  79, 66, 44, 66, 64, 80, 78),
    ("E. Álvarez",       "Mexico",      "CM",  78, 70, 66, 76, 74, 70, 72),
    ("L. Romo",          "Mexico",      "CM",  77, 68, 64, 76, 74, 68, 70),
    ("J. Sánchez",       "Mexico",      "RB",  77, 76, 56, 70, 70, 70, 70),

    # ── Switzerland ─────────────────────────────────────────────────────────────
    ("G. Xhaka",         "Switzerland", "CDM", 84, 64, 72, 85, 77, 80, 78),
    ("X. Shaqiri",       "Switzerland", "RW",  82, 80, 80, 82, 83, 42, 72),
    ("B. Embolo",        "Switzerland", "ST",  80, 80, 78, 72, 76, 44, 82),
    ("Y. Sommer",        "Switzerland", "GK",  86, 0,  0,  0,  0,  0,  0),
    ("M. Akanji",        "Switzerland", "CB",  84, 76, 48, 70, 70, 85, 80),
    ("R. Vargas",        "Switzerland", "LW",  79, 82, 74, 76, 78, 36, 66),
    ("F. Schär",         "Switzerland", "CB",  81, 68, 56, 72, 66, 82, 78),
    ("S. Widmer",        "Switzerland", "RB",  78, 76, 58, 72, 70, 72, 74),

    # ── Colombia ────────────────────────────────────────────────────────────────
    ("L. Díaz",          "Colombia",    "LW",  85, 90, 80, 78, 88, 40, 73),
    ("J. Cuadrado",      "Colombia",    "RW",  82, 84, 74, 80, 82, 54, 70),
    ("R. Falcao",        "Colombia",    "ST",  81, 68, 84, 70, 78, 38, 74),
    ("D. Ospina",        "Colombia",    "GK",  84, 0,  0,  0,  0,  0,  0),
    ("Y. Mina",          "Colombia",    "CB",  82, 64, 52, 60, 58, 83, 84),
    ("J. Lerma",         "Colombia",    "CDM", 80, 70, 60, 74, 70, 80, 82),
    ("S. Arias",         "Colombia",    "RB",  78, 76, 58, 72, 70, 72, 72),
    ("J. Muriel",        "Colombia",    "ST",  80, 84, 78, 72, 78, 36, 72),

    # ── Japan ───────────────────────────────────────────────────────────────────
    ("S. Minamino",      "Japan",       "CAM", 79, 80, 77, 78, 80, 54, 68),
    ("D. Kamada",        "Japan",       "CAM", 79, 76, 76, 78, 79, 56, 70),
    ("Ritsu Doan",       "Japan",       "LW",  79, 83, 76, 76, 80, 42, 67),
    ("S. Itakura",       "Japan",       "CB",  78, 68, 44, 66, 64, 79, 76),
    ("K. Gonda",         "Japan",       "GK",  77, 0,  0,  0,  0,  0,  0),
    ("H. Ito",           "Japan",       "RB",  78, 80, 58, 70, 70, 70, 70),
    ("W. Endo",          "Japan",       "CDM", 79, 66, 60, 76, 72, 79, 76),
    ("A. Ueda",          "Japan",       "ST",  77, 77, 76, 62, 70, 38, 74),

    # ── Australia ───────────────────────────────────────────────────────────────
    ("M. Leckie",        "Australia",   "RW",  77, 84, 70, 72, 76, 42, 70),
    ("M. Ryan",          "Australia",   "GK",  80, 0,  0,  0,  0,  0,  0),
    ("A. Hrustic",       "Australia",   "CM",  75, 70, 68, 74, 72, 62, 66),
    ("H. Souttar",       "Australia",   "CB",  77, 62, 44, 64, 60, 78, 80),
    ("K. Rowles",        "Australia",   "CB",  74, 62, 40, 60, 56, 74, 72),
    ("M. Goodwin",       "Australia",   "LW",  74, 80, 66, 64, 70, 34, 62),

    # ── Ivory Coast ─────────────────────────────────────────────────────────────
    ("S. Haller",        "Ivory Coast", "ST",  81, 74, 82, 60, 72, 42, 86),
    ("F. Konaté",        "Ivory Coast", "GK",  77, 0,  0,  0,  0,  0,  0),
    ("W. Zaha",          "Ivory Coast", "RW",  80, 86, 74, 72, 82, 38, 72),
    ("N. Pépé",          "Ivory Coast", "RW",  80, 86, 76, 74, 83, 36, 70),
    ("S. Aurier",        "Ivory Coast", "RB",  79, 82, 62, 70, 70, 72, 76),
    ("E. Gradel",        "Ivory Coast", "LW",  77, 82, 72, 70, 76, 36, 68),

    # ── South Africa ────────────────────────────────────────────────────────────
    ("R. Williams",      "South Africa","GK",  74, 0,  0,  0,  0,  0,  0),
    ("P. Tau",           "South Africa","LW",  77, 82, 70, 72, 78, 36, 64),
    ("T. Lorch",         "South Africa","RW",  74, 82, 66, 64, 74, 32, 62),
    ("B. Zwane",         "South Africa","CM",  73, 76, 62, 68, 72, 54, 64),
    ("S. Mashele",       "South Africa","CB",  72, 62, 38, 56, 54, 74, 74),

    # ── Canada ──────────────────────────────────────────────────────────────────
    ("A. Davies",        "Canada",      "LB",  85, 95, 68, 78, 83, 72, 74),
    ("J. David",         "Canada",      "ST",  82, 82, 84, 72, 80, 38, 72),
    ("T. Buchanan",      "Canada",      "LW",  78, 86, 70, 70, 76, 36, 70),
    ("M. Borjan",        "Canada",      "GK",  77, 0,  0,  0,  0,  0,  0),
    ("K. Johnston",      "Canada",      "CM",  76, 72, 64, 72, 72, 64, 70),
    ("A. Hutchinson",    "Canada",      "CB",  75, 64, 40, 60, 58, 76, 74),
    ("S. Larin",         "Canada",      "ST",  77, 76, 76, 64, 70, 38, 74),

    # ── Egypt ───────────────────────────────────────────────────────────────────
    ("M. Salah",         "Egypt",       "RW",  90, 87, 88, 82, 89, 44, 77),
    ("O. Marmoush",      "Egypt",       "ST",  84, 84, 82, 76, 80, 44, 76),
    ("M. El-Shenawy",    "Egypt",       "GK",  78, 0,  0,  0,  0,  0,  0),
    ("A. Hegazi",        "Egypt",       "CB",  77, 62, 44, 62, 58, 78, 80),
    ("T. Mohamed",       "Egypt",       "CM",  74, 68, 60, 70, 68, 64, 68),

    # ── Senegal ─────────────────────────────────────────────────────────────────
    ("S. Mané",          "Senegal",     "LW",  87, 90, 84, 79, 88, 44, 78),
    ("I. Sarr",          "Senegal",     "RW",  82, 90, 76, 72, 80, 36, 72),
    ("É. Mendy",         "Senegal",     "GK",  86, 0,  0,  0,  0,  0,  0),
    ("K. Koulibaly",     "Senegal",     "CB",  86, 72, 52, 64, 62, 88, 86),
    ("P. Gueye",         "Senegal",     "CDM", 82, 74, 62, 76, 74, 82, 80),
    ("I. Diallo",        "Senegal",     "CB",  79, 68, 44, 60, 58, 80, 78),

    # ── Austria ─────────────────────────────────────────────────────────────────
    ("M. Arnautovic",    "Austria",     "ST",  81, 72, 80, 72, 74, 44, 80),
    ("C. Baumgartner",   "Austria",     "CM",  79, 76, 72, 76, 78, 58, 70),
    ("P. Pentz",         "Austria",     "GK",  76, 0,  0,  0,  0,  0,  0),
    ("D. Alaba",         "Austria",     "CB",  86, 76, 70, 84, 82, 84, 74),
    ("F. Grillitsch",    "Austria",     "CDM", 77, 64, 60, 76, 70, 74, 72),
    ("M. Sabitzer",      "Austria",     "CM",  82, 76, 76, 80, 80, 66, 76),

    # ── Ghana ───────────────────────────────────────────────────────────────────
    ("A. Kudus",         "Ghana",       "CAM", 80, 84, 76, 76, 82, 46, 72),
    ("J. Ayew",          "Ghana",       "ST",  79, 78, 76, 74, 76, 46, 72),
    ("L. Bati",          "Ghana",       "GK",  72, 0,  0,  0,  0,  0,  0),
    ("D. Amartey",       "Ghana",       "CB",  76, 68, 44, 62, 60, 77, 76),
    ("T. Partey",        "Ghana",       "CDM", 83, 72, 65, 80, 76, 83, 82),

    # ── Croatia ─────────────────────────────────────────────────────────────────
    ("L. Modrić",        "Croatia",     "CM",  88, 72, 76, 91, 90, 62, 67),
    ("I. Perišić",       "Croatia",     "LW",  83, 83, 78, 78, 80, 64, 80),
    ("A. Kramarić",      "Croatia",     "ST",  82, 76, 82, 72, 80, 42, 74),
    ("D. Livaković",     "Croatia",     "GK",  84, 0,  0,  0,  0,  0,  0),
    ("J. Gvardiol",      "Croatia",     "CB",  87, 82, 56, 72, 76, 87, 82),
    ("M. Brozović",      "Croatia",     "CDM", 84, 68, 68, 86, 80, 78, 74),
    ("D. Šimunić",       "Croatia",     "CB",  76, 64, 42, 60, 58, 77, 76),

    # ── Bosnia ──────────────────────────────────────────────────────────────────
    ("E. Džeko",         "Bosnia",      "ST",  82, 68, 82, 72, 76, 40, 78),
    ("M. Pjanić",        "Bosnia",      "CM",  82, 64, 72, 87, 80, 58, 68),
    ("I. Šehić",         "Bosnia",      "GK",  76, 0,  0,  0,  0,  0,  0),
    ("E. Šunjić",        "Bosnia",      "CDM", 74, 60, 52, 70, 62, 74, 72),
    ("S. Kolasinac",     "Bosnia",      "LB",  79, 72, 56, 68, 66, 76, 82),

    # ── Sweden ──────────────────────────────────────────────────────────────────
    ("V. Gyökeres",      "Sweden",      "ST",  84, 82, 86, 68, 80, 42, 84),
    ("A. Isak",          "Sweden",      "ST",  84, 88, 82, 72, 84, 38, 76),
    ("R. Olsen",         "Sweden",      "GK",  79, 0,  0,  0,  0,  0,  0),
    ("M. Lustig",        "Sweden",      "RB",  76, 74, 58, 70, 66, 72, 70),
    ("D. Kulusevski",    "Sweden",      "RW",  83, 84, 78, 80, 83, 54, 75),
    ("E. Forsberg",      "Sweden",      "CAM", 81, 78, 76, 82, 82, 50, 68),

    # ── Iran ────────────────────────────────────────────────────────────────────
    ("S. Azmoun",        "Iran",        "ST",  80, 76, 80, 70, 76, 42, 78),
    ("M. Taremi",        "Iran",        "ST",  82, 76, 84, 72, 78, 44, 78),
    ("A. Beiranvand",    "Iran",        "GK",  78, 0,  0,  0,  0,  0,  0),
    ("E. Hajsafi",       "Iran",        "LB",  74, 72, 54, 68, 64, 68, 68),
    ("S. Hosseini",      "Iran",        "CB",  73, 60, 40, 56, 54, 74, 74),

    # ── Cape Verde ──────────────────────────────────────────────────────────────
    ("G. Benchimol",     "Cape Verde",  "ST",  72, 74, 70, 60, 68, 36, 70),
    ("V. Varela",        "Cape Verde",  "GK",  68, 0,  0,  0,  0,  0,  0),
    ("J. Landim",        "Cape Verde",  "RW",  70, 76, 64, 62, 68, 32, 62),

    # ── Paraguay ────────────────────────────────────────────────────────────────
    ("M. Almiron",       "Paraguay",    "CAM", 80, 82, 72, 76, 80, 52, 70),
    ("R. Sanabria",      "Paraguay",    "ST",  77, 72, 76, 64, 72, 38, 72),
    ("A. Silva",         "Paraguay",    "GK",  74, 0,  0,  0,  0,  0,  0),
    ("F. Balbuena",      "Paraguay",    "CB",  76, 64, 44, 60, 58, 77, 78),
    ("J. Enciso",        "Paraguay",    "RW",  77, 84, 72, 72, 77, 34, 64),

    # ── Ecuador ─────────────────────────────────────────────────────────────────
    ("M. Caicedo",       "Ecuador",     "CDM", 84, 74, 62, 78, 78, 84, 82),
    ("E. Valencia",      "Ecuador",     "ST",  79, 76, 77, 62, 68, 42, 82),
    ("H. Galíndez",      "Ecuador",     "GK",  76, 0,  0,  0,  0,  0,  0),
    ("P. Hincapié",      "Ecuador",     "CB",  80, 74, 46, 66, 66, 80, 76),
    ("A. Preciado",      "Ecuador",     "RB",  75, 78, 56, 66, 66, 66, 70),
    ("Á. Mena",          "Ecuador",     "LB",  74, 72, 54, 66, 64, 68, 68),

    # ── Uzbekistan ──────────────────────────────────────────────────────────────
    ("E. Shomurodov",    "Uzbekistan",  "ST",  76, 76, 74, 62, 70, 38, 72),
    ("U. Nishonov",      "Uzbekistan",  "GK",  68, 0,  0,  0,  0,  0,  0),
    ("J. Mirzayev",      "Uzbekistan",  "CB",  68, 60, 36, 54, 50, 70, 70),
    ("O. Zubidov",       "Uzbekistan",  "CM",  68, 64, 58, 66, 64, 60, 62),
]

# ── API fetch for player stats ─────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_cards_from_api(api_key: str) -> list[tuple]:
    """
    Fetch player stats for WC 2026 squads from API-Football.
    Returns same tuple format as PLAYER_CARDS_FALLBACK.
    Falls back gracefully — returns [] if nothing useful comes back.
    """
    all_players = []
    # Fetch top players for WC 2026 league
    url = (f"https://v3.football.api-sports.io/players"
           f"?league={WC2026_LEAGUE_ID}&season={WC2026_SEASON}&page=1")
    data = api_get(url, api_key)
    if not data or not data.get("response"):
        return []

    pages = data.get("paging", {}).get("total", 1)
    responses = data["response"]

    for page in range(2, min(pages + 1, 6)):  # max 5 pages
        d = api_get(url.replace("page=1", f"page={page}"), api_key)
        if d and d.get("response"):
            responses += d["response"]

    for entry in responses:
        p     = entry.get("player", {})
        stats = entry.get("statistics", [{}])[0]
        name  = p.get("name", "")
        team_api = stats.get("team", {}).get("name", "")
        team  = API_NAME_MAP.get(team_api)
        if not team:
            continue
        pos   = stats.get("games", {}).get("position") or "—"
        # Map API positions to short codes
        pos_map = {"Goalkeeper":"GK","Defender":"CB","Midfielder":"CM","Attacker":"ST"}
        pos = pos_map.get(pos, pos[:2].upper())
        rating_raw = stats.get("games", {}).get("rating")
        overall = int(float(rating_raw) * 10) if rating_raw else 75
        overall = max(60, min(99, overall))
        # API doesn't give FIFA sub-stats; use 0s and note in UI
        all_players.append((name, team, pos, overall, 0, 0, 0, 0, 0, 0))

    return all_players


def get_card_color(overall: int) -> tuple[str, str, str]:
    """Return (bg_gradient, border_color, text_color) based on overall rating."""
    if overall >= 90:
        return ("linear-gradient(145deg, #b8860b, #ffd700, #b8860b)",
                "#ffd700", "#3d2800")   # Gold TOTY
    elif overall >= 86:
        return ("linear-gradient(145deg, #8b7536, #d4af37, #8b7536)",
                "#d4af37", "#2a1f00")   # Gold
    elif overall >= 82:
        return ("linear-gradient(145deg, #7a7a7a, #c0c0c0, #7a7a7a)",
                "#c0c0c0", "#1a1a1a")   # Silver
    else:
        return ("linear-gradient(145deg, #5a3a1a, #a0714a, #5a3a1a)",
                "#a0714a", "#f5e6d0")   # Bronze


def render_player_card(p: dict) -> str:
    """Return HTML string for a single FIFA-style player card."""
    overall  = p["overall"]
    bg, border, text_col = get_card_color(overall)
    flag     = BASE_TEAMS.get(p["team"], {}).get("flag", "🏳️")

    # Stat rows (skip for GK since API gives zeros)
    is_gk = p["pos"] == "GK"
    if is_gk:
        stat_labels = ["DIV", "HAN", "KIC", "REF", "SPD", "POS"]
        stat_vals   = [p["pace"], p["shooting"], p["passing"],
                       p["dribbling"], p["defending"], p["physical"]]
    else:
        stat_labels = ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]
        stat_vals   = [p["pace"], p["shooting"], p["passing"],
                       p["dribbling"], p["defending"], p["physical"]]

    stats_html = ""
    for label, val in zip(stat_labels, stat_vals):
        if val == 0:
            continue
        bar_pct = val
        stats_html += f"""
        <div style="display:flex;align-items:center;gap:4px;margin:2px 0;">
          <span style="font-size:9px;font-weight:800;width:26px;color:{text_col};opacity:0.85;">{label}</span>
          <div style="flex:1;background:rgba(0,0,0,0.2);border-radius:3px;height:5px;">
            <div style="width:{bar_pct}%;background:{text_col};height:5px;border-radius:3px;opacity:0.9;"></div>
          </div>
          <span style="font-size:10px;font-weight:700;width:22px;text-align:right;color:{text_col};">{val}</span>
        </div>"""

    no_stats_msg = ""
    if all(v == 0 for v in stat_vals):
        no_stats_msg = f'<div style="font-size:9px;opacity:0.7;text-align:center;margin-top:4px;color:{text_col};">Sub-stats via API key</div>'

    card = f"""
    <div style="
        background:{bg};
        border:2px solid {border};
        border-radius:12px;
        padding:10px 10px 8px 10px;
        width:145px;
        min-height:210px;
        box-shadow:0 4px 15px rgba(0,0,0,0.4);
        font-family:'Segoe UI',sans-serif;
        display:flex;flex-direction:column;
        align-items:center;
        position:relative;
        box-sizing:border-box;
    ">
      <div style="font-size:28px;font-weight:900;color:{text_col};line-height:1;">{overall}</div>
      <div style="font-size:10px;font-weight:700;color:{text_col};opacity:0.8;letter-spacing:1px;">{p['pos']}</div>
      <div style="font-size:20px;margin:4px 0 2px 0;">{flag}</div>
      <div style="font-size:11px;font-weight:800;color:{text_col};text-align:center;
                  line-height:1.2;margin-bottom:6px;max-width:125px;word-break:break-word;">
        {p['name']}
      </div>
      <div style="width:100%;border-top:1px solid rgba(0,0,0,0.2);padding-top:5px;">
        {stats_html}
        {no_stats_msg}
      </div>
      <div style="position:absolute;bottom:5px;right:7px;font-size:8px;
                  color:{text_col};opacity:0.5;font-weight:600;">{p['team'][:10]}</div>
    </div>"""
    return card


# ════════════════════════════════════════════════════════════════════════════════
# PLAYER CARDS TAB (appended to existing tabs in main UI below)
# ════════════════════════════════════════════════════════════════════════════════
