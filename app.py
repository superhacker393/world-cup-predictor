import streamlit as st
import numpy as np
import pandas as pd
import requests
import random
from datetime import datetime, timezone

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="2026 World Cup Simulator",
    page_icon="🏆",
    layout="wide",
)

BASE_URL = "https://api.football-data.org/v4"

# ── Baseline team database ─────────────────────────────────────────────────────
# atk  = expected goals scored vs average opponent (Elo-derived baseline)
# def_ = multiplier on opponent attack (lower = stronger defense)
# elo  = approximate World Football Elo rating
# These are updated dynamically from the API when a key is provided.

BASELINE_TEAMS = {
    "France":        {"flag": "🇫🇷", "atk": 2.05, "def": 0.72, "elo": 2100},
    "Argentina":     {"flag": "🇦🇷", "atk": 2.00, "def": 0.78, "elo": 2060},
    "Spain":         {"flag": "🇪🇸", "atk": 1.85, "def": 0.70, "elo": 2020},
    "England":       {"flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "atk": 1.80, "def": 0.76, "elo": 2010},
    "Brazil":        {"flag": "🇧🇷", "atk": 1.90, "def": 0.74, "elo": 2000},
    "Germany":       {"flag": "🇩🇪", "atk": 1.82, "def": 0.80, "elo": 1980},
    "Netherlands":   {"flag": "🇳🇱", "atk": 1.78, "def": 0.77, "elo": 1970},
    "Portugal":      {"flag": "🇵🇹", "atk": 1.76, "def": 0.82, "elo": 1960},
    "Belgium":       {"flag": "🇧🇪", "atk": 1.74, "def": 0.85, "elo": 1950},
    "Morocco":       {"flag": "🇲🇦", "atk": 1.40, "def": 0.68, "elo": 1900},
    "USA":           {"flag": "🇺🇸", "atk": 1.55, "def": 0.86, "elo": 1880},
    "Mexico":        {"flag": "🇲🇽", "atk": 1.62, "def": 0.80, "elo": 1920},
    "Switzerland":   {"flag": "🇨🇭", "atk": 1.60, "def": 0.79, "elo": 1910},
    "Colombia":      {"flag": "🇨🇴", "atk": 1.58, "def": 0.84, "elo": 1890},
    "Japan":         {"flag": "🇯🇵", "atk": 1.52, "def": 0.82, "elo": 1870},
    "Australia":     {"flag": "🇦🇺", "atk": 1.42, "def": 0.88, "elo": 1820},
    "Norway":        {"flag": "🇳🇴", "atk": 1.65, "def": 0.86, "elo": 1870},
    "Ivory Coast":   {"flag": "🇨🇮", "atk": 1.48, "def": 0.88, "elo": 1830},
    "South Africa":  {"flag": "🇿🇦", "atk": 1.30, "def": 0.90, "elo": 1750},
    "Canada":        {"flag": "🇨🇦", "atk": 1.55, "def": 0.87, "elo": 1840},
    "Egypt":         {"flag": "🇪🇬", "atk": 1.42, "def": 0.84, "elo": 1800},
    "Senegal":       {"flag": "🇸🇳", "atk": 1.45, "def": 0.86, "elo": 1810},
    "Austria":       {"flag": "🇦🇹", "atk": 1.52, "def": 0.85, "elo": 1850},
    "Ghana":         {"flag": "🇬🇭", "atk": 1.38, "def": 0.90, "elo": 1760},
    "Croatia":       {"flag": "🇭🇷", "atk": 1.44, "def": 0.84, "elo": 1820},
    "Bosnia":        {"flag": "🇧🇦", "atk": 1.35, "def": 0.88, "elo": 1770},
    "Sweden":        {"flag": "🇸🇪", "atk": 1.55, "def": 0.85, "elo": 1840},
    "Iran":          {"flag": "🇮🇷", "atk": 1.25, "def": 0.87, "elo": 1740},
    "Cape Verde":    {"flag": "🇨🇻", "atk": 1.20, "def": 0.90, "elo": 1680},
    "Paraguay":      {"flag": "🇵🇾", "atk": 1.30, "def": 0.88, "elo": 1750},
    "Ecuador":       {"flag": "🇪🇨", "atk": 1.35, "def": 0.87, "elo": 1770},
    "Uzbekistan":    {"flag": "🇺🇿", "atk": 1.22, "def": 0.91, "elo": 1700},
}

# API name → our internal name (football-data.org uses different spellings)
API_NAME_MAP = {
    "United States": "USA",
    "Côte d'Ivoire": "Ivory Coast",
    "Bosnia and Herzegovina": "Bosnia",
    "Cabo Verde": "Cape Verde",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
}

HOST_NATIONS = {"USA", "Mexico", "Canada"}

# ── API helpers ────────────────────────────────────────────────────────────────

def api_headers(key: str) -> dict:
    return {"X-Auth-Token": key}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_wc_matches(api_key: str) -> list:
    """Fetch all WC 2026 matches from football-data.org."""
    try:
        r = requests.get(
            f"{BASE_URL}/competitions/WC/matches",
            headers=api_headers(api_key),
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("matches", [])
    except Exception as e:
        st.warning(f"Could not fetch matches: {e}")
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_wc_standings(api_key: str) -> list:
    """Fetch WC 2026 group standings."""
    try:
        r = requests.get(
            f"{BASE_URL}/competitions/WC/standings",
            headers=api_headers(api_key),
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("standings", [])
    except Exception as e:
        st.warning(f"Could not fetch standings: {e}")
        return []


def normalise_name(api_name: str) -> str:
    """Map API team name to our internal name."""
    return API_NAME_MAP.get(api_name, api_name)


def compute_live_ratings(matches: list, base: dict) -> dict:
    """
    Recalibrate attack/defense ratings from completed group-stage results.
    Uses goals scored/conceded per game as a Bayesian update on the baseline.
    weight=0.6 means live data counts 60%, baseline 40% — tunable.
    """
    if not matches:
        return base

    goals_for:     dict[str, list] = {}
    goals_against: dict[str, list] = {}

    for m in matches:
        if m.get("status") != "FINISHED":
            continue
        stage = m.get("stage", "")
        if "GROUP" not in stage.upper():
            continue

        score = m.get("score", {}).get("fullTime", {})
        home_g = score.get("home")
        away_g = score.get("away")
        if home_g is None or away_g is None:
            continue

        home = normalise_name(m["homeTeam"]["name"])
        away = normalise_name(m["awayTeam"]["name"])

        goals_for.setdefault(home, []).append(home_g)
        goals_for.setdefault(away, []).append(away_g)
        goals_against.setdefault(home, []).append(away_g)
        goals_against.setdefault(away, []).append(home_g)

    if not goals_for:
        return base

    LIVE_WEIGHT = 0.6  # how much live data overrides baseline

    updated = {}
    for team, info in base.items():
        gf = goals_for.get(team)
        ga = goals_against.get(team)
        if gf and ga:
            live_atk = np.mean(gf)
            live_def = np.mean(ga) / 1.5  # normalise to multiplier scale
            new_atk = (1 - LIVE_WEIGHT) * info["atk"] + LIVE_WEIGHT * live_atk
            new_def = (1 - LIVE_WEIGHT) * info["def"] + LIVE_WEIGHT * max(live_def, 0.40)
        else:
            new_atk = info["atk"]
            new_def = info["def"]
        updated[team] = {**info, "atk": round(new_atk, 3), "def": round(new_def, 3)}

    return updated


def build_live_bracket(matches: list) -> list:
    """
    Build R32 pairings from the API fixtures.
    Falls back to hardcoded bracket for any unconfirmed slots.
    """
    FALLBACK = [
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

    if not matches:
        return FALLBACK

    r32 = []
    for m in matches:
        stage = m.get("stage", "")
        if "ROUND_OF_32" not in stage.upper() and "LAST_32" not in stage.upper():
            continue
        home = normalise_name(m["homeTeam"].get("name", "TBD"))
        away = normalise_name(m["awayTeam"].get("name", "TBD"))
        date_str = m.get("utcDate", "")
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            date_label = dt.strftime("%b %-d")
        except Exception:
            date_label = "TBD"
        r32.append((home, away, date_label))

    return r32 if r32 else FALLBACK


# ── Poisson match engine ───────────────────────────────────────────────────────

def get_team(name: str, teams: dict) -> dict:
    return teams.get(name, {"flag": "🏳️", "atk": 1.25, "def": 0.91, "elo": 1650})


def expected_goals(team_a: str, team_b: str, home_adv: float, teams: dict) -> tuple:
    ta = get_team(team_a, teams)
    tb = get_team(team_b, teams)
    boost_a = home_adv if team_a in HOST_NATIONS else 0.0
    boost_b = home_adv if team_b in HOST_NATIONS else 0.0
    xg_a = ta["atk"] * tb["def"] * (1 + boost_a)
    xg_b = tb["atk"] * ta["def"] * (1 + boost_b)
    return xg_a, xg_b


def simulate_match(team_a: str, team_b: str, home_adv: float, teams: dict) -> tuple:
    """
    Poisson goal draw → extra time (33% xG) → penalty shootout (50/50).
    Returns (winner, goals_a, goals_b).
    """
    if team_a == "TBD":
        return team_b, 0, 0
    if team_b == "TBD":
        return team_a, 0, 0

    xg_a, xg_b = expected_goals(team_a, team_b, home_adv, teams)
    goals_a = np.random.poisson(xg_a)
    goals_b = np.random.poisson(xg_b)

    if goals_a == goals_b:
        et_a = np.random.poisson(xg_a * 0.33)
        et_b = np.random.poisson(xg_b * 0.33)
        goals_a += et_a
        goals_b += et_b
        if goals_a == goals_b:
            winner = team_a if random.random() < 0.5 else team_b
            return winner, goals_a, goals_b

    return (team_a, goals_a, goals_b) if goals_a > goals_b else (team_b, goals_a, goals_b)


def simulate_tournament(bracket: list, home_adv: float, teams: dict) -> str:
    survivors = []
    for home, away, _ in bracket:
        winner, _, _ = simulate_match(home, away, home_adv, teams)
        survivors.append(winner)

    while len(survivors) > 1:
        nxt = []
        for i in range(0, len(survivors), 2):
            if i + 1 >= len(survivors):
                nxt.append(survivors[i])
            else:
                w, _, _ = simulate_match(survivors[i], survivors[i + 1], 0, teams)
                nxt.append(w)
        survivors = nxt

    return survivors[0]


@st.cache_data(show_spinner=False)
def run_simulations(n: int, home_adv: float, seed: int, bracket_key: str, ratings_key: str) -> dict:
    """Cached simulation run. bracket_key/ratings_key are cache-busting strings."""
    # Reconstruct bracket and teams from session state (passed via keys)
    bracket = st.session_state.get("live_bracket", [])
    teams   = st.session_state.get("live_teams", BASELINE_TEAMS)

    np.random.seed(seed)
    random.seed(seed)
    counts: dict[str, int] = {}
    for _ in range(n):
        w = simulate_tournament(bracket, home_adv, teams)
        counts[w] = counts.get(w, 0) + 1
    return counts


def h2h_win_prob(team_a: str, team_b: str, home_adv: float, teams: dict, n: int = 20_000) -> dict:
    results = {"team_a": 0, "team_b": 0, "draw_90": 0}
    xg_a, xg_b = expected_goals(team_a, team_b, home_adv, teams)
    for _ in range(n):
        ga = np.random.poisson(xg_a)
        gb = np.random.poisson(xg_b)
        if ga > gb:
            results["team_a"] += 1
        elif gb > ga:
            results["team_b"] += 1
        else:
            results["draw_90"] += 1
    return {k: v / n * 100 for k, v in results.items()}


# ── Sidebar — API key & data loading ──────────────────────────────────────────

with st.sidebar:
    st.header("🔑 Data feed")
    api_key = st.text_input(
        "football-data.org API key",
        type="password",
        help="Get a free key at football-data.org/client/register",
    )

    data_loaded = False
    live_teams  = BASELINE_TEAMS.copy()
    live_bracket_raw = []

    if api_key:
        with st.spinner("Fetching live data…"):
            matches   = fetch_wc_matches(api_key)
            standings = fetch_wc_standings(api_key)

        if matches:
            live_teams        = compute_live_ratings(matches, BASELINE_TEAMS)
            live_bracket_raw  = build_live_bracket(matches)
            data_loaded       = True
            finished = sum(1 for m in matches if m.get("status") == "FINISHED")
            st.success(f"Live data active · {finished} matches loaded")
        else:
            st.error("Key invalid or quota exceeded — using baseline ratings")
            live_bracket_raw = build_live_bracket([])
    else:
        live_bracket_raw = build_live_bracket([])
        st.info("Enter API key for live ratings")

    if not data_loaded:
        st.caption("Running on Elo-calibrated baseline ratings")

    st.divider()
    st.caption("Ratings update every 5 min · Free tier: 10 req/min")

# Store in session state for cached simulation access
st.session_state["live_bracket"] = live_bracket_raw or build_live_bracket([])
st.session_state["live_teams"]   = live_teams

# ── Main UI ────────────────────────────────────────────────────────────────────

st.title("🏆 2026 World Cup Simulator")
mode_label = "🟢 Live ratings" if data_loaded else "⚪ Baseline ratings"
st.caption(f"Poisson goal model · Full knockout bracket · Monte Carlo · {mode_label}")

tab_sim, tab_h2h, tab_bracket, tab_standings, tab_model = st.tabs([
    "📊 Simulate",
    "⚔️  Head-to-head",
    "🗓️  R32 bracket",
    "📋 Standings",
    "🔬 Model",
])

# ─── TAB 1: Tournament simulation ─────────────────────────────────────────────
with tab_sim:
    st.subheader("Monte Carlo tournament simulation")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_sims   = st.slider("Simulations", 1_000, 50_000, 10_000, step=1_000)
    with c2:
        home_adv = st.slider("Host-nation advantage", 0.0, 0.6, 0.30, step=0.05)
    with c3:
        seed     = st.number_input("Random seed", value=42, step=1)

    if st.button("▶  Run simulation", type="primary", use_container_width=True):
        bracket = st.session_state["live_bracket"]
        teams   = st.session_state["live_teams"]

        # cache-bust keys so new live data forces a re-run
        b_key = str(sorted([(h, a) for h, a, _ in bracket]))
        r_key = str(sorted([(k, v["atk"], v["def"]) for k, v in teams.items()]))

        with st.spinner(f"Simulating {n_sims:,} tournaments…"):
            counts = run_simulations(n_sims, home_adv, int(seed), b_key, r_key)

        results = sorted(
            [
                {
                    "Flag":   get_team(t, teams)["flag"],
                    "Team":   t,
                    "Elo":    get_team(t, teams)["elo"],
                    "Atk λ":  round(get_team(t, teams)["atk"], 2),
                    "Def ×":  round(get_team(t, teams)["def"], 2),
                    "Wins":   c,
                    "Win %":  round(c / n_sims * 100, 2),
                }
                for t, c in counts.items()
            ],
            key=lambda x: -x["Win %"],
        )
        df = pd.DataFrame(results).reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)

        top    = df.iloc[0]
        runner = df.iloc[1]
        st.success(
            f"{top['Flag']} **{top['Team']}** — predicted champion · "
            f"{top['Win %']:.1f}% · "
            f"+{top['Win %'] - runner['Win %']:.1f}pp over {runner['Team']}"
        )

        col_tbl, col_chart = st.columns([3, 2], gap="large")
        with col_tbl:
            st.markdown("#### Championship probabilities")
            st.dataframe(
                df[["Rank", "Flag", "Team", "Elo", "Atk λ", "Def ×", "Win %"]]
                .style
                .format({"Win %": "{:.2f}%", "Elo": "{:,}", "Atk λ": "{:.2f}", "Def ×": "{:.2f}"})
                .background_gradient(cmap="Greens", subset=["Win %"]),
                use_container_width=True,
                hide_index=True,
            )
        with col_chart:
            st.markdown("#### Top 12 — win probability")
            st.bar_chart(df.head(12).set_index("Team")[["Win %"]], use_container_width=True)

# ─── TAB 2: Head-to-head ──────────────────────────────────────────────────────
with tab_h2h:
    st.subheader("Head-to-head match probability")
    teams = st.session_state["live_teams"]
    all_names = sorted(teams.keys())

    ca, _, cb = st.columns([5, 1, 5])
    with ca:
        team_a = st.selectbox("Team A", all_names, index=all_names.index("France"))
    with cb:
        team_b = st.selectbox("Team B", all_names, index=all_names.index("Argentina"))

    h2h_adv = st.slider("Home advantage", 0.0, 0.6, 0.0, step=0.05, key="h2h_adv")

    if st.button("▶  Calculate", type="primary", use_container_width=True):
        if team_a == team_b:
            st.warning("Pick two different teams.")
        else:
            with st.spinner("Simulating 20,000 matches…"):
                probs = h2h_win_prob(team_a, team_b, h2h_adv, teams)
            xg_a, xg_b = expected_goals(team_a, team_b, h2h_adv, teams)
            ta_flag = get_team(team_a, teams)["flag"]
            tb_flag = get_team(team_b, teams)["flag"]

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric(f"{ta_flag} {team_a} win", f"{probs['team_a']:.1f}%")
            with r2:
                st.metric("Draw (90 min)", f"{probs['draw_90']:.1f}%")
            with r3:
                st.metric(f"{tb_flag} {team_b} win", f"{probs['team_b']:.1f}%")

            st.divider()
            e1, e2 = st.columns(2)
            with e1:
                st.metric(f"{ta_flag} xG", f"{xg_a:.2f}")
            with e2:
                st.metric(f"{tb_flag} xG", f"{xg_b:.2f}")

            st.caption("Win % = 90 minutes only. Draws go to extra time then penalties in knockout matches.")

# ─── TAB 3: R32 bracket ───────────────────────────────────────────────────────
with tab_bracket:
    st.subheader("Round of 32 fixtures")
    teams   = st.session_state["live_teams"]
    bracket = st.session_state["live_bracket"]

    b_adv = st.slider("Host advantage", 0.0, 0.6, 0.30, step=0.05, key="b_adv")

    rows = []
    for home, away, date in bracket:
        th = get_team(home, teams)
        ta = get_team(away, teams)
        if away == "TBD":
            rows.append({
                "Date": date, "Home": f"{th['flag']} {home}",
                "Away": "🏳️ TBD", "Home xG": "—", "Away xG": "—",
                "Home win %": "—", "Draw %": "—", "Away win %": "—",
            })
        else:
            xg_h, xg_a = expected_goals(home, away, b_adv, teams)
            probs = h2h_win_prob(home, away, b_adv, teams, n=10_000)
            rows.append({
                "Date": date,
                "Home": f"{th['flag']} {home}",
                "Away": f"{ta['flag']} {away}",
                "Home xG": f"{xg_h:.2f}",
                "Away xG": f"{xg_a:.2f}",
                "Home win %": f"{probs['team_a']:.1f}%",
                "Draw %":     f"{probs['draw_90']:.1f}%",
                "Away win %": f"{probs['team_b']:.1f}%",
            })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─── TAB 4: Live standings ────────────────────────────────────────────────────
with tab_standings:
    st.subheader("Group stage standings")
    if not api_key:
        st.info("Enter your API key in the sidebar to load live standings.")
    elif not data_loaded:
        st.warning("Could not load standings — check your API key.")
    else:
        standings = fetch_wc_standings(api_key)
        if standings:
            for group in standings:
                group_name = group.get("group", "Group")
                table      = group.get("table", [])
                if not table:
                    continue
                rows = []
                for entry in table:
                    team = entry.get("team", {})
                    name = normalise_name(team.get("name", ""))
                    flag = get_team(name, live_teams).get("flag", "🏳️")
                    rows.append({
                        "Pos": entry.get("position", ""),
                        "Team": f"{flag} {name}",
                        "P":   entry.get("playedGames", 0),
                        "W":   entry.get("won", 0),
                        "D":   entry.get("draw", 0),
                        "L":   entry.get("lost", 0),
                        "GF":  entry.get("goalsFor", 0),
                        "GA":  entry.get("goalsAgainst", 0),
                        "GD":  entry.get("goalDifference", 0),
                        "Pts": entry.get("points", 0),
                    })
                with st.expander(f"**{group_name}**", expanded=True):
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No standings data available yet.")

# ─── TAB 5: Model explainer ───────────────────────────────────────────────────
with tab_model:
    st.subheader("How the model works")

    with st.expander("Poisson goal model", expanded=True):
        st.markdown("""
**Goals are drawn from a Poisson distribution** — the standard framework used by Opta, FiveThirtyEight, and bookmakers.

```
xG_A = attack_A × defense_B × (1 + home_boost if A is a host nation)
xG_B = attack_B × defense_A
```

Goals scored = `Poisson(xG)`. Draws → extra time at 33% xG → penalty shootout (50/50 coin flip).
        """)

    with st.expander("Live rating recalibration"):
        st.markdown("""
When an API key is provided, the model pulls every completed group-stage result and updates attack/defense ratings using a **Bayesian blend**:

```
new_atk = 0.4 × baseline_atk + 0.6 × (goals scored per game, live)
new_def = 0.4 × baseline_def + 0.6 × (goals conceded per game, normalised)
```

This means a team that has been scoring heavily (e.g. France 10 goals in 3 games) will get a boosted attack rating for the knockout simulations, and a team that has been leaky will get a weakened defense. Ratings refresh every 5 minutes from the API.
        """)

    with st.expander("Team baseline ratings"):
        teams_df = pd.DataFrame([
            {"Flag": v["flag"], "Team": k,
             "Attack λ": v["atk"], "Defense ×": v["def"], "Elo": v["elo"]}
            for k, v in sorted(live_teams.items(), key=lambda x: -x[1]["elo"])
        ])
        st.dataframe(
            teams_df.style.format({"Attack λ": "{:.3f}", "Defense ×": "{:.3f}", "Elo": "{:,}"}),
            use_container_width=True, hide_index=True,
        )

    with st.expander("What pro models add on top"):
        st.markdown("""
| Feature | This model | Pro model (Opta / FiveThirtyEight) |
|---|---|---|
| Goal distribution | Poisson ✅ | Poisson / Dixon-Coles ✅ |
| Rating source | Elo + live goals | MLE fit on 10k+ matches |
| Live updates | Every 5 min via API ✅ | After every event |
| Injury / squad data | Not included | Live feeds |
| Penalty model | 50/50 | Team historical penalty records |
| xG source | Rating-derived | Shot-level StatsBomb / Opta |
| Bracket structure | Real 2026 WC ✅ | Real 2026 WC ✅ |
        """)
