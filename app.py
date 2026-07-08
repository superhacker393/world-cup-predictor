import streamlit as st
import numpy as np
import pandas as pd
import random
import requests
from datetime import datetime, timedelta
from collections import defaultdict, Counter

st.set_page_config(page_title="2026 World Cup Simulator", page_icon="🏆", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#111; }
[data-testid="stSidebar"]          { background:#0d0d0d; border-right:1px solid #1e1e1e; }
[data-testid="stHeader"]           { background:transparent; }
.block-container { padding-top:1rem !important; padding-bottom:1rem !important; }
[data-testid="stVerticalBlock"]    { gap:0.35rem !important; }
div[data-testid="column"]          { padding:0 3px !important; }
[data-testid="metric-container"]   { background:#161616; border:1px solid #242424; border-radius:4px; padding:6px 10px !important; }
[data-testid="stMetricLabel"]      { color:#666 !important; font-size:10px !important; }
[data-testid="stMetricValue"]      { color:#ddd !important; font-size:1.1rem !important; font-weight:700 !important; }
[data-testid="stMetricDelta"]      { font-size:10px !important; }
[data-testid="stTabs"] button                       { color:#555; font-size:12px; font-weight:500; padding:5px 9px; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#ccc !important; border-bottom:2px solid #777 !important; }
h1 { color:#ddd !important; font-weight:800 !important; font-size:1.3rem !important; margin-bottom:0 !important; }
h2 { color:#bbb !important; font-weight:700 !important; font-size:1rem !important; }
h3,h4 { color:#999 !important; font-weight:600 !important; font-size:0.9rem !important; }
hr { border-color:#1e1e1e !important; margin:0.5rem 0 !important; }
[data-testid="baseButton-primary"]  { background:#ccc !important; color:#111 !important; font-weight:700 !important; border:none !important; border-radius:4px !important; }
[data-testid="stSidebarContent"]    { color:#888; }
[data-testid="stCaptionContainer"]  { color:#444 !important; font-size:11px !important; }
[data-testid="stDataFrame"]         { border:1px solid #1e1e1e; border-radius:4px; }
[data-testid="stSlider"]            { padding-top:2px; padding-bottom:2px; }
label[data-testid="stWidgetLabel"] p { font-size:12px !important; color:#888 !important; }
.dbar { display:flex; border:1px solid #1e1e1e; border-radius:4px; overflow:hidden; margin-bottom:6px; }
.dbar>div { flex:1; padding:5px 9px; border-right:1px solid #1e1e1e; }
.dbar>div:last-child { border-right:none; }
.dbar .dl { font-size:9px; color:#555; text-transform:uppercase; letter-spacing:.04em; }
.dbar .dv { font-size:13px; color:#ccc; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════
BASE_TEAMS = {
    "Argentina":   {"flag":"🇦🇷","atk":2.10,"def":0.74,"elo":2090},
    "France":      {"flag":"🇫🇷","atk":2.00,"def":0.73,"elo":2055},
    "Spain":       {"flag":"🇪🇸","atk":1.98,"def":0.68,"elo":2045},
    "England":     {"flag":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","atk":1.82,"def":0.74,"elo":2010},
    "Brazil":      {"flag":"🇧🇷","atk":1.85,"def":0.76,"elo":1995},
    "Germany":     {"flag":"🇩🇪","atk":1.88,"def":0.77,"elo":1985},
    "Portugal":    {"flag":"🇵🇹","atk":1.80,"def":0.80,"elo":1970},
    "Netherlands": {"flag":"🇳🇱","atk":1.76,"def":0.78,"elo":1960},
    "Morocco":     {"flag":"🇲🇦","atk":1.45,"def":0.65,"elo":1920},
    "Mexico":      {"flag":"🇲🇽","atk":1.60,"def":0.82,"elo":1915},
    "Switzerland": {"flag":"🇨🇭","atk":1.62,"def":0.77,"elo":1912},
    "USA":         {"flag":"🇺🇸","atk":1.58,"def":0.84,"elo":1900},
    "Belgium":     {"flag":"🇧🇪","atk":1.68,"def":0.84,"elo":1895},
    "Colombia":    {"flag":"🇨🇴","atk":1.65,"def":0.82,"elo":1892},
    "Japan":       {"flag":"🇯🇵","atk":1.55,"def":0.80,"elo":1878},
    "Austria":     {"flag":"🇦🇹","atk":1.58,"def":0.83,"elo":1870},
    "Norway":      {"flag":"🇳🇴","atk":1.70,"def":0.84,"elo":1868},
    "Sweden":      {"flag":"🇸🇪","atk":1.60,"def":0.83,"elo":1855},
    "Canada":      {"flag":"🇨🇦","atk":1.58,"def":0.85,"elo":1850},
    "Senegal":     {"flag":"🇸🇳","atk":1.48,"def":0.84,"elo":1842},
    "Croatia":     {"flag":"🇭🇷","atk":1.46,"def":0.82,"elo":1835},
    "Ivory Coast": {"flag":"🇨🇮","atk":1.50,"def":0.86,"elo":1832},
    "Australia":   {"flag":"🇦🇺","atk":1.44,"def":0.87,"elo":1825},
    "Ecuador":     {"flag":"🇪🇨","atk":1.40,"def":0.85,"elo":1800},
    "Egypt":       {"flag":"🇪🇬","atk":1.45,"def":0.83,"elo":1795},
    "South Africa":{"flag":"🇿🇦","atk":1.32,"def":0.89,"elo":1778},
    "Ghana":       {"flag":"🇬🇭","atk":1.40,"def":0.89,"elo":1770},
    "Bosnia":      {"flag":"🇧🇦","atk":1.38,"def":0.87,"elo":1765},
    "Iran":        {"flag":"🇮🇷","atk":1.28,"def":0.86,"elo":1755},
    "Paraguay":    {"flag":"🇵🇾","atk":1.33,"def":0.87,"elo":1748},
    "Cape Verde":  {"flag":"🇨🇻","atk":1.22,"def":0.89,"elo":1700},
    "Uzbekistan":  {"flag":"🇺🇿","atk":1.25,"def":0.90,"elo":1695},
}

PENALTY_WIN_RATE = {
    "Germany":0.75,"Argentina":0.71,"France":0.60,"Spain":0.50,
    "Brazil":0.50,"Portugal":0.67,"Netherlands":0.40,"England":0.44,
    "Croatia":0.67,"Switzerland":0.67,"Colombia":0.67,"Mexico":0.33,
    "USA":0.50,"Morocco":0.67,"Senegal":0.50,"Japan":0.50,
    "Ghana":0.33,"Belgium":0.50,"Sweden":0.50,"Norway":0.50,
    "Canada":0.50,"South Africa":0.50,"Australia":0.50,"Ivory Coast":0.40,
    "Egypt":0.44,"Austria":0.50,"Bosnia":0.50,"Iran":0.50,
    "Paraguay":0.60,"Ecuador":0.50,"Cape Verde":0.50,"Uzbekistan":0.50,
}

DEFAULT_HOST_ADV = {"USA":0.22,"Mexico":0.30,"Canada":0.18}

FALLBACK_SQUADS = {
    "Argentina":   [("L. Messi",0.35),("J. Álvarez",0.22),("L. Martínez",0.12),("R. De Paul",0.08),("Other",0.23)],
    "France":      [("K. Mbappé",0.38),("M. Thuram",0.18),("A. Griezmann",0.14),("O. Dembélé",0.10),("Other",0.20)],
    "Spain":       [("A. Morata",0.25),("L. Yamal",0.18),("F. Torres",0.16),("P. Pedri",0.12),("Other",0.29)],
    "England":     [("H. Kane",0.40),("B. Saka",0.18),("P. Foden",0.14),("J. Bellingham",0.12),("Other",0.16)],
    "Brazil":      [("Vinicius Jr.",0.34),("Rodrygo",0.20),("Raphinha",0.18),("G. Martinelli",0.10),("Other",0.18)],
    "Germany":     [("K. Havertz",0.26),("J. Musiala",0.22),("F. Wirtz",0.18),("L. Nmecha",0.12),("Other",0.22)],
    "Portugal":    [("C. Ronaldo",0.36),("B. Fernandes",0.20),("R. Leão",0.16),("G. Ramos",0.12),("Other",0.16)],
    "Netherlands": [("C. Gakpo",0.30),("M. Depay",0.22),("X. Simons",0.15),("D. Dumfries",0.10),("Other",0.23)],
    "Morocco":     [("Y. En-Nesyri",0.38),("H. Ziyech",0.22),("A. Hakimi",0.10),("S. Ezzalzouli",0.10),("Other",0.20)],
    "Mexico":      [("R. Jiménez",0.30),("H. Lozano",0.25),("A. Vega",0.15),("Other",0.30)],
    "Switzerland": [("B. Embolo",0.32),("X. Shaqiri",0.20),("R. Vargas",0.15),("Other",0.33)],
    "USA":         [("C. Pulisic",0.32),("J. Sargent",0.18),("F. Weah",0.15),("R. Reyna",0.12),("Other",0.23)],
    "Belgium":     [("R. Lukaku",0.36),("K. De Bruyne",0.18),("L. Trossard",0.14),("A. Doku",0.12),("Other",0.20)],
    "Colombia":    [("L. Díaz",0.30),("R. Falcao",0.20),("J. Cuadrado",0.14),("J. Muriel",0.12),("Other",0.24)],
    "Japan":       [("D. Kamada",0.25),("T. Minamino",0.22),("Ritsu Doan",0.20),("A. Ueda",0.12),("Other",0.21)],
    "Austria":     [("M. Arnautovic",0.30),("M. Sabitzer",0.20),("C. Baumgartner",0.18),("Other",0.32)],
    "Norway":      [("E. Haaland",0.55),("A. Sørloth",0.18),("M. Ødegaard",0.12),("Other",0.15)],
    "Sweden":      [("V. Gyökeres",0.44),("A. Isak",0.28),("D. Kulusevski",0.12),("Other",0.16)],
    "Canada":      [("J. David",0.32),("A. Davies",0.18),("T. Buchanan",0.15),("S. Larin",0.12),("Other",0.23)],
    "Senegal":     [("S. Mané",0.40),("I. Sarr",0.22),("N. Jackson",0.14),("Other",0.24)],
    "Croatia":     [("A. Kramarić",0.32),("I. Perišić",0.25),("L. Modrić",0.12),("Other",0.31)],
    "Ivory Coast": [("S. Haller",0.35),("W. Zaha",0.20),("N. Pépé",0.16),("Other",0.29)],
    "Australia":   [("M. Leckie",0.28),("A. Hrustic",0.20),("M. Goodwin",0.15),("Other",0.37)],
    "Ecuador":     [("E. Valencia",0.36),("M. Caicedo",0.18),("Á. Mena",0.12),("Other",0.34)],
    "Egypt":       [("M. Salah",0.52),("O. Marmoush",0.22),("Other",0.26)],
    "South Africa":[("P. Tau",0.32),("R. Foster",0.20),("Other",0.48)],
    "Ghana":       [("A. Kudus",0.30),("J. Ayew",0.25),("T. Partey",0.10),("Other",0.35)],
    "Bosnia":      [("E. Džeko",0.40),("M. Pjanić",0.14),("Other",0.46)],
    "Iran":        [("M. Taremi",0.38),("S. Azmoun",0.28),("Other",0.34)],
    "Paraguay":    [("M. Almiron",0.28),("R. Sanabria",0.25),("J. Enciso",0.15),("Other",0.32)],
    "Cape Verde":  [("G. Benchimol",0.30),("Other",0.70)],
    "Uzbekistan":  [("E. Shomurodov",0.40),("Other",0.60)],
}

API_NAME_MAP = {
    "France":"France","Argentina":"Argentina","Spain":"Spain","England":"England",
    "Brazil":"Brazil","Germany":"Germany","Netherlands":"Netherlands","Portugal":"Portugal",
    "Belgium":"Belgium","Morocco":"Morocco","USA":"USA","United States":"USA",
    "Mexico":"Mexico","Switzerland":"Switzerland","Colombia":"Colombia","Japan":"Japan",
    "Australia":"Australia","Norway":"Norway","Ivory Coast":"Ivory Coast",
    "South Africa":"South Africa","Canada":"Canada","Egypt":"Egypt","Senegal":"Senegal",
    "Austria":"Austria","Ghana":"Ghana","Croatia":"Croatia","Bosnia":"Bosnia",
    "Bosnia and Herzegovina":"Bosnia","Sweden":"Sweden","Iran":"Iran",
    "Cape Verde":"Cape Verde","Paraguay":"Paraguay","Ecuador":"Ecuador","Uzbekistan":"Uzbekistan",
}

WC2026_LEAGUE_ID = 1
WC2026_SEASON    = 2026

# ── Real WC 2026 R32 results (all 16 matches completed Jun 28 – Jul 3) ─────────
# Format: (home, away, date, goals_home, goals_away, status, winner)
# status: FT=90min, AET=after extra time, PEN=after penalties
R32_RESULTS = [
    ("South Africa","Canada",     "Jun 28", 0, 1, "FT",  "Canada"),
    ("Brazil",      "Japan",      "Jun 29", 2, 1, "FT",  "Brazil"),
    ("Germany",     "Paraguay",   "Jun 29", 1, 1, "PEN", "Paraguay"),   # Paraguay 4-3 on pens
    ("Netherlands", "Morocco",    "Jun 29", 1, 1, "PEN", "Morocco"),    # Morocco 3-2 on pens
    ("Norway",      "Ivory Coast","Jun 30", 2, 1, "FT",  "Norway"),
    ("France",      "Sweden",     "Jun 30", 3, 0, "FT",  "France"),
    ("Mexico",      "Ecuador",    "Jun 30", 2, 0, "FT",  "Mexico"),
    ("England",     "DR Congo",   "Jul 1",  2, 1, "AET", "England"),
    ("Belgium",     "Senegal",    "Jul 1",  3, 2, "AET", "Belgium"),
    ("USA",         "Bosnia",     "Jul 1",  2, 0, "FT",  "USA"),
    ("Spain",       "Austria",    "Jul 2",  3, 0, "FT",  "Spain"),
    ("Portugal",    "Croatia",    "Jul 2",  2, 1, "FT",  "Portugal"),
    ("Switzerland", "Algeria",    "Jul 2",  2, 0, "FT",  "Switzerland"),
    ("Australia",   "Egypt",      "Jul 3",  1, 1, "PEN", "Egypt"),      # Egypt 4-2 on pens
    ("Argentina",   "Cape Verde", "Jul 3",  3, 2, "AET", "Argentina"),
    ("Colombia",    "Ghana",      "Jul 3",  1, 0, "FT",  "Colombia"),
]

# ── Round of 16 fixtures (Jul 4-7, some in progress/upcoming as of Jul 5) ─────
R16_FIXTURES = [
    ("Morocco",      "Canada",      "Jul 4",  3, 0, "FT",  "Morocco"),    # completed
    ("France",       "Paraguay",    "Jul 4",  None, None, "FT", "France"), # France won, score TBC
    ("Brazil",       "Norway",      "Jul 5",  None, None, "NS", None),     # today 4pm ET
    ("Mexico",       "England",     "Jul 5",  None, None, "NS", None),     # today 8pm ET
    ("Portugal",     "Spain",       "Jul 6",  None, None, "NS", None),
    ("Belgium",      "USA",         "Jul 6",  None, None, "NS", None),
    ("Egypt",        "Argentina",   "Jul 7",  None, None, "NS", None),
    ("Switzerland",  "Colombia",    "Jul 7",  None, None, "NS", None),
]

R32_BRACKET = [
    ("South Africa","Canada",     "Jun 28"),
    ("Brazil",      "Japan",      "Jun 29"),
    ("Germany",     "Paraguay",   "Jun 29"),
    ("Netherlands", "Morocco",    "Jun 29"),
    ("Norway",      "Ivory Coast","Jun 30"),
    ("France",      "Sweden",     "Jun 30"),
    ("Mexico",      "Ecuador",    "Jun 30"),
    ("England",     "DR Congo",   "Jul 1"),
    ("Belgium",     "Senegal",    "Jul 1"),
    ("USA",         "Bosnia",     "Jul 1"),
    ("Spain",       "Austria",    "Jul 2"),
    ("Portugal",    "Croatia",    "Jul 2"),
    ("Switzerland", "Algeria",    "Jul 2"),
    ("Australia",   "Egypt",      "Jul 3"),
    ("Argentina",   "Cape Verde", "Jul 3"),
    ("Colombia",    "Ghana",      "Jul 3"),
]

# ── Player cards (EA FC 25, June 2026 patch) ──────────────────────────────────
# (name, team, pos, overall, pace, shooting, passing, dribbling, defending, physical)
PLAYER_CARDS_FALLBACK = [
    # Argentina
    ("L. Messi","Argentina","RW",94,82,91,92,96,35,66),
    ("E. Martínez","Argentina","GK",88,0,0,0,0,0,0),
    ("C. Romero","Argentina","CB",88,72,52,70,73,89,85),
    ("R. De Paul","Argentina","CM",85,74,74,86,84,76,79),
    ("J. Álvarez","Argentina","ST",85,86,84,77,86,45,78),
    ("L. Martínez","Argentina","ST",84,83,83,72,82,44,77),
    ("G. Lo Celso","Argentina","CM",83,75,73,85,86,65,72),
    ("N. Molina","Argentina","RB",83,83,67,75,79,75,76),
    ("E. Fernández","Argentina","CM",83,69,73,84,81,77,77),
    ("Á. Di María","Argentina","LW",82,82,79,83,86,35,64),
    ("L. Paredes","Argentina","CDM",81,62,65,84,77,73,73),
    # France
    ("K. Mbappé","France","ST",92,97,91,81,93,37,77),
    ("M. Maignan","France","GK",88,0,0,0,0,0,0),
    ("O. Dembélé","France","RW",87,93,82,80,90,36,70),
    ("A. Griezmann","France","CAM",87,76,85,87,86,55,74),
    ("J. Koundé","France","RB",86,79,53,76,80,84,75),
    ("A. Tchouaméni","France","CDM",85,71,64,82,77,85,84),
    ("A. Upamecano","France","CB",85,77,48,69,70,86,84),
    ("T. Hernández","France","LB",85,85,63,76,79,77,81),
    ("A. Camavinga","France","CM",85,81,69,83,85,76,77),
    ("K. Coman","France","RW",85,93,79,79,87,39,73),
    ("M. Thuram","France","ST",84,83,82,73,81,43,85),
    # Spain
    ("G. Rodri","Spain","CDM",91,66,72,89,84,89,79),
    ("P. Pedri","Spain","CM",89,75,76,89,92,66,69),
    ("L. Yamal","Spain","RW",87,92,81,81,91,30,62),
    ("D. Olmo","Spain","CAM",86,79,81,85,87,62,73),
    ("D. Carvajal","Spain","RB",86,76,65,83,79,83,77),
    ("A. Laporte","Spain","CB",85,69,51,79,73,87,81),
    ("M. Cucurella","Spain","LB",83,76,56,76,77,80,78),
    ("U. Simón","Spain","GK",85,0,0,0,0,0,0),
    ("A. Morata","Spain","ST",83,77,81,75,79,46,79),
    ("R. Le Normand","Spain","CB",83,69,45,71,66,84,82),
    ("F. Torres","Spain","ST",82,83,82,75,82,40,73),
    # England
    ("H. Kane","England","ST",91,71,94,84,84,48,83),
    ("P. Foden","England","CAM",89,83,84,87,91,45,70),
    ("B. Saka","England","RW",88,86,83,85,88,56,73),
    ("J. Bellingham","England","CM",88,80,85,84,87,74,85),
    ("T. Alexander-Arnold","England","RB",88,76,73,91,81,68,71),
    ("D. Rice","England","CDM",87,73,68,84,80,86,84),
    ("C. Palmer","England","CAM",85,74,84,84,88,52,70),
    ("J. Pickford","England","GK",85,0,0,0,0,0,0),
    ("O. Watkins","England","ST",84,83,82,72,80,46,79),
    ("J. Trippier","England","RB",83,73,69,84,75,75,74),
    ("M. Guehi","England","CB",83,69,45,69,67,84,81),
    # Brazil
    ("Vinicius Jr.","Brazil","LW",93,96,85,78,95,33,74),
    ("Alisson","Brazil","GK",91,0,0,0,0,0,0),
    ("E. Militão","Brazil","CB",87,77,49,69,73,88,81),
    ("Casemiro","Brazil","CDM",86,64,71,78,75,88,87),
    ("L. Paquetá","Brazil","CM",86,77,79,84,87,57,75),
    ("Rodrygo","Brazil","RW",86,88,82,79,88,36,69),
    ("Raphinha","Brazil","RW",86,85,83,81,86,41,72),
    ("G. Martinelli","Brazil","LW",85,91,81,74,85,39,78),
    ("A. Savinho","Brazil","RW",82,90,76,75,84,33,68),
    ("D. Danilo","Brazil","RB",82,77,61,78,74,78,78),
    ("M. Renan","Brazil","CB",81,71,43,66,66,82,80),
    # Germany
    ("F. Wirtz","Germany","CAM",89,80,82,87,91,49,71),
    ("J. Musiala","Germany","CAM",89,86,83,84,92,53,73),
    ("J. Kimmich","Germany","CDM",89,69,73,92,83,83,77),
    ("M. Ter Stegen","Germany","GK",90,0,0,0,0,0,0),
    ("A. Rüdiger","Germany","CB",87,79,53,66,67,88,89),
    ("L. Goretzka","Germany","CM",85,76,79,83,83,80,86),
    ("K. Havertz","Germany","ST",85,78,85,81,83,58,80),
    ("T. Müller","Germany","CAM",83,68,80,86,82,57,75),
    ("D. Raum","Germany","LB",83,81,61,79,75,75,74),
    ("B. Pavard","Germany","CB",83,75,56,74,70,83,79),
    ("L. Nmecha","Germany","ST",82,83,81,70,79,40,82),
    # Portugal
    ("R. Dias","Portugal","CB",89,73,53,71,71,90,83),
    ("C. Ronaldo","Portugal","ST",87,80,92,77,84,33,76),
    ("B. Fernandes","Portugal","CAM",87,75,83,89,85,61,78),
    ("R. Leão","Portugal","LW",86,93,82,77,87,35,75),
    ("N. Mendes","Portugal","LB",86,88,63,77,81,79,77),
    ("J. Cancelo","Portugal","RB",86,81,69,83,83,73,73),
    ("J. Félix","Portugal","CAM",85,83,82,82,88,41,71),
    ("D. Costa","Portugal","GK",85,0,0,0,0,0,0),
    ("G. Ramos","Portugal","ST",84,75,84,70,78,42,78),
    ("P. Neves","Portugal","CM",84,73,70,84,82,74,76),
    ("D. Dalot","Portugal","RB",83,81,61,75,77,75,73),
    # Netherlands
    ("V. van Dijk","Netherlands","CB",90,76,61,73,73,92,89),
    ("F. de Jong","Netherlands","CM",88,75,73,89,88,75,77),
    ("C. Gakpo","Netherlands","LW",85,86,83,77,84,43,79),
    ("X. Simons","Netherlands","CAM",85,84,80,84,86,52,70),
    ("M. Depay","Netherlands","ST",84,83,85,78,86,38,76),
    ("D. Dumfries","Netherlands","RB",84,85,66,73,74,77,85),
    ("B. Verbruggen","Netherlands","GK",84,0,0,0,0,0,0),
    ("S. de Vrij","Netherlands","CB",84,69,51,73,69,86,81),
    ("T. Reijnders","Netherlands","CM",84,76,73,81,81,72,77),
    ("N. Timber","Netherlands","CB",83,75,47,71,71,83,79),
    ("R. Malen","Netherlands","LW",82,89,78,72,83,35,70),
    # Morocco
    ("A. Hakimi","Morocco","RB",87,91,73,77,83,71,77),
    ("Y. Bounou","Morocco","GK",87,0,0,0,0,0,0),
    ("Y. En-Nesyri","Morocco","ST",83,84,82,65,76,43,78),
    ("H. Ziyech","Morocco","RW",83,79,81,84,85,41,66),
    ("S. Amrabat","Morocco","CDM",83,74,59,79,79,83,81),
    ("N. Aguerd","Morocco","CB",82,69,45,67,63,83,79),
    ("S. Ezzalzouli","Morocco","LW",81,86,74,76,82,36,68),
    ("I. Ounahi","Morocco","CM",80,73,66,79,80,70,72),
    ("R. Saïss","Morocco","CB",80,63,49,69,65,83,79),
    # Belgium
    ("K. De Bruyne","Belgium","CAM",91,75,86,94,88,65,79),
    ("T. Courtois","Belgium","GK",91,0,0,0,0,0,0),
    ("R. Lukaku","Belgium","ST",86,83,87,68,81,41,93),
    ("A. Doku","Belgium","LW",85,96,76,74,89,32,68),
    ("L. Trossard","Belgium","LW",84,83,81,79,83,47,73),
    ("J. Vertonghen","Belgium","CB",82,65,47,73,69,85,79),
    ("Y. Carrasco","Belgium","LW",82,85,77,77,84,41,73),
    ("A. Onana","Belgium","CDM",82,70,62,80,76,80,78),
    # Colombia
    ("L. Díaz","Colombia","LW",86,91,81,79,89,41,74),
    ("M. Caicedo","Colombia","CDM",85,75,63,79,79,85,83),
    ("J. Cuadrado","Colombia","RW",83,85,75,81,83,55,71),
    ("Y. Mina","Colombia","CB",83,65,53,61,59,84,85),
    ("D. Ospina","Colombia","GK",84,0,0,0,0,0,0),
    ("J. Lerma","Colombia","CDM",81,71,61,75,71,81,83),
    ("J. Muriel","Colombia","ST",81,85,79,73,79,37,73),
    ("R. Falcao","Colombia","ST",81,69,84,70,78,38,74),
    # Norway
    ("E. Haaland","Norway","ST",95,89,96,69,82,46,89),
    ("M. Ødegaard","Norway","CAM",88,79,83,91,91,59,69),
    ("A. Sørloth","Norway","ST",83,79,83,67,73,47,87),
    ("J. Strand Larsen","Norway","ST",80,79,79,63,73,39,83),
    ("S. Berge","Norway","CM",80,69,67,79,75,77,75),
    ("R. Nyland","Norway","GK",79,0,0,0,0,0,0),
    # Switzerland
    ("Y. Sommer","Switzerland","GK",86,0,0,0,0,0,0),
    ("G. Xhaka","Switzerland","CDM",85,65,73,86,78,81,79),
    ("M. Akanji","Switzerland","CB",85,77,49,71,71,86,81),
    ("X. Shaqiri","Switzerland","RW",83,81,81,83,84,43,73),
    ("F. Schär","Switzerland","CB",82,69,57,73,67,83,79),
    ("B. Embolo","Switzerland","ST",81,81,79,73,77,45,83),
    ("R. Vargas","Switzerland","LW",80,83,75,77,79,37,67),
    # USA
    ("C. Pulisic","USA","LW",85,89,81,82,87,45,75),
    ("T. Adams","USA","CDM",82,73,59,77,73,83,83),
    ("W. McKennie","USA","CM",81,77,71,76,76,73,83),
    ("R. Reyna","USA","CAM",81,81,74,80,84,44,66),
    ("M. Turner","USA","GK",80,0,0,0,0,0,0),
    ("J. Sargent","USA","ST",80,77,78,67,74,45,77),
    ("A. Dest","USA","RB",80,85,61,73,77,65,69),
    ("F. Weah","USA","RW",79,87,73,71,78,37,73),
    # Mexico
    ("G. Ochoa","Mexico","GK",84,0,0,0,0,0,0),
    ("H. Lozano","Mexico","RW",83,91,79,77,83,39,71),
    ("R. Jiménez","Mexico","ST",83,74,83,71,77,43,81),
    ("A. Vega","Mexico","LW",80,87,75,73,80,36,69),
    ("H. Moreno","Mexico","CB",80,67,45,67,65,81,79),
    ("E. Álvarez","Mexico","CM",79,71,67,77,75,71,73),
    # Japan
    ("W. Endo","Japan","CDM",80,67,61,77,73,80,77),
    ("D. Kamada","Japan","CAM",80,77,77,79,80,57,71),
    ("T. Minamino","Japan","CAM",80,81,78,79,81,55,69),
    ("Ritsu Doan","Japan","LW",80,84,77,77,81,43,68),
    ("H. Ito","Japan","RB",79,81,59,71,71,71,71),
    ("S. Itakura","Japan","CB",79,69,45,67,65,80,77),
    ("K. Gonda","Japan","GK",78,0,0,0,0,0,0),
    ("A. Ueda","Japan","ST",78,78,77,63,71,39,75),
    # Austria
    ("D. Alaba","Austria","CB",86,77,71,85,83,85,75),
    ("M. Sabitzer","Austria","CM",83,77,77,81,81,67,77),
    ("M. Arnautovic","Austria","ST",82,73,81,73,75,45,81),
    ("C. Baumgartner","Austria","CM",80,77,73,77,79,59,71),
    ("P. Pentz","Austria","GK",77,0,0,0,0,0,0),
    # Sweden
    ("V. Gyökeres","Sweden","ST",86,83,87,69,82,43,86),
    ("A. Isak","Sweden","ST",85,89,83,73,85,39,77),
    ("D. Kulusevski","Sweden","RW",84,85,79,81,84,55,76),
    ("E. Forsberg","Sweden","CAM",82,79,77,83,83,51,69),
    ("R. Olsen","Sweden","GK",80,0,0,0,0,0,0),
    # Canada
    ("A. Davies","Canada","LB",86,96,69,79,84,73,75),
    ("J. David","Canada","ST",84,83,86,73,82,39,73),
    ("T. Buchanan","Canada","LW",79,87,71,71,77,37,71),
    ("M. Borjan","Canada","GK",78,0,0,0,0,0,0),
    ("S. Larin","Canada","ST",78,77,77,65,71,39,75),
    # Senegal
    ("S. Mané","Senegal","LW",87,90,84,79,88,44,78),
    ("É. Mendy","Senegal","GK",86,0,0,0,0,0,0),
    ("K. Koulibaly","Senegal","CB",86,73,53,65,63,89,87),
    ("I. Sarr","Senegal","RW",83,91,77,73,81,37,73),
    ("P. Gueye","Senegal","CDM",83,75,63,77,75,83,81),
    ("N. Jackson","Senegal","ST",81,84,79,71,77,39,77),
    # Croatia
    ("J. Gvardiol","Croatia","CB",88,83,57,73,77,88,83),
    ("L. Modrić","Croatia","CM",87,73,77,92,91,63,68),
    ("D. Livaković","Croatia","GK",85,0,0,0,0,0,0),
    ("M. Brozović","Croatia","CDM",84,69,69,87,81,79,75),
    ("A. Kramarić","Croatia","ST",83,77,83,73,81,43,75),
    ("I. Perišić","Croatia","LW",83,83,78,78,80,64,80),
    # Ivory Coast
    ("S. Haller","Ivory Coast","ST",82,75,83,61,73,43,87),
    ("W. Zaha","Ivory Coast","RW",81,87,75,73,83,39,73),
    ("N. Pépé","Ivory Coast","RW",80,87,77,75,84,37,71),
    ("S. Aurier","Ivory Coast","RB",80,83,63,71,71,73,77),
    ("F. Konaté","Ivory Coast","GK",78,0,0,0,0,0,0),
    # Australia
    ("M. Ryan","Australia","GK",80,0,0,0,0,0,0),
    ("M. Leckie","Australia","RW",78,85,71,73,77,43,71),
    ("H. Souttar","Australia","CB",78,63,45,65,61,79,81),
    ("A. Hrustic","Australia","CM",76,71,69,75,73,63,67),
    # Egypt
    ("M. Salah","Egypt","RW",91,88,89,83,90,45,78),
    ("O. Marmoush","Egypt","ST",86,85,84,77,82,46,77),
    ("M. El-Shenawy","Egypt","GK",79,0,0,0,0,0,0),
    ("A. Hegazi","Egypt","CB",78,63,45,63,59,79,81),
    # Ecuador
    ("M. Caicedo","Ecuador","CDM",85,75,63,79,79,85,83),
    ("P. Hincapié","Ecuador","CB",81,75,47,67,67,81,77),
    ("E. Valencia","Ecuador","ST",80,77,78,63,69,43,83),
    ("H. Galíndez","Ecuador","GK",77,0,0,0,0,0,0),
    # Ghana
    ("T. Partey","Ghana","CDM",84,73,66,81,77,84,83),
    ("A. Kudus","Ghana","CAM",82,85,78,78,84,48,74),
    ("J. Ayew","Ghana","ST",80,79,77,75,77,47,73),
    ("D. Amartey","Ghana","CB",77,69,45,63,61,78,77),
    ("L. Bati","Ghana","GK",73,0,0,0,0,0,0),
    # Senegal already done; remaining teams
    ("South Africa","South Africa","—",0,0,0,0,0,0,0),  # placeholder removed below
    # South Africa
    ("P. Tau","South Africa","LW",78,83,71,73,79,37,65),
    ("R. Williams","South Africa","GK",75,0,0,0,0,0,0),
    ("T. Lorch","South Africa","RW",75,83,67,65,75,33,63),
    ("B. Zwane","South Africa","CM",74,77,63,69,73,55,65),
    # Bosnia
    ("E. Džeko","Bosnia","ST",83,69,83,73,77,41,79),
    ("M. Pjanić","Bosnia","CM",82,65,73,88,81,59,69),
    ("S. Kolasinac","Bosnia","LB",80,73,57,69,67,77,83),
    ("I. Šehić","Bosnia","GK",77,0,0,0,0,0,0),
    # Iran
    ("M. Taremi","Iran","ST",83,77,85,73,79,45,79),
    ("S. Azmoun","Iran","ST",81,77,81,71,77,43,79),
    ("A. Beiranvand","Iran","GK",79,0,0,0,0,0,0),
    # Paraguay
    ("M. Almiron","Paraguay","CAM",81,83,73,77,81,53,71),
    ("J. Enciso","Paraguay","RW",79,85,73,73,79,35,65),
    ("R. Sanabria","Paraguay","ST",78,73,77,65,73,39,73),
    ("A. Silva","Paraguay","GK",75,0,0,0,0,0,0),
    # Cape Verde
    ("G. Benchimol","Cape Verde","ST",73,75,71,61,69,37,71),
    ("J. Landim","Cape Verde","RW",71,77,65,63,69,33,63),
    ("V. Varela","Cape Verde","GK",69,0,0,0,0,0,0),
    # Uzbekistan
    ("E. Shomurodov","Uzbekistan","ST",77,77,75,63,71,39,73),
    ("O. Zubidov","Uzbekistan","CM",69,65,59,67,65,61,63),
    ("U. Nishonov","Uzbekistan","GK",69,0,0,0,0,0,0),
]
# Remove the placeholder row
PLAYER_CARDS_FALLBACK = [r for r in PLAYER_CARDS_FALLBACK if r[0] != "South Africa"]

WC_HISTORY = [
    (1930,"Uruguay","🇺🇾","Uruguay","🇺🇾","Argentina","🇦🇷","4–2","G. Stábile (ARG)",8,434000),
    (1934,"Italy","🇮🇹","Italy","🇮🇹","Czechoslovakia","🇨🇿","2–1","O. Schiavio (ITA)",4,363000),
    (1938,"France","🇫🇷","Italy","🇮🇹","Hungary","🇭🇺","4–2","L. Zsengellér (HUN)",7,375000),
    (1950,"Brazil","🇧🇷","Uruguay","🇺🇾","Brazil","🇧🇷","2–1","A. Ademir (BRA)",9,1045000),
    (1954,"Switzerland","🇨🇭","West Germany","🇩🇪","Hungary","🇭🇺","3–2","S. Morlock (GER)",6,768000),
    (1958,"Sweden","🇸🇪","Brazil","🇧🇷","Sweden","🇸🇪","5–2","Just Fontaine (FRA)",13,868000),
    (1962,"Chile","🇨🇱","Brazil","🇧🇷","Czechoslovakia","🇨🇿","3–1","V. Masek (TCH)",4,893000),
    (1966,"England","🏴󠁧󠁢󠁥󠁮󠁧󠁿","England","🏴󠁧󠁢󠁥󠁮󠁧󠁿","West Germany","🇩🇪","4–2","E. Hurst (ENG)",4,1614677),
    (1970,"Mexico","🇲🇽","Brazil","🇧🇷","Italy","🇮🇹","4–1","G. Müller (GER)",10,1604065),
    (1974,"West Germany","🇩🇪","West Germany","🇩🇪","Netherlands","🇳🇱","2–1","G. Müller (GER)",4,1865762),
    (1978,"Argentina","🇦🇷","Argentina","🇦🇷","Netherlands","🇳🇱","3–1","M. Kempes (ARG)",6,1545791),
    (1982,"Spain","🇪🇸","Italy","🇮🇹","West Germany","🇩🇪","3–1","P. Rossi (ITA)",6,2109723),
    (1986,"Mexico","🇲🇽","Argentina","🇦🇷","West Germany","🇩🇪","3–2","G. Lineker (ENG)",6,2394031),
    (1990,"Italy","🇮🇹","West Germany","🇩🇪","Argentina","🇦🇷","1–0","S. Schillaci (ITA)",6,2517348),
    (1994,"USA","🇺🇸","Brazil","🇧🇷","Italy","🇮🇹","0–0 (3–2p)","H. Stoichkov (BUL)",6,3587538),
    (1998,"France","🇫🇷","France","🇫🇷","Brazil","🇧🇷","3–0","D. Batistuta (ARG)",6,2785100),
    (2002,"Korea/Japan","🇰🇷🇯🇵","Brazil","🇧🇷","Germany","🇩🇪","2–0","R. Ronaldo (BRA)",8,2705197),
    (2006,"Germany","🇩🇪","Italy","🇮🇹","France","🇫🇷","1–1 (5–3p)","M. Klose (GER)",5,3359439),
    (2010,"South Africa","🇿🇦","Spain","🇪🇸","Netherlands","🇳🇱","1–0","T. Müller (GER)",5,3178856),
    (2014,"Brazil","🇧🇷","Germany","🇩🇪","Argentina","🇦🇷","1–0","J. Rodríguez (COL)",6,3429873),
    (2018,"Russia","🇷🇺","France","🇫🇷","Croatia","🇭🇷","4–2","H. Kane (ENG)",6,3031768),
    (2022,"Qatar","🇶🇦","Argentina","🇦🇷","France","🇫🇷","3–3 (4–2p)","K. Mbappé (FRA)",8,3404252),
]

# ══════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════
def api_get(url: str, key: str) -> dict | None:
    try:
        r = requests.get(url, headers={"x-apisports-key": key}, timeout=12)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def fetch_live_data(key: str) -> dict | None:
    return api_get("https://v3.football.api-sports.io/status", key)

def parse_account_info(status: dict) -> tuple:
    resp = (status or {}).get("response") or {}
    if isinstance(resp, list):
        resp = resp[0] if resp else {}
    plan  = (resp.get("subscription") or {}).get("plan", "unknown")
    reqs  = (resp.get("requests") or {})
    return plan, str(reqs.get("current","?")), str(reqs.get("limit_day","?"))

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent_results(key: str) -> tuple:
    league_seasons = [
        (1,2026,"WC 2026"),(1,2022,"WC 2022"),
        (4,2024,"Euro 2024"),(9,2024,"Copa Am 2024"),
        (6,2023,"AFCON 2023"),(10,2023,"AFC Asian 2023"),
        (834,2025,"Intl Friendlies 25"),(834,2024,"Intl Friendlies 24"),
    ]
    cutoff  = (datetime.now()-timedelta(days=730)).strftime("%Y-%m-%d")
    results, debug = [], []
    for lid, season, label in league_seasons:
        data      = api_get(f"https://v3.football.api-sports.io/fixtures?league={lid}&season={season}&status=FT", key)
        raw       = (data or {}).get("response") or []
        matched   = 0
        for fix in raw:
            ds  = (fix.get("fixture") or {}).get("date","")[:10]
            if ds < cutoff: continue
            hn  = (fix.get("teams") or {}).get("home",{}).get("name","")
            an  = (fix.get("teams") or {}).get("away",{}).get("name","")
            gh  = (fix.get("goals") or {}).get("home")
            ga  = (fix.get("goals") or {}).get("away")
            h   = API_NAME_MAP.get(hn); a = API_NAME_MAP.get(an)
            if h and a and gh is not None and ga is not None:
                results.append({"home":h,"away":a,"goals_home":int(gh),"goals_away":int(ga),"date":ds})
                matched += 1
        debug.append(f"{label}: {len(raw)} fixtures, {matched} matched")
    debug.append(f"Total: {len(results)}")
    return results, debug

def compute_ratings(results: list) -> dict:
    now = datetime.now()
    sc: dict = {t:[] for t in BASE_TEAMS}
    co: dict = {t:[] for t in BASE_TEAMS}
    for r in results:
        d = (now - datetime.strptime(r["date"],"%Y-%m-%d")).days
        w = 1.0 if d<=180 else (0.7 if d<=365 else 0.4)
        if r["home"] in sc:
            sc[r["home"]].append((r["goals_home"],w))
            co[r["home"]].append((r["goals_away"],w))
        if r["away"] in sc:
            sc[r["away"]].append((r["goals_away"],w))
            co[r["away"]].append((r["goals_home"],w))
    out = {}
    for t, base in BASE_TEAMS.items():
        s = sc.get(t,[])
        if len(s) >= 3:
            tw  = sum(w for _,w in s)
            atk = sum(g*w for g,w in s)/tw
            dfg = sum(g*w for g,w in co.get(t,[]))/tw
            out[t] = {**base,"atk":round(0.6*atk+0.4*base["atk"],3),"def":round(0.6*(dfg/1.35)+0.4*base["def"],3)}
        else:
            out[t] = base
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def discover_league_id(key: str) -> tuple:
    debug = []
    for name in ["FIFA World Cup","World Cup"]:
        data = api_get(f"https://v3.football.api-sports.io/leagues?name={name.replace(' ','+')}&season=2026", key)
        resp = (data or {}).get("response") or []
        debug.append(f"Search '{name}': {len(resp)} results")
        for item in resp:
            lg = (item.get("league") or {}); lid = lg.get("id")
            debug.append(f"  id={lid} name={lg.get('name','')}")
            if lid: return lid, "\n".join(debug)
    for s in [2026,2025]:
        data = api_get(f"https://v3.football.api-sports.io/leagues?id=1&season={s}", key)
        resp = (data or {}).get("response") or []
        debug.append(f"id=1 season={s}: {len(resp)} results")
        if resp: return 1, "\n".join(debug)
    data = api_get("https://v3.football.api-sports.io/leagues?type=Cup&season=2026", key)
    resp = (data or {}).get("response") or []
    debug.append(f"All Cup leagues 2026: {len(resp)}")
    for item in resp:
        lg = (item.get("league") or {}); nm = lg.get("name","").lower(); lid = lg.get("id")
        debug.append(f"  id={lid} name={lg.get('name','')}")
        if "world" in nm and lid: return lid, "\n".join(debug)
    data = api_get("https://v3.football.api-sports.io/leagues?current=true", key)
    resp = (data or {}).get("response") or []
    debug.append(f"All current leagues: {len(resp)}")
    for item in resp:
        lg = (item.get("league") or {}); nm = lg.get("name","").lower(); lid = lg.get("id")
        if "world" in nm or "fifa" in nm:
            debug.append(f"  CANDIDATE id={lid} name={lg.get('name','')}")
    debug.append("Fallback: league_id=1")
    return 1, "\n".join(debug)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_top_scorers(key: str, lid: int) -> tuple:
    debug = [f"league_id={lid} season={WC2026_SEASON}"]

    def build_squads(tg: dict) -> dict:
        out = {}
        for team, players in tg.items():
            total = sum(g for _,g in players)
            if not total: continue
            top4  = sorted(players,key=lambda x:-x[1])[:4]
            ts    = sum(g for _,g in top4)/total
            sl    = [(n,round(g/total,3)) for n,g in top4]
            if ts < 1.0: sl.append(("Other",round(1.0-ts,3)))
            out[team] = sl
        return out

    # Strategy 1: topscorers endpoint
    data = api_get(f"https://v3.football.api-sports.io/players/topscorers?league={lid}&season={WC2026_SEASON}", key)
    resp = (data or {}).get("response") or []
    debug.append(f"S1 /topscorers: {len(resp)} entries")
    tg1: dict = defaultdict(list)
    for e in resp:
        p=e.get("player") or {}; stats=((e.get("statistics") or [{}])[0])
        nm=p.get("name",""); g=int((stats.get("goals") or {}).get("total") or 0)
        tm=API_NAME_MAP.get((stats.get("team") or {}).get("name",""))
        if tm and nm and g>0: tg1[tm].append((nm,g))
    if tg1:
        debug.append(f"✅ S1 success: {len(tg1)} teams")
        return build_squads(tg1), "\n".join(debug)
    debug.append("S1 empty → S2")

    # Strategy 2: players list
    url2 = f"https://v3.football.api-sports.io/players?league={lid}&season={WC2026_SEASON}&page=1"
    d2   = api_get(url2, key)
    r2   = (d2 or {}).get("response") or []
    pages = (d2 or {}).get("paging",{}).get("total",1) if d2 else 1
    for pg in range(2,min(pages+1,6)):
        more = api_get(url2.replace("page=1",f"page={pg}"),key)
        if more and more.get("response"): r2.extend(more["response"])
    debug.append(f"S2 /players: {len(r2)} entries over {pages} pages")
    tg2: dict = defaultdict(list)
    for e in r2:
        p=e.get("player") or {}; stats=((e.get("statistics") or [{}])[0])
        nm=p.get("name",""); g=int((stats.get("goals") or {}).get("total") or 0)
        tm=API_NAME_MAP.get((stats.get("team") or {}).get("name",""))
        if tm and nm and g>0: tg2[tm].append((nm,g))
    if tg2:
        debug.append(f"✅ S2 success: {len(tg2)} teams")
        return build_squads(tg2), "\n".join(debug)
    debug.append("S2 empty → S3")

    # Strategy 3: fixture events
    fdata    = api_get(f"https://v3.football.api-sports.io/fixtures?league={lid}&season={WC2026_SEASON}&status=FT", key)
    fixtures = (fdata or {}).get("response") or []
    debug.append(f"S3: {len(fixtures)} finished fixtures")
    raw: dict = defaultdict(lambda:{"goals":0,"team":""})
    for fix in fixtures:
        fid = (fix.get("fixture") or {}).get("id","")
        if not fid: continue
        for ev in (api_get(f"https://v3.football.api-sports.io/fixtures/events?fixture={fid}", key) or {}).get("response") or []:
            if (ev.get("type") or "").lower()!="goal": continue
            det = (ev.get("detail") or "").lower()
            if "own" in det or "penalty" in det: continue
            pn = (ev.get("player") or {}).get("name","")
            tm = API_NAME_MAP.get((ev.get("team") or {}).get("name",""))
            if pn and tm:
                raw[pn]["goals"]+=1; raw[pn]["team"]=tm
    tg3: dict = defaultdict(list)
    for pn,info in raw.items():
        if info["goals"]>0 and info["team"]: tg3[info["team"]].append((pn,info["goals"]))
    debug.append(f"S3: {sum(len(v) for v in tg3.values())} records")
    if tg3:
        debug.append(f"✅ S3 success: {len(tg3)} teams")
        return build_squads(tg3), "\n".join(debug)

    debug.append("❌ All strategies failed. Check API plan / tournament progress.")
    return {}, "\n".join(debug)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_wc_results(key: str, lid: int) -> tuple:
    debug = []
    fin   = {"FT","AET","PEN","AWD","WO"}
    data  = api_get(f"https://v3.football.api-sports.io/fixtures?league={lid}&season={WC2026_SEASON}", key)
    raw   = (data or {}).get("response") or []
    debug.append(f"Total fixtures: {len(raw)}")
    matches = []
    for fix in raw:
        f=fix.get("fixture") or {}; teams=fix.get("teams") or {}
        goals=fix.get("goals") or {}; score=fix.get("score") or {}
        status=(f.get("status") or {}).get("short","NS")
        hn=API_NAME_MAP.get((teams.get("home") or {}).get("name",""),
                            (teams.get("home") or {}).get("name",""))
        an=API_NAME_MAP.get((teams.get("away") or {}).get("name",""),
                            (teams.get("away") or {}).get("name",""))
        gh=goals.get("home"); ga=goals.get("away")
        winner=None
        if status in fin and gh is not None and ga is not None:
            if gh>ga: winner=hn
            elif ga>gh: winner=an
            else:
                pen=(score.get("penalty") or {}); ph=pen.get("home"); pa=pen.get("away")
                if ph is not None and pa is not None: winner=hn if ph>pa else an
        matches.append({"home":hn,"away":an,"goals_home":int(gh) if gh is not None else None,
                        "goals_away":int(ga) if ga is not None else None,
                        "status":status,"winner":winner})
    done = sum(1 for m in matches if m["status"] in fin)
    live = sum(1 for m in matches if m["status"] in {"1H","2H","HT","ET","BT","P","INT"})
    debug.append(f"Done:{done} Live:{live} Upcoming:{len(matches)-done-live}")
    for m in [x for x in matches if x["status"] in fin][:6]:
        debug.append(f"  {m['home']} {m['goals_home']}–{m['goals_away']} {m['away']} [{m['status']}] → {m['winner']}")
    return matches, "\n".join(debug)

def build_live_bracket(api_matches: list) -> list:
    """API results first, hardcoded R32_RESULTS as fallback so scores always show."""
    lookup: dict = {}
    for m in api_matches:
        lookup[(m["home"].lower(), m["away"].lower())] = m
        lookup[(m["away"].lower(), m["home"].lower())] = m

    # Supplement with hardcoded results where API has nothing
    for home, away, date, gh, ga, status, winner in R32_RESULTS:
        key = (home.lower(), away.lower())
        if key not in lookup:
            lookup[key] = {"home":home,"away":away,"goals_home":gh,
                           "goals_away":ga,"status":status,"winner":winner}
            lookup[(away.lower(), home.lower())] = lookup[key]

    out = []
    for home, away, date in R32_BRACKET:
        live = lookup.get((home.lower(), away.lower()))
        if live:
            w = live.get("winner")
            gh_v = live.get("goals_home")
            ga_v = live.get("goals_away")
            # Flip if stored in reverse direction
            if live["home"].lower() == away.lower():
                gh_v, ga_v = ga_v, gh_v
                if w == live["home"]: w = away
                elif w == live["away"]: w = home
            out.append({"home":home,"away":away,"date":date,
                        "goals_home":gh_v,"goals_away":ga_v,
                        "status":live["status"],"winner":w,
                        "flag_h":BASE_TEAMS.get(home,{}).get("flag","🏳️"),
                        "flag_a":BASE_TEAMS.get(away,{}).get("flag","🏳️")})
        else:
            out.append({"home":home,"away":away,"date":date,
                        "goals_home":None,"goals_away":None,
                        "status":"NS","winner":None,
                        "flag_h":BASE_TEAMS.get(home,{}).get("flag","🏳️"),
                        "flag_a":BASE_TEAMS.get(away,{}).get("flag","🏳️")})
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_cards_api(key: str) -> list:
    url  = f"https://v3.football.api-sports.io/players?league={WC2026_LEAGUE_ID}&season={WC2026_SEASON}&page=1"
    data = api_get(url, key)
    if not data or not data.get("response"): return []
    pages = (data.get("paging") or {}).get("total",1)
    resp  = list(data["response"])
    for pg in range(2,min(pages+1,8)):
        more = api_get(url.replace("page=1",f"page={pg}"),key)
        if more and more.get("response"): resp.extend(more["response"])
    pm  = {"Goalkeeper":"GK","Defender":"CB","Midfielder":"CM","Attacker":"ST"}
    out = []
    for e in resp:
        p=e.get("player") or {}; stats=((e.get("statistics") or [{}])[0])
        nm=p.get("name",""); tm=API_NAME_MAP.get((stats.get("team") or {}).get("name",""))
        if not tm or not nm: continue
        pos=pm.get((stats.get("games") or {}).get("position","Midfielder"),"CM")
        try: ovr=max(60,min(99,round(float((stats.get("games") or {}).get("rating") or 0)*10))) if (stats.get("games") or {}).get("rating") else 75
        except: ovr=75
        out.append((nm,tm,pos,ovr,0,0,0,0,0,0))
    return out

# ══════════════════════════════════════════════════════════════════════
# CARD RENDER
# ══════════════════════════════════════════════════════════════════════
def card_color(ovr: int) -> tuple:
    if ovr>=90: return ("linear-gradient(145deg,#b8860b,#ffd700,#b8860b)","#ffd700","#3d2800")
    if ovr>=86: return ("linear-gradient(145deg,#8b7536,#d4af37,#8b7536)","#d4af37","#2a1f00")
    if ovr>=82: return ("linear-gradient(145deg,#7a7a7a,#c0c0c0,#7a7a7a)","#c0c0c0","#1a1a1a")
    return ("linear-gradient(145deg,#5a3a1a,#a0714a,#5a3a1a)","#a0714a","#f5e6d0")

def render_card(p: dict) -> str:
    bg,border,tc = card_color(p["overall"])
    flag = BASE_TEAMS.get(p["team"],{}).get("flag","🏳️")
    labs = ["DIV","HAN","KIC","REF","SPD","POS"] if p["pos"]=="GK" else ["PAC","SHO","PAS","DRI","DEF","PHY"]
    vals = [p["pace"],p["shooting"],p["passing"],p["dribbling"],p["defending"],p["physical"]]
    rows = []
    for l,v in zip(labs,vals):
        if v==0: continue
        rows.append('<div style="display:flex;align-items:center;gap:3px;margin:1px 0;">'
            +'<span style="font-size:8px;font-weight:800;width:24px;color:'+tc+';opacity:.85;">'+l+'</span>'
            +'<div style="flex:1;background:rgba(0,0,0,.2);border-radius:2px;height:4px;">'
            +'<div style="width:'+str(v)+'%;background:'+tc+';height:4px;border-radius:2px;opacity:.9;"></div>'
            +'</div>'
            +'<span style="font-size:9px;font-weight:700;width:20px;text-align:right;color:'+tc+';">'+str(v)+'</span>'
            +'</div>')
    sh = "".join(rows)
    nm = ('<div style="font-size:9px;opacity:.6;text-align:center;color:'+tc+';">stats in built-in mode</div>'
          if all(v==0 for v in vals) else "")
    return ('<div style="background:'+bg+';border:2px solid '+border+';border-radius:10px;'
        +'padding:8px 8px 6px;width:140px;min-height:210px;box-shadow:0 3px 12px rgba(0,0,0,.4);'
        +'font-family:inherit;display:flex;flex-direction:column;align-items:center;'
        +'position:relative;box-sizing:border-box;">'
        +'<div style="font-size:26px;font-weight:900;color:'+tc+';line-height:1;">'+str(p["overall"])+'</div>'
        +'<div style="font-size:9px;font-weight:700;color:'+tc+';opacity:.8;letter-spacing:1px;">'+p["pos"]+'</div>'
        +'<div style="font-size:18px;margin:3px 0 1px;">'+flag+'</div>'
        +'<div style="font-size:10px;font-weight:800;color:'+tc+';text-align:center;'
        +'line-height:1.2;margin-bottom:5px;max-width:118px;word-break:break-word;">'+p["name"]+'</div>'
        +'<div style="width:100%;border-top:1px solid rgba(0,0,0,.2);padding-top:4px;">'+sh+nm+'</div>'
        +'<div style="position:absolute;bottom:4px;right:6px;font-size:7px;color:'+tc+';opacity:.4;">'+p["team"][:10]+'</div>'
        +'</div>')

# ══════════════════════════════════════════════════════════════════════
# RUNTIME STATE
# ══════════════════════════════════════════════════════════════════════
TEAMS          = dict(BASE_TEAMS)
HOST_ADVANTAGE = dict(DEFAULT_HOST_ADV)
TEAM_BOOST:    dict = {}
SQUADS         = dict(FALLBACK_SQUADS)

# ══════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════
def get_team(n: str) -> dict:
    return TEAMS.get(n,{"flag":"🏳️","atk":1.25,"def":0.91,"elo":1650})

def xg(a: str, b: str) -> tuple:
    ta=get_team(a); tb=get_team(b)
    return (ta["atk"]*tb["def"]*(1+HOST_ADVANTAGE.get(a,0))*TEAM_BOOST.get(a,1.0),
            tb["atk"]*ta["def"]*(1+HOST_ADVANTAGE.get(b,0))*TEAM_BOOST.get(b,1.0))

def sim_penalties(a: str, b: str) -> str:
    pa=PENALTY_WIN_RATE.get(a,.5); pb=PENALTY_WIN_RATE.get(b,.5)
    return a if random.random()<pa/(pa+pb) else b

def dist_goals(team: str, n: int) -> dict:
    ps=SQUADS.get(team,[("Other",1.0)]); ns=[p[0] for p in ps]
    w=np.array([p[1] for p in ps],dtype=float); w/=w.sum()
    t: dict = defaultdict(int)
    for _ in range(n): t[np.random.choice(ns,p=w)]+=1
    return dict(t)

def sim_match(a: str, b: str) -> tuple:
    xa,xb=xg(a,b)
    ga=int(np.random.poisson(xa)); gb=int(np.random.poisson(xb))
    if ga==gb:
        ga+=int(np.random.poisson(xa*.33)); gb+=int(np.random.poisson(xb*.33))
        if ga==gb:
            w=sim_penalties(a,b)
            return w,ga,gb,dist_goals(a,ga),dist_goals(b,gb)
    w=a if ga>gb else b
    return w,ga,gb,dist_goals(a,ga),dist_goals(b,gb)

def h2h(a: str, b: str, n: int = 0) -> dict:
    """
    Closed-form Poisson win/draw probabilities — no simulation loop needed.
    P(A scores i, B scores j) = Poisson(xg_a,i) * Poisson(xg_b,j)
    Sums over grid up to 10 goals each. Runs in microseconds.
    n param kept for API compatibility but ignored.
    """
    xa, xb = xg(a, b)
    from scipy.stats import poisson as sp
    k = np.arange(11)
    pa = sp.pmf(k, xa)   # P(A scores 0..10)
    pb = sp.pmf(k, xb)   # P(B scores 0..10)
    # Outer product → joint probability matrix
    joint = np.outer(pa, pb)   # joint[i,j] = P(A=i, B=j)
    win_a = float(np.tril(joint, -1).sum())   # A scores more
    win_b = float(np.triu(joint, 1).sum())    # B scores more
    draw  = float(np.diag(joint).sum())       # equal
    tot   = win_a + win_b + draw
    return {
        "team_a":  round(win_a / tot * 100, 2),
        "team_b":  round(win_b / tot * 100, 2),
        "draw_90": round(draw  / tot * 100, 2),
    }

def predict_score(a: str, b: str) -> tuple:
    """
    Return most likely scoreline and top-3 scorelines with probabilities.
    Uses joint Poisson PMF grid.
    """
    from scipy.stats import poisson as sp
    xa, xb = xg(a, b)
    k = np.arange(9)
    pa = sp.pmf(k, xa)
    pb = sp.pmf(k, xb)
    joint = np.outer(pa, pb)
    # Most likely scoreline
    idx = np.unravel_index(np.argmax(joint), joint.shape)
    most_likely = (int(idx[0]), int(idx[1]), float(joint[idx]) * 100)
    # Top 5 scorelines
    flat = [(int(i), int(j), float(joint[i, j]) * 100)
            for i in range(9) for j in range(9)]
    flat.sort(key=lambda x: -x[2])
    return most_likely, flat[:5]

@st.cache_data(show_spinner=False)
def h2h_cached(a: str, b: str, thash: str) -> dict:
    """Cached wrapper — thash ensures cache invalidates when team ratings change."""
    return h2h(a, b)

@st.cache_data(show_spinner=False)
def bracket_probs_cached(thash: str) -> list:
    """
    Pre-compute all R32 and R16 match probabilities in one go.
    Returns list of dicts keyed by (home, away).
    Only computed for unplayed matches; finished ones return None.
    """
    results = {}
    all_matches = list(R32_BRACKET) + [(h, a, d) for h, a, d, *_ in R16_FIXTURES]
    for home, away, date in all_matches:
        # Check if already finished
        finished = any(
            (r[0]==home and r[1]==away and r[6] is not None)
            or (r[1]==home and r[0]==away and r[6] is not None)
            for r in R32_RESULTS
        )
        r16_finished = any(
            (f[0]==home and f[1]==away and f[6] is not None)
            or (f[1]==home and f[0]==away and f[6] is not None)
            for f in R16_FIXTURES
        )
        if not finished and not r16_finished and away not in ("TBD",):
            results[(home, away)] = h2h(home, away)
            results[(home, away)]["score"] = predict_score(home, away)
    return results

def sim_tournament_fast(rng: np.random.Generator) -> tuple:
    """Vectorised tournament — uses numpy RNG for speed."""
    sur: list = []
    pg: dict  = defaultdict(int)

    for home, away, _ in R32_BRACKET:
        if away == "TBD":
            sur.append(home)
            continue
        # Check if real result known
        real = next((r for r in R32_RESULTS
                     if (r[0]==home and r[1]==away) or (r[1]==home and r[0]==away)), None)
        if real:
            sur.append(real[6])   # use real winner
            continue
        xa, xb = xg(home, away)
        ga = int(rng.poisson(xa)); gb = int(rng.poisson(xb))
        if ga == gb:
            ga += int(rng.poisson(xa * .33)); gb += int(rng.poisson(xb * .33))
            if ga == gb:
                p_a = PENALTY_WIN_RATE.get(home, .5); p_b = PENALTY_WIN_RATE.get(away, .5)
                w = home if rng.random() < p_a / (p_a + p_b) else away
            else:
                w = home if ga > gb else away
        else:
            w = home if ga > gb else away
        # Distribute goals to players
        for team, n_g in ((home, ga), (away, gb)):
            players = SQUADS.get(team, [("Other", 1.0)])
            names   = [p[0] for p in players]
            wts     = np.array([p[1] for p in players], dtype=float); wts /= wts.sum()
            for _ in range(n_g):
                pg[names[int(rng.choice(len(names), p=wts))]] += 1
        sur.append(w)

    while len(sur) > 1:
        nxt = []
        for i in range(0, len(sur), 2):
            if i + 1 >= len(sur):
                nxt.append(sur[i]); continue
            a, b = sur[i], sur[i+1]
            xa, xb = xg(a, b)
            ga = int(rng.poisson(xa)); gb = int(rng.poisson(xb))
            if ga == gb:
                ga += int(rng.poisson(xa * .33)); gb += int(rng.poisson(xb * .33))
                if ga == gb:
                    p_a = PENALTY_WIN_RATE.get(a, .5); p_b = PENALTY_WIN_RATE.get(b, .5)
                    w = a if rng.random() < p_a / (p_a + p_b) else b
                else:
                    w = a if ga > gb else b
            else:
                w = a if ga > gb else b
            for team, n_g in ((a, ga), (b, gb)):
                players = SQUADS.get(team, [("Other", 1.0)])
                names   = [p[0] for p in players]
                wts     = np.array([p[1] for p in players], dtype=float); wts /= wts.sum()
                for _ in range(n_g):
                    pg[names[int(rng.choice(len(names), p=wts))]] += 1
            nxt.append(w)
        sur = nxt

    return sur[0], dict(pg)

def find_team(player: str) -> str:
    for t, ps in SQUADS.items():
        if any(nm == player for nm, _ in ps):
            return t
    return "—"

def find_flag(player: str) -> str:
    return TEAMS.get(find_team(player), {}).get("flag", "")

@st.cache_data(show_spinner=False)
def run_sims(n: int, seed: int, thash: str) -> tuple:
    rng = np.random.default_rng(seed)
    wc:  dict = {}
    gbt: dict = defaultdict(list)
    for _ in range(n):
        ch, pg = sim_tournament_fast(rng)
        wc[ch] = wc.get(ch, 0) + 1
        for p, g in pg.items():
            gbt[p].append(g)
    ga  = {p: sum(gs) / n for p, gs in gbt.items() if p != "Other"}
    tot = sum(ga.values())
    return wc, ({p: v / tot for p, v in ga.items()} if tot > 0 else {})

# ══════════════════════════════════════════════════════════════════════
# UI — Header
# ══════════════════════════════════════════════════════════════════════
st.title("🏆 2026 World Cup Simulator")
st.caption("Poisson model · Monte Carlo · Elo ratings · Live API · USA / Mexico / Canada")

top_elo_team = max(BASE_TEAMS, key=lambda k: BASE_TEAMS[k]["elo"])
st.markdown(f"""<div class="dbar">
<div><div class="dl">Teams</div><div class="dv">{len(BASE_TEAMS)}</div></div>
<div><div class="dl">R32 fixtures</div><div class="dv">{len(R32_BRACKET)}</div></div>
<div><div class="dl">Player cards</div><div class="dv">{len(PLAYER_CARDS_FALLBACK)}</div></div>
<div><div class="dl">WC editions</div><div class="dv">22</div></div>
<div><div class="dl">Avg Elo</div><div class="dv">{int(sum(v['elo'] for v in BASE_TEAMS.values())/len(BASE_TEAMS))}</div></div>
<div><div class="dl">Top Elo</div><div class="dv">{BASE_TEAMS[top_elo_team]['flag']} {top_elo_team}</div></div>
<div><div class="dl">Hosts</div><div class="dv">🇺🇸 🇲🇽 🇨🇦</div></div>
</div>""", unsafe_allow_html=True)

# ── Module-scope defaults ─────────────────────────────────────────────────────
ratings_source = "Built-in (June 2026)"
n_matches_used = 0
squads_source  = "Built-in squad data"
api_log:       list = []   # (icon, label, detail)
scorer_debug   = ""
league_debug   = ""
wc_lid         = WC2026_LEAGUE_ID

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    api_key = st.text_input("API-Football key", type="password",
                            help="api-football.com key for live data")
    use_api = st.checkbox("Enable live API", value=False)

    if use_api and api_key:
        with st.spinner("Connecting…"):
            status = fetch_live_data(api_key)
        if not status:
            st.warning("⚠️ Connection failed.")
            api_log.append(("❌","API connection","Returned None — check key"))
        else:
            plan,used,limit = parse_account_info(status)
            api_log.append(("✅","Connected",f"Plan: {plan} · Reqs: {used}/{limit}"))
            st.success(f"✅ {plan} · {used}/{limit} reqs")

            with st.spinner("Ratings…"):
                res_data, rat_debug = fetch_recent_results(api_key)
            if res_data:
                TEAMS.update(compute_ratings(res_data))
                n_matches_used = len(res_data)
                ratings_source = f"Live ({n_matches_used} matches)"
                api_log.append(("✅","Ratings",f"{n_matches_used} matches\n"+"\n".join(rat_debug)))
                st.success(f"✅ {n_matches_used} matches")
            else:
                api_log.append(("⚠️","Ratings","0 matches\n"+"\n".join(rat_debug)))

            with st.spinner("League ID…"):
                wc_lid, league_debug = discover_league_id(api_key)
            api_log.append(("🔍","League ID",f"league_id={wc_lid}"))

            with st.spinner("Scorers…"):
                api_squads, scorer_debug = fetch_top_scorers(api_key, wc_lid)
            if api_squads:
                SQUADS.update(api_squads)
                squads_source = f"Live WC ({len(api_squads)} teams)"
                api_log.append(("✅","Scorers",f"{len(api_squads)} teams · {sum(len(v) for v in api_squads.values())} records"))
                st.success(f"✅ Scorers: {len(api_squads)} teams")
            else:
                api_log.append(("❌","Scorers","0 records — see API Status tab"))
                st.warning("⚠️ No scorer data — see API Status tab")
    else:
        st.info("🔵 Add key for live data")
        api_log.append(("🔵","Offline","Using built-in data"))

    st.divider()
    st.markdown("**🏟️ Host advantage**")
    ha_usa    = st.slider("🇺🇸 USA",    0.0,0.6, DEFAULT_HOST_ADV["USA"],    0.01, key="ha_usa")
    ha_mexico = st.slider("🇲🇽 Mexico", 0.0,0.6, DEFAULT_HOST_ADV["Mexico"], 0.01, key="ha_mex")
    ha_canada = st.slider("🇨🇦 Canada", 0.0,0.6, DEFAULT_HOST_ADV["Canada"], 0.01, key="ha_can")
    HOST_ADVANTAGE.update({"USA":ha_usa,"Mexico":ha_mexico,"Canada":ha_canada})

    st.divider()
    st.markdown("**⚡ Team boost**")
    boost_team = st.selectbox("Team",["(none)"]+sorted(TEAMS.keys()),key="boost_team")
    boost_val  = st.slider("Attack ×",0.5,2.0,1.0,0.05,key="boost_val")
    if boost_team!="(none)":
        TEAM_BOOST[boost_team]=boost_val
        if boost_val!=1.0:
            st.caption(f"{TEAMS.get(boost_team,{}).get('flag','')} {boost_team} ×{boost_val:.2f}")

    st.divider()
    n_boost = sum(1 for v in TEAM_BOOST.values() if v!=1.0)
    n_hadj  = sum(1 for k,v in HOST_ADVANTAGE.items() if abs(v-DEFAULT_HOST_ADV.get(k,0))>0.001)
    for line in [
        f"Ratings: {ratings_source}",
        f"Scorers: {squads_source}",
        f"Boosts active: {n_boost} · Host adj: {n_hadj}",
        f"Squad records: {sum(len(v) for v in SQUADS.values())}",
        f"Pen data: {len(PENALTY_WIN_RATE)} teams",
    ]: st.caption(line)

# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab_sim,tab_gb,tab_cards,tab_h2h,tab_bracket,tab_map,tab_hist,tab_model,tab_api = st.tabs([
    "📊 Simulation","👟 Golden Boot","🃏 Cards",
    "⚔️ H2H","🗓️ Bracket","🗺️ Map",
    "🏆 History","🔬 Model","🔌 API",
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1 — Simulation
# ─────────────────────────────────────────────────────────────────────
with tab_sim:
    st.subheader("Monte Carlo tournament simulation")
    c1,c2 = st.columns(2)
    with c1:
        n_sims = st.slider("Simulations",1_000,5_000_000,100_000,100_000)
    with c2:
        seed = st.number_input("Seed (0=random)",value=0,step=1)

    if st.button("▶ Run",type="primary",use_container_width=True,key="run_sim"):
        s    = int(seed) if int(seed)!=0 else random.randint(1,10_000_000)
        thsh = str(hash(str(sorted((k,v["atk"],v["def"],TEAM_BOOST.get(k,1.0)) for k,v in TEAMS.items()))))
        with st.spinner(f"Simulating {n_sims:,} tournaments…"):
            wc,gb_prob = run_sims(n_sims,s,thsh)

        rows=[{"Flag":get_team(t)["flag"],"Team":t,"Elo":get_team(t)["elo"],
               "Wins":c,"Win%":round(c/n_sims*100,2)} for t,c in wc.items()]
        df=(pd.DataFrame(rows).sort_values("Win%",ascending=False).reset_index(drop=True))
        df.insert(0,"#",df.index+1)

        top=df.iloc[0]
        gap=f" · +{top['Win%']-df.iloc[1]['Win%']:.1f}pp over {df.iloc[1]['Team']}" if len(df)>1 else ""
        st.success(f"{top['Flag']} **{top['Team']}** champion · {top['Win%']:.1f}%{gap}")

        # 5-metric dense row
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Simulations",f"{n_sims:,}")
        m2.metric("Teams w/ title",int((df["Wins"]>0).sum()))
        m3.metric("Top-5 share",f"{df.head(5)['Win%'].sum():.1f}%")
        m4.metric("Favourite odds",f"{100/top['Win%']:.1f}:1" if top['Win%']>0 else "—")
        m5.metric("Spread σ",f"{df['Win%'].std():.2f}")

        cl,cr = st.columns([3,2],gap="medium")
        with cl:
            st.markdown("#### Full field — championship %")
            st.dataframe(
                df[["#","Flag","Team","Elo","Wins","Win%"]]
                .style.format({"Win%":"{:.2f}%","Elo":"{:,}","Wins":"{:,}"}),
                use_container_width=True,hide_index=True,height=430)
        with cr:
            st.markdown("#### Top 12")
            st.bar_chart(df.head(12).set_index("Team")[["Win%"]],use_container_width=True)
            st.markdown("#### By confederation")
            cm={"France":"UEFA","Spain":"UEFA","England":"UEFA","Germany":"UEFA","Portugal":"UEFA",
                "Netherlands":"UEFA","Belgium":"UEFA","Switzerland":"UEFA","Austria":"UEFA",
                "Norway":"UEFA","Sweden":"UEFA","Croatia":"UEFA","Bosnia":"UEFA",
                "Argentina":"CONMEBOL","Brazil":"CONMEBOL","Colombia":"CONMEBOL",
                "Ecuador":"CONMEBOL","Paraguay":"CONMEBOL",
                "Morocco":"CAF","Senegal":"CAF","Egypt":"CAF","Ghana":"CAF",
                "Ivory Coast":"CAF","South Africa":"CAF","Cape Verde":"CAF",
                "USA":"CONCACAF","Mexico":"CONCACAF","Canada":"CONCACAF",
                "Japan":"AFC","Iran":"AFC","Uzbekistan":"AFC","Australia":"AFC"}
            df["C"]=df["Team"].map(cm).fillna("—")
            st.bar_chart(df.groupby("C")["Win%"].sum().sort_values(ascending=False),use_container_width=True)

        st.session_state.update({"gb_prob":gb_prob,"n_sims":n_sims,"last_wc":wc})
        st.caption("👟 Golden Boot tab · 🗺️ Map tab updates with win %")

# ─────────────────────────────────────────────────────────────────────
# TAB 2 — Golden Boot
# ─────────────────────────────────────────────────────────────────────
with tab_gb:
    st.subheader("👟 Golden Boot — predicted top scorers")
    if "gb_prob" not in st.session_state:
        st.info("Run simulation first (📊 tab).")
    else:
        gbp  = st.session_state["gb_prob"]
        nran = st.session_state.get("n_sims", 100_000)
        rows = [{"Player":p,"Team":find_team(p),"Flag":find_flag(p),"GB%":round(v*100,2)}
                for p,v in sorted(gbp.items(),key=lambda x:-x[1]) if p!="Other"]
        gdf  = pd.DataFrame(rows).head(20).reset_index(drop=True)
        gdf.insert(0,"#",gdf.index+1)
        if gdf.empty:
            st.warning("No player data.")
        else:
            t=gdf.iloc[0]
            st.success(f"{t['Flag']} **{t['Player']}** ({t['Team']}) — Golden Boot {t['GB%']:.1f}%")
            cl,cr = st.columns([3,2],gap="large")
            with cl:
                st.markdown("#### Top 20 contenders")
                st.dataframe(gdf[["#","Flag","Player","Team","GB%"]]
                    .style.format({"GB%":"{:.2f}%"}),
                    use_container_width=True,hide_index=True)
            with cr:
                st.markdown("#### Top 10")
                st.bar_chart(gdf.head(10).set_index("Player")[["GB%"]],use_container_width=True)
            st.caption(f"{nran:,} sims · proportional share of avg tournament goals · 'Other' excluded")

# ─────────────────────────────────────────────────────────────────────
# TAB 3 — Player Cards
# ─────────────────────────────────────────────────────────────────────
with tab_cards:
    st.subheader("🃏 EA FC 25 Player Cards — WC 2026")
    st.caption("June 2026 ratings · Gold 90+ · Gold 86–89 · Silver 82–85 · Bronze <82")

    raw: list = []
    if use_api and api_key:
        with st.spinner("Fetching from API…"):
            raw = fetch_player_cards_api(api_key)
        if raw: st.success(f"✅ {len(raw)} players from API")
        else:   st.info("ℹ️ Using built-in ratings.")
    if not raw: raw = list(PLAYER_CARDS_FALLBACK)

    cards = [{"name":r[0],"team":r[1],"pos":r[2],"overall":r[3],
              "pace":r[4],"shooting":r[5],"passing":r[6],
              "dribbling":r[7],"defending":r[8],"physical":r[9]}
             for r in raw]
    cards.sort(key=lambda x:-x["overall"])

    f1,f2,f3,f4 = st.columns([3,2,2,2])
    with f1: ft = st.selectbox("Team",["All"]+sorted({c["team"] for c in cards}),key="ct")
    with f2: fp = st.selectbox("Pos", ["All"]+sorted({c["pos"] for c in cards}),key="cp")
    with f3: fo = st.slider("Min OVR",60,94,60,1,key="co")
    with f4: fc = st.selectbox("Sort",["Overall ↓","Overall ↑","Name"],key="cs")

    fil = [c for c in cards if (ft=="All" or c["team"]==ft)
                              and (fp=="All" or c["pos"]==fp)
                              and c["overall"]>=fo]
    if fc=="Overall ↑": fil.sort(key=lambda x:x["overall"])
    elif fc=="Name":    fil.sort(key=lambda x:x["name"])

    st.caption(f"**{len(fil)}** players · {len([c for c in fil if c['overall']>=90])} TOTY gold · "
               f"{len([c for c in fil if 86<=c['overall']<90])} gold · "
               f"{len([c for c in fil if 82<=c['overall']<86])} silver · "
               f"{len([c for c in fil if c['overall']<82])} bronze")

    CPR=6
    for i in range(0,len(fil),CPR):
        cols=st.columns(CPR)
        for col,card in zip(cols,fil[i:i+CPR]):
            with col:
                st.markdown(render_card(card),unsafe_allow_html=True)
                st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
    if not fil: st.info("No players match filters.")

# ─────────────────────────────────────────────────────────────────────
# TAB 4 — Head-to-head
# ─────────────────────────────────────────────────────────────────────
with tab_h2h:
    st.subheader("⚔️ Head-to-head")
    st.caption("20,000-match Poisson simulation · historical penalty rates · xG model")
    tns = sorted(TEAMS.keys())
    ca,cvs,cb = st.columns([5,1,5])
    with ca: ta = st.selectbox("Team A",tns,index=tns.index("France"),key="h2h_a")
    with cvs: st.markdown("<br><br>**vs**",unsafe_allow_html=True)
    with cb: tb = st.selectbox("Team B",tns,index=tns.index("Argentina"),key="h2h_b")

    if st.button("▶ Calculate",type="primary",use_container_width=True,key="h2h_go"):
        if ta==tb:
            st.warning("Pick different teams.")
        else:
            # h2h is now closed-form — instant, no spinner needed
            pr = h2h(ta, tb)
            ml, top5 = predict_score(ta, tb)
            xa, xb   = xg(ta, tb)
            fa=get_team(ta)["flag"]; fb=get_team(tb)["flag"]
            pa=PENALTY_WIN_RATE.get(ta,.5); pb=PENALTY_WIN_RATE.get(tb,.5); pt=pa+pb
            ea=get_team(ta)["elo"]; eb=get_team(tb)["elo"]

            # Dense 7-metric row
            c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
            c1.metric(f"{fa} Win 90'",f"{pr['team_a']:.1f}%")
            c2.metric("Draw 90'",f"{pr['draw_90']:.1f}%")
            c3.metric(f"{fb} Win 90'",f"{pr['team_b']:.1f}%")
            c4.metric(f"{fa} xG",f"{xa:.2f}")
            c5.metric(f"{fb} xG",f"{xb:.2f}")
            c6.metric(f"{fa} Pen",f"{pa/pt*100:.0f}%")
            c7.metric(f"{fb} Pen",f"{pb/pt*100:.0f}%")

            st.divider()
            col_sc, col_elo = st.columns([3, 2], gap="medium")
            with col_sc:
                st.markdown(f"#### Predicted score: **{fa} {ml[0]}–{ml[1]} {fb}** ({ml[2]:.1f}% chance)")
                st.caption("Top 5 most likely scorelines (90 min)")
                sc_rows = []
                for i_g, j_g, pct in top5:
                    outcome = "Win" if i_g > j_g else ("Loss" if j_g > i_g else "Draw")
                    sc_rows.append({"Scoreline": f"{ta} {i_g}–{j_g} {tb}",
                                    "Prob": f"{pct:.2f}%", "Result": outcome})
                st.dataframe(pd.DataFrame(sc_rows), use_container_width=True, hide_index=True)
            with col_elo:
                d1,d2 = st.columns(2)
                d1.metric(f"{fa} Elo",f"{ea:,}",f"{ea-eb:+,}")
                d2.metric(f"{fb} Elo",f"{eb:,}")
                st.metric("xG ratio",f"{xa/xb:.2f}:1" if xb>0 else "—")
                st.metric("Model margin",f"{pr['team_a']-pr['team_b']:+.1f}pp")

            notes=[]
            if HOST_ADVANTAGE.get(ta,0): notes.append(f"{fa} home +{HOST_ADVANTAGE[ta]*100:.0f}%")
            if HOST_ADVANTAGE.get(tb,0): notes.append(f"{fb} home +{HOST_ADVANTAGE[tb]*100:.0f}%")
            if TEAM_BOOST.get(ta,1)!=1: notes.append(f"{fa} boost ×{TEAM_BOOST[ta]:.2f}")
            if TEAM_BOOST.get(tb,1)!=1: notes.append(f"{fb} boost ×{TEAM_BOOST[tb]:.2f}")
            if notes: st.caption(" · ".join(notes))
            st.caption("Closed-form Poisson · Win/draw = 90 min · Knockout draws → ET → penalties")

# ─────────────────────────────────────────────────────────────────────
# TAB 5 — Bracket
# ─────────────────────────────────────────────────────────────────────
with tab_bracket:
    st.subheader("🗓️ Round of 32 — Live Bracket")

    live_ms:  list = []
    res_debug = ""
    if use_api and api_key:
        with st.spinner("Fetching results…"):
            live_ms, res_debug = fetch_wc_results(api_key, wc_lid)
        fin_s = {"FT","AET","PEN","AWD","WO"}
        live_s= {"1H","2H","HT","ET","BT","P","INT"}
        n_done = sum(1 for m in live_ms if m["status"] in fin_s)
        n_live = sum(1 for m in live_ms if m["status"] in live_s)
        hc = st.columns(4)
        hc[0].metric("Played",n_done)
        hc[1].metric("Live",n_live)
        hc[2].metric("Remaining",max(0,16-n_done))
        hc[3].caption(f"Auto-refresh every 5 min · {len(live_ms)} total fixtures")
        if st.button("🔄 Refresh now",key="ref_br"):
            fetch_wc_results.clear(); st.rerun()
    else:
        st.info("🔵 Connect API for live scores. Showing probabilities.")

    st.divider()
    use_live  = True
    bl        = build_live_bracket(live_ms)
    fin_set   = {"FT","AET","PEN","AWD","WO"}
    live_set  = {"1H","2H","HT","ET","BT","P","INT"}

    # Compute all probabilities in one cached batch call (runs in milliseconds)
    thash_br = str(hash(str(sorted((k,v["atk"],v["def"],TEAM_BOOST.get(k,1.0)) for k,v in TEAMS.items()))))
    all_probs = bracket_probs_cached(thash_br)

    bd = []
    for i,(home,away,date) in enumerate(R32_BRACKET):
        le     = bl[i]
        status = le["status"]
        winner = le["winner"]
        gh     = le["goals_home"]
        ga_v   = le["goals_away"]
        pr     = all_probs.get((home, away)) if status in ("NS", None) else None
        xa, xb = (xg(home, away) if status in ("NS", None) and away not in ("TBD",) else (None, None))
        ph=PENALTY_WIN_RATE.get(home,.5); pa=PENALTY_WIN_RATE.get(away,.5); pt=ph+pa
        bd.append({"home":home,"away":away,"date":date,
                   "flag_h":BASE_TEAMS.get(home,{}).get("flag","🏳️"),
                   "flag_a":BASE_TEAMS.get(away,{}).get("flag","🏳️"),
                   "status":status,"winner":winner,"gh":gh,"ga":ga_v,
                   "pr":pr,"xa":xa,"xb":xb,"ph":ph/pt*100,"pa":pa/pt*100})

    def bcard(m: dict) -> str:
        fin = m["status"] in fin_set; lv = m["status"] in live_set
        if fin:
            sb='<span style="background:#22c55e;color:#000;font-size:8px;font-weight:800;padding:1px 5px;border-radius:4px;">FT</span>'
        elif lv:
            sb='<span style="background:#ef4444;color:#fff;font-size:8px;font-weight:800;padding:1px 5px;border-radius:4px;">● LIVE</span>'
        elif m["status"]=="TBD":
            sb='<span style="color:#444;font-size:9px;">TBD</span>'
        else:
            sb=f'<span style="color:#555;font-size:9px;">{m["date"]}</span>'

        if fin and m["winner"]:
            ch="#22c55e"; ca="#f87171"
            col_h=ch if m["winner"]==m["home"] else ca
            col_a=ch if m["winner"]==m["away"] else ca
            bg_h="rgba(34,197,94,0.07)" if m["winner"]==m["home"] else "transparent"
            bg_a="rgba(34,197,94,0.07)" if m["winner"]==m["away"] else "transparent"
        elif lv:
            col_h=col_a="#facc15"; bg_h=bg_a="transparent"
        elif m["pr"] and m["away"]!="TBD":
            close=abs(m["pr"]["team_a"]-m["pr"]["team_b"])<5
            col_h="#facc15" if close else ("#4ade80" if m["pr"]["team_a"]>m["pr"]["team_b"] else "#f87171")
            col_a="#facc15" if close else ("#4ade80" if m["pr"]["team_b"]>m["pr"]["team_a"] else "#f87171")
            bg_h=bg_a="transparent"
        else:
            col_h=col_a="#555"; bg_h=bg_a="transparent"

        if fin or lv:
            sh=str(m["gh"]) if m["gh"] is not None else "?"
            sa=str(m["ga"]) if m["ga"] is not None else "?"
            rh=f'<span style="font-size:19px;font-weight:900;color:{col_h};">{sh}</span>'
            ra=f'<span style="font-size:19px;font-weight:900;color:{col_a};">{sa}</span>'
            bar=""
            if m["status"]=="AET": bar='<div style="text-align:center;font-size:8px;color:#555;margin:1px 0;">AET</div>'
            elif m["status"]=="PEN": bar='<div style="text-align:center;font-size:8px;color:#555;margin:1px 0;">penalties</div>'
        elif m["away"]=="TBD":
            rh=ra=bar=""
        elif m["pr"]:
            rh=f'<span style="font-size:11px;font-weight:800;color:{col_h};">{m["pr"]["team_a"]:.1f}%</span>'
            ra=f'<span style="font-size:11px;font-weight:800;color:{col_a};">{m["pr"]["team_b"]:.1f}%</span>'
            bh=int(m["pr"]["team_a"]); bd2=int(m["pr"]["draw_90"]); ba=int(m["pr"]["team_b"])
            bar=(f'<div style="display:flex;height:3px;border-radius:2px;overflow:hidden;margin:2px 0;gap:1px;">'
                 f'<div style="width:{bh}%;background:{col_h};"></div>'
                 f'<div style="width:{bd2}%;background:#333;"></div>'
                 f'<div style="width:{ba}%;background:{col_a};"></div></div>')
        else:
            rh=ra=bar=""

        ch_h=" 👑" if fin and m["winner"]==m["home"] else ""
        ch_a=" 👑" if fin and m["winner"]==m["away"] else ""
        xg_line=""
        if not fin and not lv and m["xa"] is not None and m["pr"] is not None:
            # Predicted score from pr dict (stored by bracket_probs_cached)
            score_data = m["pr"].get("score")
            pred_str = ""
            if score_data:
                ml, _ = score_data
                pred_str = f' · Pred <b>{ml[0]}–{ml[1]}</b> ({ml[2]:.1f}%)'
            xg_line=(f'<div style="font-size:9px;color:#444;margin-top:1px;">'
                     f'xG {m["flag_h"]}{m["xa"]:.2f} {m["flag_a"]}{m["xb"]:.2f}'
                     f' · Pen {m["ph"]:.0f}%/{m["pa"]:.0f}%{pred_str}</div>')
        return (f'<div style="border:1px solid #1e1e1e;border-radius:5px;padding:7px 9px;background:#161616;margin-bottom:5px;">'
            f'<div style="margin-bottom:4px;">{sb}</div>'
            f'<div style="display:flex;align-items:center;gap:5px;padding:2px 3px;background:{bg_h};border-radius:3px;margin-bottom:1px;">'
            f'<span style="font-size:13px;">{m["flag_h"]}</span>'
            f'<span style="flex:1;font-size:12px;font-weight:600;color:#ccc;">{m["home"]}{ch_h}</span>{rh}</div>'
            +bar+
            f'<div style="display:flex;align-items:center;gap:5px;padding:2px 3px;background:{bg_a};border-radius:3px;margin-top:1px;">'
            f'<span style="font-size:13px;">{m["flag_a"]}</span>'
            f'<span style="flex:1;font-size:12px;font-weight:600;color:#ccc;">{m["away"]}{ch_a}</span>{ra}</div>'
            +xg_line+'</div>')

    cl,cg,cr = st.columns([5,1,5])
    with cl:
        st.markdown("**Matches 1–8**")
        st.markdown("".join(bcard(m) for m in bd[:8]),unsafe_allow_html=True)
    with cg:
        st.markdown('<div style="border-left:1px dashed #1e1e1e;height:100%;margin:0 auto;width:0;"></div>',unsafe_allow_html=True)
    with cr:
        st.markdown("**Matches 9–16**")
        st.markdown("".join(bcard(m) for m in bd[8:]),unsafe_allow_html=True)
    if res_debug:
        with st.expander("🔍 Debug"):
            st.code(res_debug)
    st.caption("🟢 winner · 🔴 eliminated · % = win prob (unplayed) · 👑 advanced · All R32 results hardcoded from real tournament")

    # ── Round of 16 ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Round of 16")
    st.caption("Jul 4–7 · Matches updating live via API when connected")

    r16_cards_html = []
    for home, away, date, gh, ga, status, winner in R16_FIXTURES:
        fh = BASE_TEAMS.get(home, {}).get("flag", "🏳️")
        fa = BASE_TEAMS.get(away, {}).get("flag", "🏳️")
        fin = status in {"FT","AET","PEN"}
        lv  = status in {"1H","2H","HT","ET","BT","P","INT"}

        # Also check API for this match
        for m in live_ms:
            k1 = (m["home"].lower(), m["away"].lower())
            k2 = (home.lower(), away.lower())
            if k1 == k2 or (m["home"].lower()==away.lower() and m["away"].lower()==home.lower()):
                if m["status"] in {"FT","AET","PEN","AWD","WO","1H","2H","HT","ET","BT","P","INT"}:
                    gh = m["goals_home"]; ga = m["goals_away"]
                    status = m["status"]; winner = m["winner"]
                    fin = status in {"FT","AET","PEN","AWD","WO"}
                    lv  = not fin and status != "NS"
                    break

        if fin:
            sb = '<span style="background:#22c55e;color:#000;font-size:8px;font-weight:800;padding:1px 5px;border-radius:4px;">FT</span>'
            col_h = "#22c55e" if winner == home else "#f87171"
            col_a = "#22c55e" if winner == away else "#f87171"
            bg_h  = "rgba(34,197,94,0.07)" if winner == home else "transparent"
            bg_a  = "rgba(34,197,94,0.07)" if winner == away else "transparent"
            rh = f'<span style="font-size:19px;font-weight:900;color:{col_h};">{gh if gh is not None else "?"}</span>'
            ra = f'<span style="font-size:19px;font-weight:900;color:{col_a};">{ga if ga is not None else "?"}</span>'
            ch_h = " 👑" if winner == home else ""
            ch_a = " 👑" if winner == away else ""
        elif lv:
            sb = '<span style="background:#ef4444;color:#fff;font-size:8px;font-weight:800;padding:1px 5px;border-radius:4px;">● LIVE</span>'
            col_h = col_a = "#facc15"; bg_h = bg_a = "transparent"
            rh = f'<span style="font-size:19px;font-weight:900;color:#facc15;">{gh if gh is not None else "0"}</span>'
            ra = f'<span style="font-size:19px;font-weight:900;color:#facc15;">{ga if ga is not None else "0"}</span>'
            ch_h = ch_a = ""
        else:
            sb = f'<span style="color:#555;font-size:9px;">{date}</span>'
            pr2 = all_probs.get((home, away)) or h2h(home, away)
            col_h = "#4ade80" if pr2["team_a"] > pr2["team_b"] else "#f87171"
            col_a = "#4ade80" if pr2["team_b"] > pr2["team_a"] else "#f87171"
            if abs(pr2["team_a"]-pr2["team_b"]) < 5: col_h = col_a = "#facc15"
            bg_h = bg_a = "transparent"
            # Predicted score
            sc_data = pr2.get("score")
            pred = f" · pred {sc_data[0][0]}–{sc_data[0][1]} ({sc_data[0][2]:.1f}%)" if sc_data else ""
            rh = f'<span style="font-size:11px;font-weight:800;color:{col_h};">{pr2["team_a"]:.1f}%</span>'
            ra = f'<span style="font-size:11px;font-weight:800;color:{col_a};">{pr2["team_b"]:.1f}%</span>'
            ch_h = ch_a = ""

        r16_cards_html.append(
            f'<div style="border:1px solid #1e1e1e;border-radius:5px;padding:7px 9px;background:#161616;margin-bottom:5px;">'
            f'<div style="margin-bottom:4px;">{sb}</div>'
            f'<div style="display:flex;align-items:center;gap:5px;padding:2px 3px;background:{bg_h};border-radius:3px;margin-bottom:1px;">'
            f'<span style="font-size:13px;">{fh}</span>'
            f'<span style="flex:1;font-size:12px;font-weight:600;color:#ccc;">{home}{ch_h}</span>{rh}</div>'
            f'<div style="display:flex;align-items:center;gap:5px;padding:2px 3px;background:{bg_a};border-radius:3px;">'
            f'<span style="font-size:13px;">{fa}</span>'
            f'<span style="flex:1;font-size:12px;font-weight:600;color:#ccc;">{away}{ch_a}</span>{ra}</div>'
            f'</div>'
        )

    # Display R16 in 2 rows of 4
    r16_c1, r16_gap, r16_c2 = st.columns([5,1,5])
    with r16_c1:
        st.markdown("**Matches 1–4**")
        st.markdown("".join(r16_cards_html[:4]), unsafe_allow_html=True)
    with r16_gap:
        st.markdown('<div style="border-left:1px dashed #1e1e1e;height:100%;margin:0 auto;width:0;"></div>', unsafe_allow_html=True)
    with r16_c2:
        st.markdown("**Matches 5–8**")
        st.markdown("".join(r16_cards_html[4:]), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 6 — Map
# ─────────────────────────────────────────────────────────────────────
with tab_map:
    st.subheader("🗺️ Team Map — 2026 World Cup")
    st.caption("All 32 nations · dot size = Elo · click for full stats")

    COORDS = {
        "Argentina":(-34.6,-58.4),"Australia":(-33.9,151.2),"Austria":(48.2,16.4),
        "Belgium":(50.8,4.4),"Bosnia":(43.8,18.4),"Brazil":(-15.8,-47.9),
        "Canada":(45.4,-75.7),"Cape Verde":(14.9,-23.5),"Colombia":(4.7,-74.1),
        "Croatia":(45.8,15.9),"Ecuador":(-0.2,-78.5),"Egypt":(30.1,31.2),
        "England":(51.5,-0.1),"France":(48.9,2.3),"Germany":(52.5,13.4),
        "Ghana":(5.6,-0.2),"Iran":(35.7,51.4),"Ivory Coast":(5.3,-4.0),
        "Japan":(35.7,139.7),"Mexico":(19.4,-99.1),"Morocco":(34.0,-6.8),
        "Netherlands":(52.4,4.9),"Norway":(59.9,10.7),"Paraguay":(-25.3,-57.6),
        "Portugal":(38.7,-9.1),"Senegal":(14.7,-17.4),"South Africa":(-25.7,28.2),
        "Spain":(40.4,-3.7),"Sweden":(59.3,18.1),"Switzerland":(46.9,7.4),
        "USA":(38.9,-77.0),"Uzbekistan":(41.3,69.2),
    }
    win_data={}
    if "last_wc" in st.session_state:
        wc=st.session_state["last_wc"]; n=st.session_state.get("n_sims",1)
        win_data={t:round(c/n*100,1) for t,c in wc.items()}

    mrows=[]
    for team,(lat,lon) in COORDS.items():
        td=BASE_TEAMS.get(team,{}); flag=td.get("flag","🏳️")
        top2=[p for p,_ in FALLBACK_SQUADS.get(team,[])[:2] if p!="Other"]
        mrows.append({"team":team,"flag":flag,"lat":lat,"lon":lon,
                      "elo":td.get("elo",0),"atk":td.get("atk",0),"def":td.get("def",0),
                      "win_pct":win_data.get(team),"pen_r":PENALTY_WIN_RATE.get(team,.5),
                      "top_players":", ".join(top2)})
    dfm=pd.DataFrame(mrows)

    map_html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body{margin:0;background:#111;font-family:system-ui,sans-serif;}
#map{width:100%;height:540px;}
.leaflet-container{background:#111 !important;}
.panel{position:absolute;top:10px;right:10px;z-index:1000;background:#161616;
       border:1px solid #1e1e1e;border-radius:5px;padding:12px;min-width:195px;
       max-width:230px;color:#ccc;display:none;font-size:12px;}
.panel.vis{display:block;}
.pf{font-size:28px;text-align:center;}
.pn{font-size:14px;font-weight:700;text-align:center;color:#ddd;margin:3px 0 7px;}
.sr{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e1e1e;}
.sl{color:#555;}.sv{color:#ccc;font-weight:600;}
</style>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head><body>
<div id="map"></div><div class="panel" id="panel"></div>
<script>
const T=""" + dfm.to_json(orient="records") + """;
const map=L.map('map',{center:[20,10],zoom:2,minZoom:1,maxZoom:6});
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  {attribution:'©OSM ©CartoDB',subdomains:'abcd',maxZoom:19}).addTo(map);
const maxE=Math.max(...T.map(t=>t.elo)),minE=Math.min(...T.map(t=>t.elo));
T.forEach(t=>{
  const n=(t.elo-minE)/(maxE-minE),r=7+n*13;
  const col=n>0.7?'#dddddd':n>0.4?'#888888':'#444444';
  const c=L.circleMarker([t.lat,t.lon],{radius:r,fillColor:col,color:'#0d0d0d',weight:1.5,fillOpacity:0.9}).addTo(map);
  c.bindTooltip(t.flag+' '+t.team,{permanent:false,direction:'top',offset:[0,-r]});
  c.on('click',()=>{
    const p=document.getElementById('panel');
    const w=t.win_pct!==null?`<div class="sr"><span class="sl">Win %</span><span class="sv">${t.win_pct.toFixed(1)}%</span></div>`:'';
    p.innerHTML=`<div class="pf">${t.flag}</div><div class="pn">${t.team}</div>
      <div class="sr"><span class="sl">Elo</span><span class="sv">${t.elo.toLocaleString()}</span></div>
      <div class="sr"><span class="sl">Attack λ</span><span class="sv">${t.atk.toFixed(2)}</span></div>
      <div class="sr"><span class="sl">Defense ×</span><span class="sv">${t.def.toFixed(2)}</span></div>
      <div class="sr"><span class="sl">Pen rate</span><span class="sv">${(t.pen_r*100).toFixed(0)}%</span></div>
      <div class="sr"><span class="sl">Stars</span><span class="sv" style="font-size:10px;text-align:right;max-width:120px">${t.top_players||'—'}</span></div>
      ${w}`;
    p.classList.add('vis');
  });
});
map.on('click',()=>document.getElementById('panel').classList.remove('vis'));
</script></body></html>"""

    st.components.v1.html(map_html, height=560, scrolling=False)
    st.divider()
    st.markdown("#### All 32 teams — ranked by Elo")
    tbl=[]
    for _,row in dfm.sort_values("elo",ascending=False).iterrows():
        tbl.append({"":row["flag"],"Team":row["team"],"Elo":int(row["elo"]),
                    "Atk":round(row["atk"],2),"Def":round(row["def"],2),
                    "Pen%":f"{row['pen_r']*100:.0f}%",
                    "Win%":f"{row['win_pct']:.1f}%" if row["win_pct"] is not None else "—",
                    "Stars":row["top_players"]})
    tdf=pd.DataFrame(tbl).reset_index(drop=True)
    tdf.insert(0,"#",range(1,len(tdf)+1))
    st.dataframe(tdf,use_container_width=True,hide_index=True)
    st.caption("Win% updates after running simulation (📊 tab)")

# ─────────────────────────────────────────────────────────────────────
# TAB 7 — History
# ─────────────────────────────────────────────────────────────────────
with tab_hist:
    st.subheader("🏆 FIFA World Cup — Complete History 1930–2022")

    wc_cnt = Counter(r[3] for r in WC_HISTORY)
    top5   = wc_cnt.most_common(5)
    mc = st.columns(5)
    for col,(nation,wins) in zip(mc,top5):
        flag=next((r[4] for r in WC_HISTORY if r[3]==nation),"")
        col.metric(f"{flag} {nation}",f"{wins} titles")

    # Dense extra stats row
    st.divider()
    s1,s2,s3,s4,s5 = st.columns(5)
    s1.metric("Total editions","22")
    s2.metric("Most goals in final","10 (1954 3–2 still record)")
    s3.metric("All-time top scorer","Just Fontaine 13 (1958)")
    s4.metric("Highest attendance",f"{max(r[10] for r in WC_HISTORY):,}")
    s5.metric("Host wins",sum(1 for r in WC_HISTORY if r[3]==r[1]))

    st.divider()
    fc1,fc2=st.columns([3,2])
    with fc1:
        filt=st.selectbox("Filter by team",["All"]+sorted({r[3] for r in WC_HISTORY}|{r[5] for r in WC_HISTORY}),key="hf")
    with fc2:
        vm=st.radio("View",["Table","Timeline"],horizontal=True,key="hv")

    rows_f=[r for r in WC_HISTORY if filt=="All" or r[3]==filt or r[5]==filt]

    if vm=="Table":
        hdf=pd.DataFrame(rows_f,columns=["Year","Host","HF","Winner","WF","Runner-up","RF","Score","Top scorer","Goals","Attendance"])
        ddf=hdf[["Year","HF","Host","WF","Winner","RF","Runner-up","Score","Top scorer","Goals","Attendance"]].copy()
        st.dataframe(ddf.style.format({"Attendance":"{:,}"}),use_container_width=True,hide_index=True)
    else:
        tl='<div style="font-family:inherit;">'
        for r in reversed(rows_f):
            yr,host,hf,winner,wf,runup,rf,score,scorer,goals,att=r
            hw=winner==host
            tl+=(f'<div style="display:flex;gap:10px;margin-bottom:8px;padding:9px;'
                 f'background:#161616;border:1px solid #1e1e1e;border-radius:5px;">'
                 f'<div style="min-width:40px;text-align:center;">'
                 f'<div style="font-size:14px;font-weight:800;color:#ccc;">{yr}</div>'
                 f'<div style="font-size:10px;color:#444;">{hf}</div></div>'
                 f'<div style="flex:1;">'
                 f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:2px;">'
                 f'<span style="font-size:16px;">{wf}</span>'
                 f'<span style="font-size:13px;font-weight:700;color:#ddd;">{winner}</span>'
                 f'{"<span style=background:#222;color:#888;font-size:9px;padding:1px 5px;border-radius:3px;>🏠 host</span>" if hw else ""}'
                 f'</div>'
                 f'<div style="font-size:11px;color:#555;margin-bottom:1px;">'
                 f'vs {rf} {runup} · <span style="color:#888;">{score}</span></div>'
                 f'<div style="font-size:10px;color:#444;">⚽ {scorer} ({goals} goals) · 🏟 {att:,} · {host}</div>'
                 f'</div></div>')
        tl+='</div>'
        st.markdown(tl,unsafe_allow_html=True)

    st.divider()
    st.markdown("#### All-time titles")
    wcc=pd.DataFrame(wc_cnt.most_common(),columns=["Country","Titles"])
    st.bar_chart(wcc.set_index("Country"),use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 8 — Model
# ─────────────────────────────────────────────────────────────────────
with tab_model:
    if use_api and api_key and n_matches_used>0:
        st.success(f"🟢 Live ratings — {n_matches_used} matches")
    else:
        st.info("🔵 Built-in ratings (June 2026)")

    with st.expander("Poisson goal model",expanded=True):
        st.markdown("""
```
xG_A = attack_A × defense_B × (1 + home_boost_A) × manual_boost_A
xG_B = attack_B × defense_A × (1 + home_boost_B) × manual_boost_B
```
Goals ~ Poisson(xG). Draw → 30-min ET (33% xG) → historical penalty rates.
""")
    with st.expander("Team ratings"):
        edf=pd.DataFrame([{"Flag":v["flag"],"Team":k,"Atk":v["atk"],"Def":v["def"],
                            "Boost":TEAM_BOOST.get(k,1.0),"Elo":v["elo"]}
                          for k,v in sorted(TEAMS.items(),key=lambda x:-x[1]["elo"])])
        st.dataframe(edf.style.format({"Atk":"{:.3f}","Def":"{:.3f}","Boost":"{:.2f}×","Elo":"{:,}"}),
                     use_container_width=True,hide_index=True)
    with st.expander("Penalty shootout rates"):
        pdf=pd.DataFrame([{"Flag":get_team(t)["flag"],"Team":t,"Win rate":f"{r*100:.0f}%"}
                          for t,r in sorted(PENALTY_WIN_RATE.items(),key=lambda x:-x[1]) if t in TEAMS])
        st.dataframe(pdf,use_container_width=True,hide_index=True)
        st.caption("WC + major continental tournaments through 2024")
    with st.expander("vs professional models"):
        st.markdown("""
|Feature|This|Pro|
|---|---|---|
|Goal dist|Poisson ✅|Poisson/neg-binomial|
|Ratings|Live API + recency ✅|MLE on 10k+ matches|
|Home adv|Per-host adjustable ✅|Per-stadium + crowd|
|Manual boost|Any team ✅|Automated|
|Injuries|Not included|Live feeds|
|Penalties|Historical rates ✅|Player-level records|
|Golden Boot|Goal-share model ✅|Shot-level xG|
""")

# ─────────────────────────────────────────────────────────────────────
# TAB 9 — API Status
# ─────────────────────────────────────────────────────────────────────
with tab_api:
    st.subheader("🔌 API Status")
    if not use_api or not api_key:
        st.info("Enable API and add key in sidebar.")
    else:
        ok_all = all(i=="✅" for i,_,_ in api_log)
        ok_any = any(i=="✅" for i,_,_ in api_log)
        if ok_all:   st.success("✅ All checks passed")
        elif ok_any: st.warning("⚠️ Partially working")
        else:        st.error("❌ No live data")

        st.divider()
        for icon,label,detail in api_log:
            ci,cb=st.columns([1,10])
            with ci: st.markdown(f"## {icon}")
            with cb:
                st.markdown(f"**{label}**")
                lines=detail.strip().split("\n")
                st.caption(lines[0])
                if len(lines)>1:
                    with st.expander("Detail"): st.code("\n".join(lines[1:]))

        st.divider()
        with st.expander("League ID discovery log"):
            st.code(league_debug or "Not run")
        with st.expander("Scorer fetch log"):
            st.code(scorer_debug or "Not run")

        st.divider()
        st.markdown("""**If scorers return 0:**
1. Check league ID discovery log — if `league_id=1` wasn't confirmed, your plan may not include WC 2026
2. Free plans exclude player stats — need at least **Starter** plan
3. Check requests: `X/Y` shown in connection step — if at limit all calls return empty
4. Early in tournament `/topscorers` may be empty — Strategy 3 (fixture events) covers this once matches finish
""")
        if st.button("🔄 Clear cache + retry",key="clr"):
            for fn in [fetch_live_data,fetch_recent_results,discover_league_id,
                       fetch_top_scorers,fetch_wc_results,fetch_player_cards_api]:
                fn.clear()
            st.success("Cleared — reload page to re-fetch.")
            st.rerun()
