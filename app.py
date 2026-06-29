import streamlit as st
import numpy as np
import pandas as pd
import random
import requests
from datetime import datetime, timedelta
from collections import defaultdict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="2026 World Cup Simulator", page_icon="🏆", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# DATA — Updated June 2026
# Elo ratings from eloratings.net · atk/def calibrated from 2024-2026 results
# ══════════════════════════════════════════════════════════════════════════════
BASE_TEAMS = {
    # Team: flag, attack λ, defense multiplier, Elo
    "Argentina":    {"flag":"🇦🇷","atk":2.10,"def":0.74,"elo":2090},
    "France":       {"flag":"🇫🇷","atk":2.00,"def":0.73,"elo":2055},
    "Spain":        {"flag":"🇪🇸","atk":1.98,"def":0.68,"elo":2045},
    "England":      {"flag":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","atk":1.82,"def":0.74,"elo":2010},
    "Brazil":       {"flag":"🇧🇷","atk":1.85,"def":0.76,"elo":1995},
    "Germany":      {"flag":"🇩🇪","atk":1.88,"def":0.77,"elo":1985},
    "Portugal":     {"flag":"🇵🇹","atk":1.80,"def":0.80,"elo":1970},
    "Netherlands":  {"flag":"🇳🇱","atk":1.76,"def":0.78,"elo":1960},
    "Morocco":      {"flag":"🇲🇦","atk":1.45,"def":0.65,"elo":1920},
    "Mexico":       {"flag":"🇲🇽","atk":1.60,"def":0.82,"elo":1915},
    "Switzerland":  {"flag":"🇨🇭","atk":1.62,"def":0.77,"elo":1912},
    "USA":          {"flag":"🇺🇸","atk":1.58,"def":0.84,"elo":1900},
    "Belgium":      {"flag":"🇧🇪","atk":1.68,"def":0.84,"elo":1895},
    "Colombia":     {"flag":"🇨🇴","atk":1.65,"def":0.82,"elo":1892},
    "Japan":        {"flag":"🇯🇵","atk":1.55,"def":0.80,"elo":1878},
    "Austria":      {"flag":"🇦🇹","atk":1.58,"def":0.83,"elo":1870},
    "Norway":       {"flag":"🇳🇴","atk":1.70,"def":0.84,"elo":1868},
    "Sweden":       {"flag":"🇸🇪","atk":1.60,"def":0.83,"elo":1855},
    "Canada":       {"flag":"🇨🇦","atk":1.58,"def":0.85,"elo":1850},
    "Senegal":      {"flag":"🇸🇳","atk":1.48,"def":0.84,"elo":1842},
    "Croatia":      {"flag":"🇭🇷","atk":1.46,"def":0.82,"elo":1835},
    "Ivory Coast":  {"flag":"🇨🇮","atk":1.50,"def":0.86,"elo":1832},
    "Australia":    {"flag":"🇦🇺","atk":1.44,"def":0.87,"elo":1825},
    "Ecuador":      {"flag":"🇪🇨","atk":1.40,"def":0.85,"elo":1800},
    "Egypt":        {"flag":"🇪🇬","atk":1.45,"def":0.83,"elo":1795},
    "South Africa": {"flag":"🇿🇦","atk":1.32,"def":0.89,"elo":1778},
    "Ghana":        {"flag":"🇬🇭","atk":1.40,"def":0.89,"elo":1770},
    "Bosnia":       {"flag":"🇧🇦","atk":1.38,"def":0.87,"elo":1765},
    "Iran":         {"flag":"🇮🇷","atk":1.28,"def":0.86,"elo":1755},
    "Paraguay":     {"flag":"🇵🇾","atk":1.33,"def":0.87,"elo":1748},
    "Cape Verde":   {"flag":"🇨🇻","atk":1.22,"def":0.89,"elo":1700},
    "Uzbekistan":   {"flag":"🇺🇿","atk":1.25,"def":0.90,"elo":1695},
}

# Historical penalty shootout win rates (WC + major tournaments through 2024)
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

DEFAULT_HOST_ADVANTAGE = {"USA":0.22,"Mexico":0.30,"Canada":0.18}

# Squad goal-share data (updated June 2026)
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

# API name mapping
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

R32_BRACKET = [
    ("South Africa","Canada",     "Jun 28"),
    ("Brazil",      "Japan",      "Jun 29"),
    ("Morocco",     "Netherlands","Jun 29"),
    ("Norway",      "Ivory Coast","Jun 30"),
    ("France",      "Sweden",     "Jul 1"),
    ("USA",         "Bosnia",     "Jul 1"),
    ("England",     "Senegal",    "Jul 1"),
    ("Belgium",     "TBD",        "Jul 1"),
    ("Spain",       "Austria",    "Jul 2"),
    ("Switzerland", "Iran",       "Jul 2"),
    ("Portugal",    "Ghana",      "Jul 2"),
    ("Australia",   "Egypt",      "Jul 3"),
    ("Argentina",   "Cape Verde", "Jul 3"),
    ("Colombia",    "Croatia",    "Jul 3"),
    ("Mexico",      "TBD",        "Jul 4"),
    ("Germany",     "Paraguay",   "Jul 4"),
]

# Updated FC 25 / EA FC 25 ratings (June 2026 patch)
# (name, team, pos, overall, pace, shooting, passing, dribbling, defending, physical)
PLAYER_CARDS_FALLBACK = [
    # ── Argentina ───────────────────────────────────────────────
    ("L. Messi",          "Argentina",   "RW",  94, 82, 91, 92, 96, 35, 66),
    ("E. Martínez",       "Argentina",   "GK",  88,  0,  0,  0,  0,  0,  0),
    ("C. Romero",         "Argentina",   "CB",  88, 72, 52, 70, 73, 89, 85),
    ("R. De Paul",        "Argentina",   "CM",  85, 74, 74, 86, 84, 76, 79),
    ("J. Álvarez",        "Argentina",   "ST",  85, 86, 84, 77, 86, 45, 78),
    ("L. Martínez",       "Argentina",   "ST",  84, 83, 83, 72, 82, 44, 77),
    ("G. Lo Celso",       "Argentina",   "CM",  83, 75, 73, 85, 86, 65, 72),
    ("N. Molina",         "Argentina",   "RB",  83, 83, 67, 75, 79, 75, 76),
    ("E. Fernández",      "Argentina",   "CM",  83, 69, 73, 84, 81, 77, 77),
    ("Á. Di María",       "Argentina",   "LW",  82, 82, 79, 83, 86, 35, 64),
    ("L. Paredes",        "Argentina",   "CDM", 81, 62, 65, 84, 77, 73, 73),

    # ── France ──────────────────────────────────────────────────
    ("K. Mbappé",         "France",      "ST",  92, 97, 91, 81, 93, 37, 77),
    ("M. Maignan",        "France",      "GK",  88,  0,  0,  0,  0,  0,  0),
    ("A. Griezmann",      "France",      "CAM", 87, 76, 85, 87, 86, 55, 74),
    ("J. Koundé",         "France",      "RB",  86, 79, 53, 76, 80, 84, 75),
    ("A. Tchouaméni",     "France",      "CDM", 85, 71, 64, 82, 77, 85, 84),
    ("A. Upamecano",      "France",      "CB",  85, 77, 48, 69, 70, 86, 84),
    ("T. Hernández",      "France",      "LB",  85, 85, 63, 76, 79, 77, 81),
    ("A. Camavinga",      "France",      "CM",  85, 81, 69, 83, 85, 76, 77),
    ("K. Coman",          "France",      "RW",  85, 93, 79, 79, 87, 39, 73),
    ("M. Thuram",         "France",      "ST",  84, 83, 82, 73, 81, 43, 85),
    ("O. Dembélé",        "France",      "RW",  87, 93, 82, 80, 90, 36, 70),

    # ── Spain ────────────────────────────────────────────────────
    ("G. Rodri",          "Spain",       "CDM", 91, 66, 72, 89, 84, 89, 79),
    ("P. Pedri",          "Spain",       "CM",  89, 75, 76, 89, 92, 66, 69),
    ("L. Yamal",          "Spain",       "RW",  87, 92, 81, 81, 91, 30, 62),
    ("D. Olmo",           "Spain",       "CAM", 86, 79, 81, 85, 87, 62, 73),
    ("D. Carvajal",       "Spain",       "RB",  86, 76, 65, 83, 79, 83, 77),
    ("J. Bellingham",     "Spain",       "CM",  88, 80, 85, 84, 87, 74, 85),
    ("A. Laporte",        "Spain",       "CB",  85, 69, 51, 79, 73, 87, 81),
    ("M. Cucurella",      "Spain",       "LB",  83, 76, 56, 76, 77, 80, 78),
    ("U. Simón",          "Spain",       "GK",  85,  0,  0,  0,  0,  0,  0),
    ("A. Morata",         "Spain",       "ST",  83, 77, 81, 75, 79, 46, 79),
    ("R. Le Normand",     "Spain",       "CB",  83, 69, 45, 71, 66, 84, 82),

    # ── England ──────────────────────────────────────────────────
    ("H. Kane",           "England",     "ST",  91, 71, 94, 84, 84, 48, 83),
    ("P. Foden",          "England",     "CAM", 89, 83, 84, 87, 91, 45, 70),
    ("B. Saka",           "England",     "RW",  88, 86, 83, 85, 88, 56, 73),
    ("J. Bellingham",     "England",     "CM",  88, 80, 85, 84, 87, 74, 85),
    ("T. Alexander-Arnold","England",    "RB",  88, 76, 73, 91, 81, 68, 71),
    ("D. Rice",           "England",     "CDM", 87, 73, 68, 84, 80, 86, 84),
    ("J. Pickford",       "England",     "GK",  85,  0,  0,  0,  0,  0,  0),
    ("J. Trippier",       "England",     "RB",  83, 73, 69, 84, 75, 75, 74),
    ("M. Guehi",          "England",     "CB",  83, 69, 45, 69, 67, 84, 81),
    ("C. Palmer",         "England",     "CAM", 85, 74, 84, 84, 88, 52, 70),
    ("O. Watkins",        "England",     "ST",  84, 83, 82, 72, 80, 46, 79),

    # ── Brazil ───────────────────────────────────────────────────
    ("Vinicius Jr.",      "Brazil",      "LW",  93, 96, 85, 78, 95, 33, 74),
    ("Alisson",           "Brazil",      "GK",  91,  0,  0,  0,  0,  0,  0),
    ("E. Militão",        "Brazil",      "CB",  87, 77, 49, 69, 73, 88, 81),
    ("L. Paquetá",        "Brazil",      "CM",  86, 77, 79, 84, 87, 57, 75),
    ("Rodrygo",           "Brazil",      "RW",  86, 88, 82, 79, 88, 36, 69),
    ("Raphinha",          "Brazil",      "RW",  86, 85, 83, 81, 86, 41, 72),
    ("G. Martinelli",     "Brazil",      "LW",  85, 91, 81, 74, 85, 39, 78),
    ("Casemiro",          "Brazil",      "CDM", 86, 64, 71, 78, 75, 88, 87),
    ("D. Danilo",         "Brazil",      "RB",  82, 77, 61, 78, 74, 78, 78),
    ("M. Renan",          "Brazil",      "CB",  81, 71, 43, 66, 66, 82, 80),
    ("A. Savinho",        "Brazil",      "RW",  82, 90, 76, 75, 84, 33, 68),

    # ── Germany ──────────────────────────────────────────────────
    ("F. Wirtz",          "Germany",     "CAM", 89, 80, 82, 87, 91, 49, 71),
    ("J. Musiala",        "Germany",     "CAM", 89, 86, 83, 84, 92, 53, 73),
    ("J. Kimmich",        "Germany",     "CDM", 89, 69, 73, 92, 83, 83, 77),
    ("M. Ter Stegen",     "Germany",     "GK",  90,  0,  0,  0,  0,  0,  0),
    ("A. Rüdiger",        "Germany",     "CB",  87, 79, 53, 66, 67, 88, 89),
    ("L. Goretzka",       "Germany",     "CM",  85, 76, 79, 83, 83, 80, 86),
    ("K. Havertz",        "Germany",     "ST",  85, 78, 85, 81, 83, 58, 80),
    ("T. Müller",         "Germany",     "CAM", 83, 68, 80, 86, 82, 57, 75),
    ("D. Raum",           "Germany",     "LB",  83, 81, 61, 79, 75, 75, 74),
    ("B. Pavard",         "Germany",     "CB",  83, 75, 56, 74, 70, 83, 79),
    ("L. Nmecha",         "Germany",     "ST",  82, 83, 81, 70, 79, 40, 82),

    # ── Portugal ─────────────────────────────────────────────────
    ("C. Ronaldo",        "Portugal",    "ST",  87, 80, 92, 77, 84, 33, 76),
    ("R. Dias",           "Portugal",    "CB",  89, 73, 53, 71, 71, 90, 83),
    ("B. Fernandes",      "Portugal",    "CAM", 87, 75, 83, 89, 85, 61, 78),
    ("R. Leão",           "Portugal",    "LW",  86, 93, 82, 77, 87, 35, 75),
    ("N. Mendes",         "Portugal",    "LB",  86, 88, 63, 77, 81, 79, 77),
    ("J. Cancelo",        "Portugal",    "RB",  86, 81, 69, 83, 83, 73, 73),
    ("J. Félix",          "Portugal",    "CAM", 85, 83, 82, 82, 88, 41, 71),
    ("D. Costa",          "Portugal",    "GK",  85,  0,  0,  0,  0,  0,  0),
    ("G. Ramos",          "Portugal",    "ST",  84, 75, 84, 70, 78, 42, 78),
    ("P. Neves",          "Portugal",    "CM",  84, 73, 70, 84, 82, 74, 76),
    ("D. Dalot",          "Portugal",    "RB",  83, 81, 61, 75, 77, 75, 73),

    # ── Netherlands ──────────────────────────────────────────────
    ("V. van Dijk",       "Netherlands", "CB",  90, 76, 61, 73, 73, 92, 89),
    ("F. de Jong",        "Netherlands", "CM",  88, 75, 73, 89, 88, 75, 77),
    ("C. Gakpo",          "Netherlands", "LW",  85, 86, 83, 77, 84, 43, 79),
    ("X. Simons",         "Netherlands", "CAM", 85, 84, 80, 84, 86, 52, 70),
    ("M. Depay",          "Netherlands", "ST",  84, 83, 85, 78, 86, 38, 76),
    ("D. Dumfries",       "Netherlands", "RB",  84, 85, 66, 73, 74, 77, 85),
    ("B. Verbruggen",     "Netherlands", "GK",  84,  0,  0,  0,  0,  0,  0),
    ("S. de Vrij",        "Netherlands", "CB",  84, 69, 51, 73, 69, 86, 81),
    ("T. Reijnders",      "Netherlands", "CM",  84, 76, 73, 81, 81, 72, 77),
    ("N. Timber",         "Netherlands", "CB",  83, 75, 47, 71, 71, 83, 79),
    ("R. Malen",          "Netherlands", "LW",  82, 89, 78, 72, 83, 35, 70),

    # ── Morocco ──────────────────────────────────────────────────
    ("A. Hakimi",         "Morocco",     "RB",  87, 91, 73, 77, 83, 71, 77),
    ("Y. Bounou",         "Morocco",     "GK",  87,  0,  0,  0,  0,  0,  0),
    ("Y. En-Nesyri",      "Morocco",     "ST",  83, 84, 82, 65, 76, 43, 78),
    ("H. Ziyech",         "Morocco",     "RW",  83, 79, 81, 84, 85, 41, 66),
    ("S. Amrabat",        "Morocco",     "CDM", 83, 74, 59, 79, 79, 83, 81),
    ("N. Aguerd",         "Morocco",     "CB",  82, 69, 45, 67, 63, 83, 79),
    ("S. Ezzalzouli",     "Morocco",     "LW",  81, 86, 74, 76, 82, 36, 68),
    ("I. Ounahi",         "Morocco",     "CM",  80, 73, 66, 79, 80, 70, 72),
    ("R. Saïss",          "Morocco",     "CB",  80, 63, 49, 69, 65, 83, 79),
    ("B. Diaz",           "Morocco",     "ST",  79, 81, 78, 69, 76, 39, 73),

    # ── Belgium ──────────────────────────────────────────────────
    ("K. De Bruyne",      "Belgium",     "CAM", 91, 75, 86, 94, 88, 65, 79),
    ("T. Courtois",       "Belgium",     "GK",  91,  0,  0,  0,  0,  0,  0),
    ("R. Lukaku",         "Belgium",     "ST",  86, 83, 87, 68, 81, 41, 93),
    ("A. Doku",           "Belgium",     "LW",  85, 96, 76, 74, 89, 32, 68),
    ("L. Trossard",       "Belgium",     "LW",  84, 83, 81, 79, 83, 47, 73),
    ("J. Vertonghen",     "Belgium",     "CB",  82, 65, 47, 73, 69, 85, 79),
    ("Y. Carrasco",       "Belgium",     "LW",  82, 85, 77, 77, 84, 41, 73),
    ("A. Onana",          "Belgium",     "CDM", 82, 70, 62, 80, 76, 80, 78),
    ("T. Alderweireld",   "Belgium",     "CB",  81, 63, 51, 77, 65, 85, 78),

    # ── Colombia ─────────────────────────────────────────────────
    ("L. Díaz",           "Colombia",    "LW",  86, 91, 81, 79, 89, 41, 74),
    ("D. Ospina",         "Colombia",    "GK",  84,  0,  0,  0,  0,  0,  0),
    ("J. Cuadrado",       "Colombia",    "RW",  83, 85, 75, 81, 83, 55, 71),
    ("Y. Mina",           "Colombia",    "CB",  83, 65, 53, 61, 59, 84, 85),
    ("R. Falcao",         "Colombia",    "ST",  81, 69, 84, 70, 78, 38, 74),
    ("J. Lerma",          "Colombia",    "CDM", 81, 71, 61, 75, 71, 81, 83),
    ("J. Muriel",         "Colombia",    "ST",  81, 85, 79, 73, 79, 37, 73),
    ("M. Caicedo",        "Colombia",    "CDM", 85, 75, 63, 79, 79, 85, 83),

    # ── Norway ───────────────────────────────────────────────────
    ("E. Haaland",        "Norway",      "ST",  95, 89, 96, 69, 82, 46, 89),
    ("M. Ødegaard",       "Norway",      "CAM", 88, 79, 83, 91, 91, 59, 69),
    ("A. Sørloth",        "Norway",      "ST",  83, 79, 83, 67, 73, 47, 87),
    ("J. Strand Larsen",  "Norway",      "ST",  80, 79, 79, 63, 73, 39, 83),
    ("S. Berge",          "Norway",      "CM",  80, 69, 67, 79, 75, 77, 75),
    ("R. Nyland",         "Norway",      "GK",  79,  0,  0,  0,  0,  0,  0),
    ("L. Ostigard",       "Norway",      "CB",  79, 67, 43, 59, 57, 80, 81),

    # ── Switzerland ──────────────────────────────────────────────
    ("Y. Sommer",         "Switzerland", "GK",  86,  0,  0,  0,  0,  0,  0),
    ("G. Xhaka",          "Switzerland", "CDM", 85, 65, 73, 86, 78, 81, 79),
    ("M. Akanji",         "Switzerland", "CB",  85, 77, 49, 71, 71, 86, 81),
    ("X. Shaqiri",        "Switzerland", "RW",  83, 81, 81, 83, 84, 43, 73),
    ("F. Schär",          "Switzerland", "CB",  82, 69, 57, 73, 67, 83, 79),
    ("B. Embolo",         "Switzerland", "ST",  81, 81, 79, 73, 77, 45, 83),
    ("R. Vargas",         "Switzerland", "LW",  80, 83, 75, 77, 79, 37, 67),
    ("S. Widmer",         "Switzerland", "RB",  79, 77, 59, 73, 71, 73, 75),

    # ── USA ──────────────────────────────────────────────────────
    ("C. Pulisic",        "USA",         "LW",  85, 89, 81, 82, 87, 45, 75),
    ("T. Adams",          "USA",         "CDM", 82, 73, 59, 77, 73, 83, 83),
    ("M. Turner",         "USA",         "GK",  80,  0,  0,  0,  0,  0,  0),
    ("W. McKennie",       "USA",         "CM",  81, 77, 71, 76, 76, 73, 83),
    ("R. Reyna",          "USA",         "CAM", 81, 81, 74, 80, 84, 44, 66),
    ("F. Weah",           "USA",         "RW",  79, 87, 73, 71, 78, 37, 73),
    ("J. Sargent",        "USA",         "ST",  80, 77, 78, 67, 74, 45, 77),
    ("A. Dest",           "USA",         "RB",  80, 85, 61, 73, 77, 65, 69),
    ("T. Ream",           "USA",         "CB",  78, 61, 41, 67, 59, 79, 77),

    # ── Mexico ───────────────────────────────────────────────────
    ("G. Ochoa",          "Mexico",      "GK",  84,  0,  0,  0,  0,  0,  0),
    ("H. Lozano",         "Mexico",      "RW",  83, 91, 79, 77, 83, 39, 71),
    ("R. Jiménez",        "Mexico",      "ST",  83, 74, 83, 71, 77, 43, 81),
    ("A. Vega",           "Mexico",      "LW",  80, 87, 75, 73, 80, 36, 69),
    ("H. Moreno",         "Mexico",      "CB",  80, 67, 45, 67, 65, 81, 79),
    ("E. Álvarez",        "Mexico",      "CM",  79, 71, 67, 77, 75, 71, 73),
    ("L. Romo",           "Mexico",      "CM",  78, 69, 65, 77, 75, 69, 71),

    # ── Japan ────────────────────────────────────────────────────
    ("W. Endo",           "Japan",       "CDM", 80, 67, 61, 77, 73, 80, 77),
    ("D. Kamada",         "Japan",       "CAM", 80, 77, 77, 79, 80, 57, 71),
    ("T. Minamino",       "Japan",       "CAM", 80, 81, 78, 79, 81, 55, 69),
    ("Ritsu Doan",        "Japan",       "LW",  80, 84, 77, 77, 81, 43, 68),
    ("K. Gonda",          "Japan",       "GK",  78,  0,  0,  0,  0,  0,  0),
    ("H. Ito",            "Japan",       "RB",  79, 81, 59, 71, 71, 71, 71),
    ("S. Itakura",        "Japan",       "CB",  79, 69, 45, 67, 65, 80, 77),
    ("A. Ueda",           "Japan",       "ST",  78, 78, 77, 63, 71, 39, 75),

    # ── Austria ──────────────────────────────────────────────────
    ("D. Alaba",          "Austria",     "CB",  86, 77, 71, 85, 83, 85, 75),
    ("M. Sabitzer",       "Austria",     "CM",  83, 77, 77, 81, 81, 67, 77),
    ("M. Arnautovic",     "Austria",     "ST",  82, 73, 81, 73, 75, 45, 81),
    ("C. Baumgartner",    "Austria",     "CM",  80, 77, 73, 77, 79, 59, 71),
    ("P. Pentz",          "Austria",     "GK",  77,  0,  0,  0,  0,  0,  0),
    ("F. Grillitsch",     "Austria",     "CDM", 78, 65, 61, 77, 71, 75, 73),

    # ── Sweden ───────────────────────────────────────────────────
    ("V. Gyökeres",       "Sweden",      "ST",  86, 83, 87, 69, 82, 43, 86),
    ("A. Isak",           "Sweden",      "ST",  85, 89, 83, 73, 85, 39, 77),
    ("D. Kulusevski",     "Sweden",      "RW",  84, 85, 79, 81, 84, 55, 76),
    ("E. Forsberg",       "Sweden",      "CAM", 82, 79, 77, 83, 83, 51, 69),
    ("R. Olsen",          "Sweden",      "GK",  80,  0,  0,  0,  0,  0,  0),
    ("M. Lustig",         "Sweden",      "RB",  77, 75, 59, 71, 67, 73, 71),

    # ── Canada ───────────────────────────────────────────────────
    ("A. Davies",         "Canada",      "LB",  86, 96, 69, 79, 84, 73, 75),
    ("J. David",          "Canada",      "ST",  84, 83, 86, 73, 82, 39, 73),
    ("T. Buchanan",       "Canada",      "LW",  79, 87, 71, 71, 77, 37, 71),
    ("M. Borjan",         "Canada",      "GK",  78,  0,  0,  0,  0,  0,  0),
    ("S. Larin",          "Canada",      "ST",  78, 77, 77, 65, 71, 39, 75),
    ("K. Johnston",       "Canada",      "CM",  77, 73, 65, 73, 73, 65, 71),

    # ── Senegal ──────────────────────────────────────────────────
    ("S. Mané",           "Senegal",     "LW",  87, 90, 84, 79, 88, 44, 78),
    ("É. Mendy",          "Senegal",     "GK",  86,  0,  0,  0,  0,  0,  0),
    ("K. Koulibaly",      "Senegal",     "CB",  86, 73, 53, 65, 63, 89, 87),
    ("I. Sarr",           "Senegal",     "RW",  83, 91, 77, 73, 81, 37, 73),
    ("P. Gueye",          "Senegal",     "CDM", 83, 75, 63, 77, 75, 83, 81),
    ("N. Jackson",        "Senegal",     "ST",  81, 84, 79, 71, 77, 39, 77),
    ("I. Diallo",         "Senegal",     "CB",  80, 69, 45, 61, 59, 81, 79),

    # ── Croatia ──────────────────────────────────────────────────
    ("L. Modrić",         "Croatia",     "CM",  87, 73, 77, 92, 91, 63, 68),
    ("J. Gvardiol",       "Croatia",     "CB",  88, 83, 57, 73, 77, 88, 83),
    ("D. Livaković",      "Croatia",     "GK",  85,  0,  0,  0,  0,  0,  0),
    ("A. Kramarić",       "Croatia",     "ST",  83, 77, 83, 73, 81, 43, 75),
    ("M. Brozović",       "Croatia",     "CDM", 84, 69, 69, 87, 81, 79, 75),
    ("I. Perišić",        "Croatia",     "LW",  83, 83, 78, 78, 80, 64, 80),

    # ── Ivory Coast ──────────────────────────────────────────────
    ("S. Haller",         "Ivory Coast", "ST",  82, 75, 83, 61, 73, 43, 87),
    ("W. Zaha",           "Ivory Coast", "RW",  81, 87, 75, 73, 83, 39, 73),
    ("N. Pépé",           "Ivory Coast", "RW",  80, 87, 77, 75, 84, 37, 71),
    ("F. Konaté",         "Ivory Coast", "GK",  78,  0,  0,  0,  0,  0,  0),
    ("S. Aurier",         "Ivory Coast", "RB",  80, 83, 63, 71, 71, 73, 77),
    ("E. Gradel",         "Ivory Coast", "LW",  78, 83, 73, 71, 77, 37, 69),

    # ── Australia ────────────────────────────────────────────────
    ("M. Ryan",           "Australia",   "GK",  80,  0,  0,  0,  0,  0,  0),
    ("M. Leckie",         "Australia",   "RW",  78, 85, 71, 73, 77, 43, 71),
    ("A. Hrustic",        "Australia",   "CM",  76, 71, 69, 75, 73, 63, 67),
    ("H. Souttar",        "Australia",   "CB",  78, 63, 45, 65, 61, 79, 81),
    ("M. Goodwin",        "Australia",   "LW",  75, 81, 67, 65, 71, 35, 63),

    # ── Ecuador ──────────────────────────────────────────────────
    ("M. Caicedo",        "Ecuador",     "CDM", 85, 75, 63, 79, 79, 85, 83),
    ("P. Hincapié",       "Ecuador",     "CB",  81, 75, 47, 67, 67, 81, 77),
    ("E. Valencia",       "Ecuador",     "ST",  80, 77, 78, 63, 69, 43, 83),
    ("H. Galíndez",       "Ecuador",     "GK",  77,  0,  0,  0,  0,  0,  0),
    ("A. Preciado",       "Ecuador",     "RB",  76, 79, 57, 67, 67, 67, 71),

    # ── Egypt ────────────────────────────────────────────────────
    ("M. Salah",          "Egypt",       "RW",  91, 88, 89, 83, 90, 45, 78),
    ("O. Marmoush",       "Egypt",       "ST",  86, 85, 84, 77, 82, 46, 77),
    ("M. El-Shenawy",     "Egypt",       "GK",  79,  0,  0,  0,  0,  0,  0),
    ("A. Hegazi",         "Egypt",       "CB",  78, 63, 45, 63, 59, 79, 81),
    ("T. Mohamed",        "Egypt",       "CM",  75, 69, 61, 71, 69, 65, 69),

    # ── South Africa ─────────────────────────────────────────────
    ("P. Tau",            "South Africa","LW",  78, 83, 71, 73, 79, 37, 65),
    ("R. Williams",       "South Africa","GK",  75,  0,  0,  0,  0,  0,  0),
    ("T. Lorch",          "South Africa","RW",  75, 83, 67, 65, 75, 33, 63),
    ("B. Zwane",          "South Africa","CM",  74, 77, 63, 69, 73, 55, 65),
    ("S. Mashele",        "South Africa","CB",  73, 63, 39, 57, 55, 75, 75),

    # ── Ghana ────────────────────────────────────────────────────
    ("T. Partey",         "Ghana",       "CDM", 84, 73, 66, 81, 77, 84, 83),
    ("A. Kudus",          "Ghana",       "CAM", 82, 85, 78, 78, 84, 48, 74),
    ("J. Ayew",           "Ghana",       "ST",  80, 79, 77, 75, 77, 47, 73),
    ("L. Bati",           "Ghana",       "GK",  73,  0,  0,  0,  0,  0,  0),
    ("D. Amartey",        "Ghana",       "CB",  77, 69, 45, 63, 61, 78, 77),

    # ── Bosnia ───────────────────────────────────────────────────
    ("E. Džeko",          "Bosnia",      "ST",  83, 69, 83, 73, 77, 41, 79),
    ("M. Pjanić",         "Bosnia",      "CM",  82, 65, 73, 88, 81, 59, 69),
    ("I. Šehić",          "Bosnia",      "GK",  77,  0,  0,  0,  0,  0,  0),
    ("S. Kolasinac",      "Bosnia",      "LB",  80, 73, 57, 69, 67, 77, 83),
    ("E. Šunjić",         "Bosnia",      "CDM", 75, 61, 53, 71, 63, 75, 73),

    # ── Iran ─────────────────────────────────────────────────────
    ("M. Taremi",         "Iran",        "ST",  83, 77, 85, 73, 79, 45, 79),
    ("S. Azmoun",         "Iran",        "ST",  81, 77, 81, 71, 77, 43, 79),
    ("A. Beiranvand",     "Iran",        "GK",  79,  0,  0,  0,  0,  0,  0),
    ("E. Hajsafi",        "Iran",        "LB",  75, 73, 55, 69, 65, 69, 69),

    # ── Paraguay ─────────────────────────────────────────────────
    ("M. Almiron",        "Paraguay",    "CAM", 81, 83, 73, 77, 81, 53, 71),
    ("R. Sanabria",       "Paraguay",    "ST",  78, 73, 77, 65, 73, 39, 73),
    ("J. Enciso",         "Paraguay",    "RW",  79, 85, 73, 73, 79, 35, 65),
    ("A. Silva",          "Paraguay",    "GK",  75,  0,  0,  0,  0,  0,  0),
    ("F. Balbuena",       "Paraguay",    "CB",  77, 65, 45, 61, 59, 78, 79),

    # ── Cape Verde ───────────────────────────────────────────────
    ("G. Benchimol",      "Cape Verde",  "ST",  73, 75, 71, 61, 69, 37, 71),
    ("V. Varela",         "Cape Verde",  "GK",  69,  0,  0,  0,  0,  0,  0),
    ("J. Landim",         "Cape Verde",  "RW",  71, 77, 65, 63, 69, 33, 63),

    # ── Uzbekistan ───────────────────────────────────────────────
    ("E. Shomurodov",     "Uzbekistan",  "ST",  77, 77, 75, 63, 71, 39, 73),
    ("U. Nishonov",       "Uzbekistan",  "GK",  69,  0,  0,  0,  0,  0,  0),
    ("O. Zubidov",        "Uzbekistan",  "CM",  69, 65, 59, 67, 65, 61, 63),
    ("J. Mirzayev",       "Uzbekistan",  "CB",  69, 61, 37, 55, 51, 71, 71),
]

# ══════════════════════════════════════════════════════════════════════════════
# API HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def api_get(url: str, api_key: str) -> dict | None:
    try:
        r = requests.get(url, headers={"x-apisports-key": api_key}, timeout=12)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def fetch_live_data(api_key: str) -> dict | None:
    return api_get("https://v3.football.api-sports.io/status", api_key)

def parse_account_info(status: dict) -> tuple[str, str, str]:
    """
    Extract plan name and request counts from /status response.
    API-Football returns: {"response": {"account": {...}, "subscription": {...}, "requests": {...}}}
    """
    resp = (status or {}).get("response") or {}
    # Handle both dict (v3) and list (some plan variants)
    if isinstance(resp, list):
        resp = resp[0] if resp else {}
    plan    = (resp.get("subscription") or {}).get("plan", "unknown")
    reqs    = (resp.get("requests") or {})
    used    = str(reqs.get("current", "?"))
    limit   = str(reqs.get("limit_day", "?"))
    return plan, used, limit

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent_international_results(api_key: str) -> tuple[list[dict], list[str]]:
    """
    Returns (results, debug_lines) so caller can show what was fetched.
    Tries multiple league/season combos and reports counts for each.
    """
    # League ID → season pairs. WC 2026 is league 1 season 2026 on API-Football.
    # Also try friendlies (league 834 = international friendlies on some plans).
    league_seasons = [
        (1,   2026, "WC 2026"),
        (1,   2022, "WC 2022"),
        (4,   2024, "UEFA Euro 2024"),
        (9,   2024, "Copa América 2024"),
        (6,   2023, "AFCON 2023"),
        (10,  2023, "AFC Asian Cup 2023"),
        (834, 2025, "Intl Friendlies 2025"),
        (834, 2024, "Intl Friendlies 2024"),
    ]
    cutoff  = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    results = []
    debug   = []

    for lid, season, label in league_seasons:
        url  = f"https://v3.football.api-sports.io/fixtures?league={lid}&season={season}&status=FT"
        data = api_get(url, api_key)
        raw_count = len((data or {}).get("response") or []) if data else 0
        matched   = 0
        if data:
            for fix in (data.get("response") or []):
                date_str  = (fix.get("fixture") or {}).get("date", "")[:10]
                if date_str < cutoff:
                    continue
                home_name = (fix.get("teams") or {}).get("home", {}).get("name", "")
                away_name = (fix.get("teams") or {}).get("away", {}).get("name", "")
                goals_h   = (fix.get("goals") or {}).get("home")
                goals_a   = (fix.get("goals") or {}).get("away")
                home = API_NAME_MAP.get(home_name)
                away = API_NAME_MAP.get(away_name)
                if home and away and goals_h is not None and goals_a is not None:
                    results.append({"home": home, "away": away,
                                     "goals_home": int(goals_h),
                                     "goals_away": int(goals_a),
                                     "date": date_str})
                    matched += 1
        debug.append(f"{label} (league={lid} season={season}): {raw_count} fixtures returned, {matched} matched our teams")

    debug.append(f"Total usable results: {len(results)}")
    return results, debug

@st.cache_data(ttl=3600, show_spinner=False)
def discover_wc2026_league_id(api_key: str) -> tuple[int, str]:
    """
    Find the correct league ID for WC 2026 on this API plan.
    Returns (league_id, debug_log).
    """
    debug = []

    # Strategy A: search by name + season
    for search_name in ["FIFA World Cup", "World Cup"]:
        url  = f"https://v3.football.api-sports.io/leagues?name={search_name.replace(' ', '+')}&season=2026"
        data = api_get(url, api_key)
        resp = (data or {}).get("response") or []
        debug.append(f"Search '{search_name}' season=2026: {len(resp)} results")
        for item in resp:
            lg  = (item.get("league") or {})
            lid = lg.get("id")
            nm  = lg.get("name", "")
            debug.append(f"  → id={lid} name={nm}")
            if lid:
                debug.append(f"✅ Using league_id={lid}")
                return lid, "\n".join(debug)

    # Strategy B: check league id=1 for seasons 2026 and 2025
    for try_season in [2026, 2025]:
        url2  = f"https://v3.football.api-sports.io/leagues?id=1&season={try_season}"
        data2 = api_get(url2, api_key)
        resp2 = (data2 or {}).get("response") or []
        debug.append(f"Check league id=1 season={try_season}: {len(resp2)} results")
        for item in resp2:
            seas_list = [s.get("year") for s in (item.get("seasons") or [])]
            debug.append(f"  Seasons available: {seas_list}")
        if resp2:
            debug.append("✅ league_id=1 confirmed available")
            return 1, "\n".join(debug)

    # Strategy C: all Cup leagues for 2026
    url3  = "https://v3.football.api-sports.io/leagues?type=Cup&season=2026"
    data3 = api_get(url3, api_key)
    resp3 = (data3 or {}).get("response") or []
    debug.append(f"All Cup leagues season=2026: {len(resp3)} results")
    for item in resp3:
        lg   = (item.get("league") or {})
        name = lg.get("name", "").lower()
        lid  = lg.get("id")
        debug.append(f"  id={lid} name={lg.get('name','')}")
        if "world" in name and lid:
            debug.append(f"✅ World Cup candidate found: id={lid}")
            return lid, "\n".join(debug)

    # Strategy D: raw dump of ALL leagues available on this plan
    url4  = "https://v3.football.api-sports.io/leagues?current=true"
    data4 = api_get(url4, api_key)
    resp4 = (data4 or {}).get("response") or []
    debug.append(f"All current leagues on plan: {len(resp4)} results")
    world_candidates = []
    for item in resp4:
        lg   = (item.get("league") or {})
        name = lg.get("name", "").lower()
        lid  = lg.get("id")
        if "world" in name or "fifa" in name:
            world_candidates.append(f"id={lid} name={lg.get('name','')}")
    if world_candidates:
        debug.append("World/FIFA leagues found: " + ", ".join(world_candidates))
    else:
        debug.append("No World/FIFA leagues in current active leagues on this plan")
        debug.append("⚠️ Your API plan may not include the 2026 World Cup")
        debug.append("   Check: api-football.com/documentation → /leagues endpoint")

    debug.append("Falling back to league_id=1 (default)")
    return 1, "\n".join(debug)
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
    AVG = 1.35
    new: dict = {}
    for team, base in BASE_TEAMS.items():
        sc = scored.get(team, [])
        co = conceded.get(team, [])
        if len(sc) >= 3:
            tw  = sum(w for _, w in sc)
            atk = sum(g * w for g, w in sc) / tw
            dfg = sum(g * w for g, w in co) / tw
            new[team] = {**base,
                         "atk": round(0.6 * atk + 0.4 * base["atk"], 3),
                         "def": round(0.6 * (dfg / AVG) + 0.4 * base["def"], 3)}
        else:
            new[team] = base
    return new

@st.cache_data(ttl=3600, show_spinner=False)
def discover_wc2026_league_id(api_key: str) -> tuple[int, str]:
    """
    Search the API for the correct league ID for the 2026 World Cup.
    Returns (league_id, debug_info).
    """
    debug = []
    # Try searching by name
    url  = "https://v3.football.api-sports.io/leagues?name=FIFA+World+Cup&season=2026"
    data = api_get(url, api_key)
    resp = (data or {}).get("response") or []
    debug.append(f"League search (FIFA World Cup 2026): {len(resp)} results")
    for item in resp:
        lg   = (item.get("league") or {})
        seas = (item.get("seasons") or [])
        lid  = lg.get("id")
        name = lg.get("name","")
        debug.append(f"  Found: id={lid} name={name} seasons={[s.get('year') for s in seas]}")
        if lid:
            return lid, "\n".join(debug)

    # Try league ID 1 with different season spellings
    for try_season in [2026, 2025]:
        url2  = f"https://v3.football.api-sports.io/leagues?id=1&season={try_season}"
        data2 = api_get(url2, api_key)
        resp2 = (data2 or {}).get("response") or []
        debug.append(f"League id=1 season={try_season}: {len(resp2)} results")
        if resp2:
            return 1, "\n".join(debug)

    # Broader search — list all active leagues with "world" in name
    url3  = "https://v3.football.api-sports.io/leagues?type=Cup&season=2026"
    data3 = api_get(url3, api_key)
    resp3 = (data3 or {}).get("response") or []
    debug.append(f"Cup leagues season=2026: {len(resp3)} results")
    for item in resp3:
        lg   = (item.get("league") or {})
        name = lg.get("name", "").lower()
        lid  = lg.get("id")
        if "world" in name:
            debug.append(f"  World Cup candidate: id={lid} name={lg.get('name')}")
            return lid, "\n".join(debug)

    debug.append("Could not auto-detect WC 2026 league ID")
    return 1, "\n".join(debug)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_wc2026_top_scorers(api_key: str, league_id: int) -> tuple[dict, str]:
    """
    Multi-strategy scorer fetch using the correct league ID.
    Returns (squads_dict, debug_message).
    """
    debug_lines = []
    debug_lines.append(f"Using league_id={league_id}, season={WC2026_SEASON}")

    def build_squads(team_goals: dict) -> dict:
        squads: dict = {}
        for team, players in team_goals.items():
            total = sum(g for _, g in players)
            if total == 0:
                continue
            top4      = sorted(players, key=lambda x: -x[1])[:4]
            top_share = sum(g for _, g in top4) / total
            sl        = [(nm, round(g / total, 3)) for nm, g in top4]
            if top_share < 1.0:
                sl.append(("Other", round(1.0 - top_share, 3)))
            squads[team] = sl
        return squads

    # ── Strategy 1: /players/topscorers ──────────────────────────────────────
    url1  = (f"https://v3.football.api-sports.io/players/topscorers"
             f"?league={league_id}&season={WC2026_SEASON}")
    data1 = api_get(url1, api_key)
    resp1 = (data1 or {}).get("response") or []
    debug_lines.append(f"Strategy 1 (/topscorers): {len(resp1)} entries")
    team_goals_1: dict = defaultdict(list)
    for entry in resp1:
        p        = entry.get("player") or {}
        stats    = ((entry.get("statistics") or [{}])[0])
        name     = p.get("name", "")
        goals    = int((stats.get("goals") or {}).get("total") or 0)
        team_api = (stats.get("team") or {}).get("name", "")
        team     = API_NAME_MAP.get(team_api)
        if team and name and goals > 0:
            team_goals_1[team].append((name, goals))
    if team_goals_1:
        debug_lines.append(f"✅ Strategy 1 success: {len(team_goals_1)} teams")
        return build_squads(team_goals_1), "\n".join(debug_lines)
    debug_lines.append("Strategy 1: no goals found → trying Strategy 2")

    # ── Strategy 2: /players (full list, scan for goals) ─────────────────────
    url2      = (f"https://v3.football.api-sports.io/players"
                 f"?league={league_id}&season={WC2026_SEASON}&page=1")
    data2     = api_get(url2, api_key)
    resp2     = (data2 or {}).get("response") or []
    total_pg  = (data2 or {}).get("paging", {}).get("total", 1) if data2 else 1
    for pg in range(2, min(total_pg + 1, 6)):
        more = api_get(url2.replace("page=1", f"page={pg}"), api_key)
        if more and more.get("response"):
            resp2.extend(more["response"])
    debug_lines.append(f"Strategy 2 (/players): {len(resp2)} player entries across {total_pg} pages")
    team_goals_2: dict = defaultdict(list)
    for entry in resp2:
        p        = entry.get("player") or {}
        stats    = ((entry.get("statistics") or [{}])[0])
        name     = p.get("name", "")
        goals    = int((stats.get("goals") or {}).get("total") or 0)
        team_api = (stats.get("team") or {}).get("name", "")
        team     = API_NAME_MAP.get(team_api)
        if team and name and goals > 0:
            team_goals_2[team].append((name, goals))
    if team_goals_2:
        debug_lines.append(f"✅ Strategy 2 success: {len(team_goals_2)} teams")
        return build_squads(team_goals_2), "\n".join(debug_lines)
    debug_lines.append("Strategy 2: no goals found → trying Strategy 3 (fixture events)")

    # ── Strategy 3: goal events from each completed fixture ───────────────────
    fix_url  = (f"https://v3.football.api-sports.io/fixtures"
                f"?league={league_id}&season={WC2026_SEASON}&status=FT")
    fix_data = api_get(fix_url, api_key)
    fixtures = (fix_data or {}).get("response") or []
    debug_lines.append(f"Strategy 3: {len(fixtures)} completed fixtures")
    player_goals_raw: dict = defaultdict(lambda: {"goals": 0, "team": ""})
    for fix in fixtures:
        fix_id = (fix.get("fixture") or {}).get("id", "")
        if not fix_id:
            continue
        ev_data = api_get(
            f"https://v3.football.api-sports.io/fixtures/events?fixture={fix_id}",
            api_key)
        for ev in (ev_data or {}).get("response") or []:
            if (ev.get("type") or "").lower() != "goal":
                continue
            detail = (ev.get("detail") or "").lower()
            if "own goal" in detail or "penalty" in detail:
                continue
            pname    = (ev.get("player") or {}).get("name", "")
            team_api = (ev.get("team") or {}).get("name", "")
            team     = API_NAME_MAP.get(team_api)
            if pname and team:
                player_goals_raw[pname]["goals"] += 1
                player_goals_raw[pname]["team"]   = team
    team_goals_3: dict = defaultdict(list)
    for pname, info in player_goals_raw.items():
        if info["goals"] > 0 and info["team"]:
            team_goals_3[info["team"]].append((pname, info["goals"]))
    total_goals_found = sum(len(v) for v in team_goals_3.values())
    debug_lines.append(f"Strategy 3: {total_goals_found} player-goal records from events")
    if team_goals_3:
        debug_lines.append(f"✅ Strategy 3 success: {len(team_goals_3)} teams")
        return build_squads(team_goals_3), "\n".join(debug_lines)

    debug_lines.append("❌ All strategies returned 0 data.")
    debug_lines.append("Possible causes:")
    debug_lines.append("  • WC 2026 group stage may not have produced scorer data yet on this API plan")
    debug_lines.append(f"  • Confirmed league_id={league_id} — check above discovery log if wrong")
    debug_lines.append("  • Your API plan may not include player stats for this competition")
    debug_lines.append("  • Try: https://v3.football.api-sports.io/leagues?id=" + str(league_id) + "&season=2026 in browser with your key")
    return {}, "\n".join(debug_lines)
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_cards_from_api(api_key: str) -> list[tuple]:
    base_url = (f"https://v3.football.api-sports.io/players"
                f"?league={WC2026_LEAGUE_ID}&season={WC2026_SEASON}&page=1")
    data = api_get(base_url, api_key)
    if not data or not data.get("response"):
        return []
    total_pages = ((data.get("paging") or {}).get("total", 1))
    responses   = list(data["response"])
    for page in range(2, min(total_pages + 1, 8)):
        more = api_get(base_url.replace("page=1", f"page={page}"), api_key)
        if more and more.get("response"):
            responses.extend(more["response"])
    pos_map = {"Goalkeeper":"GK","Defender":"CB","Midfielder":"CM","Attacker":"ST"}
    out = []
    for entry in responses:
        p        = entry.get("player") or {}
        stats    = ((entry.get("statistics") or [{}])[0])
        name     = p.get("name", "")
        team_api = ((stats.get("team") or {}).get("name", ""))
        team     = API_NAME_MAP.get(team_api)
        if not team or not name:
            continue
        pos_raw  = ((stats.get("games") or {}).get("position") or "Midfielder")
        pos      = pos_map.get(pos_raw, pos_raw[:2].upper())
        rating_r = ((stats.get("games") or {}).get("rating"))
        try:
            overall = max(60, min(99, round(float(rating_r) * 10))) if rating_r else 75
        except (ValueError, TypeError):
            overall = 75
        out.append((name, team, pos, overall, 0, 0, 0, 0, 0, 0))
    return out

# ══════════════════════════════════════════════════════════════════════════════
# CARD RENDERING
# ══════════════════════════════════════════════════════════════════════════════
def get_card_color(overall: int) -> tuple:
    if overall >= 90:
        return ("linear-gradient(145deg,#b8860b,#ffd700,#b8860b)", "#ffd700", "#3d2800")
    elif overall >= 86:
        return ("linear-gradient(145deg,#8b7536,#d4af37,#8b7536)", "#d4af37", "#2a1f00")
    elif overall >= 82:
        return ("linear-gradient(145deg,#7a7a7a,#c0c0c0,#7a7a7a)", "#c0c0c0", "#1a1a1a")
    else:
        return ("linear-gradient(145deg,#5a3a1a,#a0714a,#5a3a1a)", "#a0714a", "#f5e6d0")

def render_player_card(p: dict) -> str:
    overall              = p["overall"]
    bg, border, text_col = get_card_color(overall)
    flag                 = BASE_TEAMS.get(p["team"], {}).get("flag", "🏳️")
    is_gk                = p["pos"] == "GK"
    stat_labels = ["DIV","HAN","KIC","REF","SPD","POS"] if is_gk else ["PAC","SHO","PAS","DRI","DEF","PHY"]
    stat_vals   = [p["pace"], p["shooting"], p["passing"],
                   p["dribbling"], p["defending"], p["physical"]]
    # Build stat bars using plain string concat — avoids f-string CSS brace collision
    rows = []
    for label, val in zip(stat_labels, stat_vals):
        if val == 0:
            continue
        rows.append(
            '<div style="display:flex;align-items:center;gap:4px;margin:2px 0;">'
            + '<span style="font-size:9px;font-weight:800;width:26px;color:' + text_col + ';opacity:0.85;">' + label + '</span>'
            + '<div style="flex:1;background:rgba(0,0,0,0.2);border-radius:3px;height:5px;">'
            + '<div style="width:' + str(val) + '%;background:' + text_col + ';height:5px;border-radius:3px;opacity:0.9;"></div>'
            + '</div>'
            + '<span style="font-size:10px;font-weight:700;width:22px;text-align:right;color:' + text_col + ';">' + str(val) + '</span>'
            + '</div>'
        )
    stats_html   = "".join(rows)
    no_stats_msg = ('<div style="font-size:9px;opacity:0.7;text-align:center;margin-top:4px;color:'
                    + text_col + ';">Stats in built-in mode</div>') if all(v == 0 for v in stat_vals) else ""
    return (
        '<div style="background:' + bg + ';border:2px solid ' + border + ';border-radius:12px;'
        + 'padding:10px 10px 8px 10px;width:145px;min-height:220px;'
        + 'box-shadow:0 4px 15px rgba(0,0,0,0.4);font-family:Segoe UI,sans-serif;'
        + 'display:flex;flex-direction:column;align-items:center;position:relative;box-sizing:border-box;">'
        + '<div style="font-size:28px;font-weight:900;color:' + text_col + ';line-height:1;">' + str(overall) + '</div>'
        + '<div style="font-size:10px;font-weight:700;color:' + text_col + ';opacity:0.8;letter-spacing:1px;">' + p["pos"] + '</div>'
        + '<div style="font-size:20px;margin:4px 0 2px 0;">' + flag + '</div>'
        + '<div style="font-size:11px;font-weight:800;color:' + text_col + ';text-align:center;'
        + 'line-height:1.2;margin-bottom:6px;max-width:125px;word-break:break-word;">' + p["name"] + '</div>'
        + '<div style="width:100%;border-top:1px solid rgba(0,0,0,0.2);padding-top:5px;">'
        + stats_html + no_stats_msg
        + '</div>'
        + '<div style="position:absolute;bottom:5px;right:7px;font-size:8px;'
        + 'color:' + text_col + ';opacity:0.5;font-weight:600;">' + p["team"][:10] + '</div>'
        + '</div>'
    )

# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME STATE — mutated by sidebar, read by simulation functions
# ══════════════════════════════════════════════════════════════════════════════
TEAMS          = dict(BASE_TEAMS)
HOST_ADVANTAGE = dict(DEFAULT_HOST_ADVANTAGE)
TEAM_BOOST: dict = {}
SQUADS         = dict(FALLBACK_SQUADS)

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def get_team(name: str) -> dict:
    return TEAMS.get(name, {"flag":"🏳️","atk":1.25,"def":0.91,"elo":1650})

def expected_goals(team_a: str, team_b: str) -> tuple:
    ta, tb    = get_team(team_a), get_team(team_b)
    boost_a   = HOST_ADVANTAGE.get(team_a, 0.0)
    boost_b   = HOST_ADVANTAGE.get(team_b, 0.0)
    manual_a  = TEAM_BOOST.get(team_a, 1.0)
    manual_b  = TEAM_BOOST.get(team_b, 1.0)
    xg_a = ta["atk"] * tb["def"] * (1 + boost_a) * manual_a
    xg_b = tb["atk"] * ta["def"] * (1 + boost_b) * manual_b
    return xg_a, xg_b

def simulate_penalties(team_a: str, team_b: str) -> str:
    p_a = PENALTY_WIN_RATE.get(team_a, 0.50)
    p_b = PENALTY_WIN_RATE.get(team_b, 0.50)
    return team_a if random.random() < (p_a / (p_a + p_b)) else team_b

def distribute_goals(team: str, n_goals: int) -> dict:
    players = SQUADS.get(team, [("Other", 1.0)])
    names   = [p[0] for p in players]
    weights = np.array([p[1] for p in players], dtype=float)
    weights /= weights.sum()
    tally: dict = defaultdict(int)
    for _ in range(n_goals):
        tally[np.random.choice(names, p=weights)] += 1
    return dict(tally)

def simulate_match(team_a: str, team_b: str) -> tuple:
    xg_a, xg_b = expected_goals(team_a, team_b)
    ga = int(np.random.poisson(xg_a))
    gb = int(np.random.poisson(xg_b))
    if ga == gb:
        ga += int(np.random.poisson(xg_a * 0.33))
        gb += int(np.random.poisson(xg_b * 0.33))
        if ga == gb:
            winner = simulate_penalties(team_a, team_b)
            return winner, ga, gb, distribute_goals(team_a, ga), distribute_goals(team_b, gb)
    winner = team_a if ga > gb else team_b
    return winner, ga, gb, distribute_goals(team_a, ga), distribute_goals(team_b, gb)

def simulate_tournament() -> tuple:
    survivors: list  = []
    player_goals: dict = defaultdict(int)
    for home, away, _ in R32_BRACKET:
        if away == "TBD":
            survivors.append(home)
        else:
            winner, _, _, sc_h, sc_a = simulate_match(home, away)
            for p, g in {**sc_h, **sc_a}.items(): player_goals[p] += g
            survivors.append(winner)
    while len(survivors) > 1:
        nxt = []
        for i in range(0, len(survivors), 2):
            if i + 1 >= len(survivors):
                nxt.append(survivors[i])
            else:
                winner, _, _, sc_a, sc_b = simulate_match(survivors[i], survivors[i+1])
                for p, g in {**sc_a, **sc_b}.items(): player_goals[p] += g
                nxt.append(winner)
        survivors = nxt
    return survivors[0], dict(player_goals)

@st.cache_data(show_spinner=False)
def run_simulations(n: int, seed: int, teams_hash: str) -> tuple:
    np.random.seed(seed)
    random.seed(seed)
    win_counts: dict = {}
    gb_totals: dict  = defaultdict(list)
    for _ in range(n):
        champ, pg = simulate_tournament()
        win_counts[champ] = win_counts.get(champ, 0) + 1
        for p, g in pg.items():
            gb_totals[p].append(g)
    gb_avg   = {p: sum(gs) / n for p, gs in gb_totals.items() if p != "Other"}
    total_g  = sum(gb_avg.values())
    gb_prob  = {p: v / total_g for p, v in gb_avg.items()} if total_g > 0 else {}
    return win_counts, gb_prob

def _find_team(player: str) -> str:
    for team, players in SQUADS.items():
        if any(nm == player for nm, _ in players):
            return team
    return "—"

def _find_flag(player: str) -> str:
    return TEAMS.get(_find_team(player), {}).get("flag", "")

def h2h_win_prob(team_a: str, team_b: str, n: int = 20_000) -> dict:
    res = {"team_a": 0, "team_b": 0, "draw_90": 0}
    xg_a, xg_b = expected_goals(team_a, team_b)
    for _ in range(n):
        ga = int(np.random.poisson(xg_a))
        gb = int(np.random.poisson(xg_b))
        if ga > gb:   res["team_a"] += 1
        elif gb > ga: res["team_b"] += 1
        else:         res["draw_90"] += 1
    return {k: v / n * 100 for k, v in res.items()}

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("🏆 2026 World Cup Simulator")
st.caption("Poisson goal model · Live API ratings · Historical penalty rates · Golden Boot · Monte Carlo")

# Module-scope defaults so all tabs can read them even without API
ratings_source  = "Built-in (June 2026)"
n_matches_used  = 0
squads_source   = "Built-in squad data"
api_status_log: list[tuple[str,str,str]] = []  # (icon, label, detail)
scorer_debug_log = ""
league_debug_log = ""
wc_league_id_found = WC2026_LEAGUE_ID

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Settings")
    api_key = st.text_input("API-Football Key", type="password",
                            help="api-football.com key for live ratings & scorer data")
    use_api = st.checkbox("Enable live API", value=False)

    if use_api and api_key:
        # Step 1: connection test
        with st.spinner("Connecting…"):
            status = fetch_live_data(api_key)
        if not status:
            st.warning("⚠️ API connection failed.")
            api_status_log.append(("❌", "API connection", "fetch_live_data returned None — check your key"))
        else:
            plan, used, limit = parse_account_info(status)
            api_status_log.append(("✅", "API connected", f"Plan: {plan} · Requests today: {used}/{limit}"))
            st.success("✅ API connected")

            # Step 2: match ratings
            with st.spinner("Fetching match results…"):
                results_data, ratings_debug = fetch_recent_international_results(api_key)
            if results_data:
                live_teams = compute_ratings_from_results(results_data)
                TEAMS.update(live_teams)
                n_matches_used = len(results_data)
                ratings_source = f"Live API ({n_matches_used} matches)"
                api_status_log.append(("✅", "Team ratings", f"{n_matches_used} international matches fetched and applied\n" + "\n".join(ratings_debug)))
                st.success(f"✅ Ratings: {n_matches_used} matches")
            else:
                api_status_log.append(("⚠️", "Team ratings", "0 matches returned\n" + "\n".join(ratings_debug)))

            # Step 3: discover WC 2026 league ID
            with st.spinner("Finding WC 2026 league ID…"):
                wc_league_id_found, league_debug_log = discover_wc2026_league_id(api_key)
            api_status_log.append(("🔍", "League ID discovery", f"Using league_id={wc_league_id_found}"))

            # Step 4: scorer data
            with st.spinner("Fetching scorer data…"):
                api_squads, scorer_debug_log = fetch_wc2026_top_scorers(api_key, wc_league_id_found)
            if api_squads:
                SQUADS.update(api_squads)
                squads_source = f"Live WC scorers ({len(api_squads)} teams)"
                api_status_log.append(("✅", "Scorer data", f"{len(api_squads)} teams · {sum(len(v) for v in api_squads.values())} player records loaded"))
                st.success(f"✅ Scorers: {len(api_squads)} teams")
            else:
                api_status_log.append(("❌", "Scorer data", "0 records returned — see API Status tab for full debug log"))
                st.warning("⚠️ No scorer data — see API Status tab")
    else:
        st.info("🔵 Add an API key to enable live ratings.")
        api_status_log.append(("🔵", "API disabled", "Enable live API and enter a key to connect"))

    st.divider()
    st.subheader("🏟️ Host-nation advantage")
    ha_usa    = st.slider("🇺🇸 USA",    0.0, 0.60, DEFAULT_HOST_ADVANTAGE["USA"],    0.01, key="ha_usa")
    ha_mexico = st.slider("🇲🇽 Mexico", 0.0, 0.60, DEFAULT_HOST_ADVANTAGE["Mexico"], 0.01, key="ha_mex")
    ha_canada = st.slider("🇨🇦 Canada", 0.0, 0.60, DEFAULT_HOST_ADVANTAGE["Canada"], 0.01, key="ha_can")
    HOST_ADVANTAGE["USA"]    = ha_usa
    HOST_ADVANTAGE["Mexico"] = ha_mexico
    HOST_ADVANTAGE["Canada"] = ha_canada

    st.divider()
    st.subheader("⚡ Team boost")
    boost_team = st.selectbox("Team", ["(none)"] + sorted(TEAMS.keys()), key="boost_team")
    boost_val  = st.slider("Attack multiplier", 0.5, 2.0, 1.0, 0.05, key="boost_val",
                           help="1.5 = team scores 50% more than their rating suggests")
    if boost_team != "(none)":
        TEAM_BOOST[boost_team] = boost_val
        if boost_val != 1.0:
            st.caption(f"{TEAMS.get(boost_team,{}).get('flag','')} {boost_team} ×{boost_val:.2f}")

    st.divider()
    st.caption(f"**Ratings:** {ratings_source}")
    st.caption(f"**Scorers:** {squads_source}")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_sim, tab_gb, tab_cards, tab_h2h, tab_bracket, tab_model, tab_api = st.tabs([
    "📊 Simulation", "👟 Golden Boot", "🃏 Player Cards",
    "⚔️ Head-to-head", "🗓️ R32 Bracket", "🔬 How it works", "🔌 API Status",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Tournament Simulation
# ════════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.subheader("Monte Carlo tournament simulation")
    c1, c2 = st.columns(2)
    with c1:
        n_sims = st.slider("Simulations", 1_000, 5_000_000, 100_000, 100_000,
                           help="More = more accurate. 5M takes a few minutes.")
    with c2:
        seed = st.number_input("Random seed (0 = random)", value=0, step=1,
                               help="Fix for reproducible results.")

    if st.button("▶  Run simulation", type="primary", use_container_width=True, key="run_sim"):
        actual_seed = int(seed) if int(seed) != 0 else random.randint(1, 10_000_000)
        teams_hash  = str(hash(str(sorted(
            (k, v["atk"], v["def"], TEAM_BOOST.get(k, 1.0)) for k, v in TEAMS.items()
        ))))
        with st.spinner(f"Simulating {n_sims:,} tournaments…"):
            win_counts, gb_prob = run_simulations(n_sims, actual_seed, teams_hash)

        rows = [{"Flag": get_team(t)["flag"], "Team": t, "Elo": get_team(t)["elo"],
                 "Wins": c, "Win %": round(c / n_sims * 100, 2)}
                for t, c in win_counts.items()]
        df = (pd.DataFrame(rows).sort_values("Win %", ascending=False)
              .reset_index(drop=True))
        df.insert(0, "Rank", df.index + 1)

        top = df.iloc[0]
        runner_str = ""
        if len(df) > 1:
            r = df.iloc[1]
            runner_str = f" · +{top['Win %'] - r['Win %']:.1f}pp over {r['Team']}"
        st.success(f"{top['Flag']} **{top['Team']}** — predicted champion "
                   f"({top['Win %']:.1f}%{runner_str})")

        col_t, col_c = st.columns([3, 2], gap="large")
        with col_t:
            st.markdown("#### Championship probabilities")
            st.dataframe(
                df[["Rank","Flag","Team","Elo","Win %"]]
                .style.format({"Win %":"{:.2f}%","Elo":"{:,}"})
                .background_gradient(cmap="Greens", subset=["Win %"]),
                use_container_width=True, hide_index=True)
        with col_c:
            st.markdown("#### Top 12 win probability")
            st.bar_chart(df.head(12).set_index("Team")[["Win %"]], use_container_width=True)

        st.session_state["gb_prob"] = gb_prob
        st.session_state["n_sims"]  = n_sims
        st.info("👟 Switch to the Golden Boot tab for top scorer predictions.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Golden Boot
# ════════════════════════════════════════════════════════════════════════════════
with tab_gb:
    st.subheader("🥇 Golden Boot — predicted top scorers")
    if "gb_prob" not in st.session_state:
        st.info("Run the tournament simulation first (📊 tab).")
    else:
        gb_prob = st.session_state["gb_prob"]
        n_ran   = st.session_state["n_sims"]
        gb_rows = [{"Player": p, "Team": _find_team(p), "Flag": _find_flag(p),
                    "Golden Boot %": round(v * 100, 2)}
                   for p, v in sorted(gb_prob.items(), key=lambda x: -x[1])
                   if p != "Other"]
        gb_df   = pd.DataFrame(gb_rows).head(20).reset_index(drop=True)
        gb_df.insert(0, "Rank", gb_df.index + 1)

        if gb_df.empty:
            st.warning("No player goal data available.")
        else:
            top_p = gb_df.iloc[0]
            st.success(f"{top_p['Flag']} **{top_p['Player']}** ({top_p['Team']}) — "
                       f"most likely Golden Boot ({top_p['Golden Boot %']:.1f}%)")
            col_gt, col_gc = st.columns([3, 2], gap="large")
            with col_gt:
                st.markdown("#### Top 20 contenders")
                st.dataframe(
                    gb_df[["Rank","Flag","Player","Team","Golden Boot %"]]
                    .style.format({"Golden Boot %":"{:.2f}%"})
                    .background_gradient(cmap="Oranges", subset=["Golden Boot %"]),
                    use_container_width=True, hide_index=True)
            with col_gc:
                st.markdown("#### Top 10 probability")
                st.bar_chart(gb_df.head(10).set_index("Player")[["Golden Boot %"]],
                             use_container_width=True)
            st.caption(f"Based on {n_ran:,} simulations. "
                       f"Probability = proportional share of avg tournament goals. 'Other' excluded.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Player Cards
# ════════════════════════════════════════════════════════════════════════════════
with tab_cards:
    st.subheader("🃏 FIFA Player Cards — 2026 World Cup")
    st.caption("Ranked best to worst · EA FC 25 ratings (June 2026 patch) · Full 6-stat cards")

    raw_cards: list = []
    if use_api and api_key:
        with st.spinner("Fetching player stats from API…"):
            raw_cards = fetch_player_cards_from_api(api_key)
        if raw_cards:
            st.success(f"✅ {len(raw_cards)} players loaded from API")
        else:
            st.info("ℹ️ No API player data yet — using built-in EA FC 25 ratings.")
    if not raw_cards:
        raw_cards = list(PLAYER_CARDS_FALLBACK)

    all_cards = [
        {"name":r[0],"team":r[1],"pos":r[2],"overall":r[3],
         "pace":r[4],"shooting":r[5],"passing":r[6],
         "dribbling":r[7],"defending":r[8],"physical":r[9]}
        for r in raw_cards
    ]
    all_cards.sort(key=lambda x: -x["overall"])

    fc1, fc2, fc3 = st.columns([3, 2, 2])
    with fc1:
        team_opts   = ["All teams"] + sorted({c["team"] for c in all_cards})
        filter_team = st.selectbox("Filter by team", team_opts, key="cards_team")
    with fc2:
        pos_opts    = ["All positions"] + sorted({c["pos"] for c in all_cards})
        filter_pos  = st.selectbox("Filter by position", pos_opts, key="cards_pos")
    with fc3:
        min_ovr = st.slider("Min overall", 60, 94, 60, 1, key="cards_ovr")

    filtered = [c for c in all_cards
                if (filter_team == "All teams" or c["team"] == filter_team)
                and (filter_pos == "All positions" or c["pos"] == filter_pos)
                and c["overall"] >= min_ovr]

    st.caption(f"Showing **{len(filtered)}** players")

    CPR = 6
    for row_start in range(0, len(filtered), CPR):
        cols = st.columns(CPR)
        for col, card in zip(cols, filtered[row_start:row_start + CPR]):
            with col:
                st.markdown(render_player_card(card), unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if not filtered:
        st.info("No players match the current filters.")

    st.divider()
    st.caption("🥇 Gold (90+) · 🟡 Gold (86–89) · ⬜ Silver (82–85) · 🟫 Bronze (<82) · "
               "Sub-stats from EA FC 25. API mode shows live WC ratings; sub-stats require built-in data.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Head-to-head
# ════════════════════════════════════════════════════════════════════════════════
with tab_h2h:
    st.subheader("Head-to-head match probability")
    st.caption("20,000-match simulation · Poisson model · historical penalty rates")

    all_team_names = sorted(TEAMS.keys())
    ca, cvs, cb = st.columns([5, 1, 5])
    with ca:
        team_a = st.selectbox("Team A", all_team_names,
                              index=all_team_names.index("France"), key="h2h_a")
    with cvs:
        st.markdown("<br><br>vs", unsafe_allow_html=True)
    with cb:
        team_b = st.selectbox("Team B", all_team_names,
                              index=all_team_names.index("Argentina"), key="h2h_b")

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
            with c4: st.metric(f"{ta_flag} xG/match",   f"{xg_a:.2f}")
            with c5: st.metric(f"{tb_flag} xG/match",   f"{xg_b:.2f}")
            with c6: st.metric(f"{ta_flag} Pen win %",  f"{pen_a/pen_tot*100:.0f}%")
            with c7: st.metric(f"{tb_flag} Pen win %",  f"{pen_b/pen_tot*100:.0f}%")

            notes = []
            ha_a = HOST_ADVANTAGE.get(team_a, 0)
            ha_b = HOST_ADVANTAGE.get(team_b, 0)
            ba   = TEAM_BOOST.get(team_a, 1.0)
            bb   = TEAM_BOOST.get(team_b, 1.0)
            if ha_a:     notes.append(f"{ta_flag} home +{ha_a*100:.0f}%")
            if ha_b:     notes.append(f"{tb_flag} home +{ha_b*100:.0f}%")
            if ba != 1.0: notes.append(f"{ta_flag} boost ×{ba:.2f}")
            if bb != 1.0: notes.append(f"{tb_flag} boost ×{bb:.2f}")
            if notes:
                st.caption(" · ".join(notes))
            st.caption("Win/draw % = 90 min only. Knockout draws go to extra time then penalties.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — R32 Bracket (visual)
# ════════════════════════════════════════════════════════════════════════════════
with tab_bracket:
    st.subheader("🗓️ Round of 32 — Visual Bracket")
    st.caption("Green = favourite · Red = underdog · Yellow = within 5% · Updates with sidebar settings.")

    with st.spinner("Computing probabilities…"):
        bracket_data = []
        for home, away, date in R32_BRACKET:
            th = get_team(home)
            if away == "TBD":
                bracket_data.append({"date":date,"home":home,"away":"TBD",
                                     "flag_h":th["flag"],"flag_a":"🏳️",
                                     "xg_h":None,"xg_a":None,
                                     "win_h":None,"draw":None,"win_a":None,
                                     "pen_h":None,"pen_a":None})
            else:
                ta       = get_team(away)
                xg_h, xg_a = expected_goals(home, away)
                probs    = h2h_win_prob(home, away, n=10_000)
                ph       = PENALTY_WIN_RATE.get(home, 0.50)
                pa       = PENALTY_WIN_RATE.get(away, 0.50)
                pt       = ph + pa
                bracket_data.append({"date":date,"home":home,"away":away,
                                     "flag_h":th["flag"],"flag_a":ta["flag"],
                                     "xg_h":xg_h,"xg_a":xg_a,
                                     "win_h":probs["team_a"],"draw":probs["draw_90"],"win_a":probs["team_b"],
                                     "pen_h":ph/pt*100,"pen_a":pa/pt*100})

    def match_card(m: dict) -> str:
        if m["win_h"] is None:
            return (
                '<div style="border:1px solid #444;border-radius:8px;padding:8px 10px;'
                'background:#1e1e2e;margin-bottom:6px;font-family:Segoe UI,sans-serif;">'
                '<div style="font-size:10px;color:#888;margin-bottom:4px;">' + m["date"] + '</div>'
                '<div style="display:flex;justify-content:space-between;align-items:center;">'
                '<span style="font-size:13px;font-weight:700;color:#fff;">' + m["flag_h"] + ' ' + m["home"] + '</span>'
                '<span style="font-size:10px;color:#aaa;padding:0 6px;">vs</span>'
                '<span style="font-size:13px;font-weight:700;color:#888;">🏳️ TBD</span>'
                '</div></div>'
            )
        close  = abs(m["win_h"] - m["win_a"]) < 5
        col_h  = "#facc15" if close else ("#4ade80" if m["win_h"] > m["win_a"] else "#f87171")
        col_a  = "#facc15" if close else ("#4ade80" if m["win_a"] > m["win_h"] else "#f87171")
        bh, bd, ba = int(m["win_h"]), int(m["draw"]), int(m["win_a"])
        return (
            '<div style="border:1px solid #333;border-radius:8px;padding:8px 10px;'
            'background:#1a1a2e;margin-bottom:6px;font-family:Segoe UI,sans-serif;">'
            '<div style="font-size:10px;color:#888;margin-bottom:5px;">' + m["date"] + '</div>'
            '<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">'
            '<span style="font-size:13px;min-width:22px;">' + m["flag_h"] + '</span>'
            '<span style="flex:1;font-size:12px;font-weight:700;color:#fff;">' + m["home"] + '</span>'
            '<span style="font-size:11px;color:#aaa;min-width:36px;text-align:right;">xG ' + f'{m["xg_h"]:.2f}' + '</span>'
            '<span style="font-size:12px;font-weight:800;color:' + col_h + ';min-width:38px;text-align:right;">' + f'{m["win_h"]:.1f}%' + '</span>'
            '</div>'
            '<div style="display:flex;height:4px;border-radius:2px;overflow:hidden;margin-bottom:3px;gap:1px;">'
            '<div style="width:' + str(bh) + '%;background:' + col_h + ';"></div>'
            '<div style="width:' + str(bd) + '%;background:#555;"></div>'
            '<div style="width:' + str(ba) + '%;background:' + col_a + ';"></div>'
            '</div>'
            '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
            '<span style="font-size:13px;min-width:22px;">' + m["flag_a"] + '</span>'
            '<span style="flex:1;font-size:12px;font-weight:700;color:#fff;">' + m["away"] + '</span>'
            '<span style="font-size:11px;color:#aaa;min-width:36px;text-align:right;">xG ' + f'{m["xg_a"]:.2f}' + '</span>'
            '<span style="font-size:12px;font-weight:800;color:' + col_a + ';min-width:38px;text-align:right;">' + f'{m["win_a"]:.1f}%' + '</span>'
            '</div>'
            '<div style="border-top:1px solid #333;padding-top:3px;display:flex;justify-content:space-between;">'
            '<span style="font-size:9px;color:#777;">Pen: ' + m["flag_h"] + ' ' + f'{m["pen_h"]:.0f}%' + ' vs ' + m["flag_a"] + ' ' + f'{m["pen_a"]:.0f}%' + '</span>'
            '<span style="font-size:9px;color:#555;">Draw ' + f'{m["draw"]:.1f}%' + '</span>'
            '</div></div>'
        )

    col_l, col_div, col_r = st.columns([5, 1, 5])
    with col_l:
        st.markdown("**Matches 1–8**")
        st.markdown("".join(match_card(m) for m in bracket_data[:8]), unsafe_allow_html=True)
    with col_div:
        st.markdown('<div style="border-left:2px dashed #444;height:100%;margin:0 auto;width:0;"></div>',
                    unsafe_allow_html=True)
    with col_r:
        st.markdown("**Matches 9–16**")
        st.markdown("".join(match_card(m) for m in bracket_data[8:]), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — Model explainer
# ════════════════════════════════════════════════════════════════════════════════
with tab_model:
    if use_api and api_key and n_matches_used > 0:
        st.success(f"🟢 Live ratings active — {n_matches_used} international matches (last 2 years).")
    else:
        st.info("🔵 Built-in ratings (June 2026). Add an API key in the sidebar for live data.")

    st.subheader("How the model works")

    with st.expander("Poisson goal model", expanded=True):
        st.markdown("""
**Goals are drawn from a Poisson distribution each match.**
```
xG_A = attack_A × defense_B × (1 + home_boost_A) × manual_boost_A
xG_B = attack_B × defense_A × (1 + home_boost_B) × manual_boost_B
```
Draws → 30-min extra time (33% xG) → historical penalty shootout rates.

Home advantage is **per-host-nation** and adjustable via sidebar sliders.
""")

    with st.expander("Team & player data"):
        st.markdown("""
**Team ratings** are calibrated from 2024–2026 international results and World Football Elo.
Attack (λ) = expected goals vs average opponent. Defense = multiplier on opponent attack.

**Player cards** use EA FC 25 ratings (June 2026 patch). When API is connected, overall
ratings come from live WC 2026 match performance (match ratings × 10).

**Squad goal shares** are based on 2025–2026 international and club goal records.
""")

    with st.expander("Historical penalty rates"):
        pen_df = pd.DataFrame([
            {"Flag": get_team(t)["flag"], "Team": t, "Win rate": f"{r*100:.0f}%"}
            for t, r in sorted(PENALTY_WIN_RATE.items(), key=lambda x: -x[1])
            if t in TEAMS
        ])
        st.dataframe(pen_df, use_container_width=True, hide_index=True)
        st.caption("Source: all World Cup + major continental tournaments through 2024.")

    with st.expander("Current team ratings"):
        elo_df = pd.DataFrame([
            {"Flag": v["flag"], "Team": k, "Atk λ": v["atk"],
             "Def mult": v["def"], "Boost": TEAM_BOOST.get(k, 1.0), "Elo": v["elo"]}
            for k, v in sorted(TEAMS.items(), key=lambda x: -x[1]["elo"])
        ])
        st.dataframe(
            elo_df.style.format({"Atk λ":"{:.3f}","Def mult":"{:.3f}","Boost":"{:.2f}x","Elo":"{:,}"}),
            use_container_width=True, hide_index=True)

    with st.expander("What a pro model adds"):
        st.markdown("""
| Feature | This model | Pro model |
|---|---|---|
| Goal distribution | Poisson ✅ | Poisson/neg-binomial ✅ |
| Team ratings | Live API + recency weighting ✅ | MLE on 10k+ matches |
| Home advantage | Per-host-nation, adjustable ✅ | Per-stadium + crowd |
| Manual boost | Any team ✅ | Automated |
| Injuries/squad news | Not included | Live feeds |
| Penalties | Historical win rates ✅ | Player-level records |
| Golden Boot | Goal-share weighted ✅ | Shot-level xG per player |
| Bracket | Real 2026 WC ✅ | Real 2026 WC ✅ |
""")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — API Status
# ════════════════════════════════════════════════════════════════════════════════
with tab_api:
    st.subheader("🔌 API Status — Live diagnostics")
    st.caption("Every step of the API connection is shown here with pass/fail and detail.")

    if not use_api or not api_key:
        st.info("Enable the live API checkbox and enter your API key in the sidebar to see diagnostics here.")
    else:
        # Summary status cards
        all_ok = all(icon == "✅" for icon, _, _ in api_status_log)
        if all_ok:
            st.success("✅ All API checks passed — live data is active.")
        else:
            any_ok = any(icon == "✅" for icon, _, _ in api_status_log)
            if any_ok:
                st.warning("⚠️ API partially working — some data is live, some is built-in.")
            else:
                st.error("❌ API not returning data — all built-in data is being used.")

        st.divider()
        st.markdown("### Step-by-step results")
        for icon, label, detail in api_status_log:
            col_icon, col_body = st.columns([1, 10])
            with col_icon:
                st.markdown(f"## {icon}")
            with col_body:
                st.markdown(f"**{label}**")
                # First line as caption, rest in expander if multi-line
                lines = detail.strip().split("\n")
                st.caption(lines[0])
                if len(lines) > 1:
                    with st.expander("Full detail"):
                        st.code("\n".join(lines[1:]))

        st.divider()
        st.markdown("### League ID discovery log")
        st.code(league_debug_log if league_debug_log else "Not run yet — enable API above.")

        st.divider()
        st.markdown("### Scorer fetch log")
        st.code(scorer_debug_log if scorer_debug_log else "Not run yet — enable API above.")

        st.divider()
        st.markdown("### What to do if scorers return 0")
        st.markdown("""
**Check these in order:**

1. **Verify the league ID** — look at the League ID discovery log above. If it says "Could not auto-detect", your API plan may not have the 2026 World Cup indexed yet. Try manually testing this URL in your browser (replace `YOUR_KEY`):
   ```
   https://v3.football.api-sports.io/leagues?name=FIFA+World+Cup&season=2026&X-RapidAPI-Key=YOUR_KEY
   ```

2. **Check your API plan** — free plans on api-football.com often exclude player statistics and live World Cup data. You need at least the **Starter** plan for player stats.

3. **Check requests remaining** — the connection step above shows `Requests today: X/Y`. If you're at the limit, all calls return empty.

4. **Tournament timing** — if matches are still in the group stage, `/topscorers` may not be populated yet. Strategy 3 (fixture events) should still work once matches have finished.

5. **Copy this URL and test it directly** in your browser with your key as the header — if it returns data there but not here, it's a caching issue (click the refresh button in the sidebar to clear cache):
   ```
   https://v3.football.api-sports.io/fixtures?league=1&season=2026&status=FT
   ```
""")

        if st.button("🔄 Clear API cache and retry", key="clear_cache"):
            fetch_live_data.clear()
            fetch_recent_international_results.clear()
            discover_wc2026_league_id.clear()
            fetch_wc2026_top_scorers.clear()
            fetch_player_cards_from_api.clear()
            st.success("Cache cleared — reload the page to re-fetch all data.")
            st.rerun()
