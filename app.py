import hashlib
import random
import re
import unicodedata
import zlib
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson

st.set_page_config(page_title="Champions League Predictor", page_icon="⭐", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#070b1f; }
[data-testid="stSidebar"]          { background:#050817; border-right:1px solid #16204a; }
[data-testid="stHeader"]           { background:transparent; }
.block-container { padding-top:1rem !important; padding-bottom:1rem !important; }
[data-testid="stVerticalBlock"]    { gap:0.35rem !important; }
div[data-testid="column"]          { padding:0 3px !important; }
[data-testid="stMetric"]           { background:#0c1430; border:1px solid #1a2554; border-radius:4px; padding:6px 10px !important; }
[data-testid="stMetricLabel"]      { color:#7d8bbd !important; font-size:10px !important; }
[data-testid="stMetricValue"]      { color:#e6ebff !important; font-size:1.1rem !important; font-weight:700 !important; }
[data-testid="stTabs"] button                       { color:#5d6a99; font-size:12px; font-weight:500; padding:5px 9px; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#e6ebff !important; border-bottom:2px solid #3d6bff !important; }
h1 { color:#e6ebff !important; font-weight:800 !important; font-size:1.3rem !important; margin-bottom:0 !important; }
h2 { color:#c8d1f5 !important; font-weight:700 !important; font-size:1rem !important; }
h3,h4 { color:#9aa6d6 !important; font-weight:600 !important; font-size:0.9rem !important; }
hr { border-color:#16204a !important; margin:0.5rem 0 !important; }
[data-testid="stBaseButton-primary"] { background:#3d6bff !important; color:#fff !important; font-weight:700 !important; border:none !important; border-radius:4px !important; }
[data-testid="stCaptionContainer"]  { color:#56628f !important; font-size:11px !important; }
[data-testid="stDataFrame"]         { border:1px solid #16204a; border-radius:4px; }
label[data-testid="stWidgetLabel"] p { font-size:12px !important; color:#8a96c4 !important; }
.dbar { display:flex; border:1px solid #16204a; border-radius:4px; overflow:hidden; margin-bottom:6px; background:#0a1028; }
.dbar>div { flex:1; padding:5px 9px; border-right:1px solid #16204a; }
.dbar>div:last-child { border-right:none; }
.dbar .dl { font-size:9px; color:#56628f; text-transform:uppercase; letter-spacing:.04em; }
.dbar .dv { font-size:13px; color:#dfe5ff; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════
# Built-in field: the 36 league-phase clubs (pots as drawn for 2025/26).
# name: (code, flag, association, elo, pot, colour_1, colour_2)
CLUB_DATA = {
    "Paris Saint-Germain":  ("PSG","🇫🇷","FRA",2010,1,"#004170","#DA291C"),
    "Real Madrid":          ("RMA","🇪🇸","ESP",1975,1,"#FFFFFF","#FEBE10"),
    "Manchester City":      ("MCI","🏴󠁧󠁢󠁥󠁮󠁧󠁿","ENG",1960,1,"#6CABDD","#1C2C5B"),
    "Bayern Munich":        ("FCB","🇩🇪","GER",1995,1,"#DC052D","#0066B2"),
    "Liverpool":            ("LIV","🏴󠁧󠁢󠁥󠁮󠁧󠁿","ENG",1985,1,"#C8102E","#F6EB61"),
    "Inter":                ("INT","🇮🇹","ITA",1940,1,"#010E80","#000000"),
    "Chelsea":              ("CHE","🏴󠁧󠁢󠁥󠁮󠁧󠁿","ENG",1920,1,"#034694","#FFFFFF"),
    "Borussia Dortmund":    ("BVB","🇩🇪","GER",1850,1,"#FDE100","#000000"),
    "Barcelona":            ("BAR","🇪🇸","ESP",1980,1,"#A50044","#004D98"),
    "Arsenal":              ("ARS","🏴󠁧󠁢󠁥󠁮󠁧󠁿","ENG",2005,2,"#EF0107","#FFFFFF"),
    "Bayer Leverkusen":     ("B04","🇩🇪","GER",1835,2,"#E32221","#000000"),
    "Atlético Madrid":      ("ATM","🇪🇸","ESP",1890,2,"#CB3524","#262F61"),
    "Benfica":              ("SLB","🇵🇹","POR",1830,2,"#E83030","#FFFFFF"),
    "Atalanta":             ("ATA","🇮🇹","ITA",1830,2,"#1E71B8","#000000"),
    "Villarreal":           ("VIL","🇪🇸","ESP",1825,2,"#FFE667","#005187"),
    "Juventus":             ("JUV","🇮🇹","ITA",1840,2,"#FFFFFF","#000000"),
    "Eintracht Frankfurt":  ("SGE","🇩🇪","GER",1790,2,"#E1000F","#000000"),
    "Club Brugge":          ("CLB","🇧🇪","BEL",1745,2,"#0067CE","#000000"),
    "Tottenham":            ("TOT","🏴󠁧󠁢󠁥󠁮󠁧󠁿","ENG",1815,3,"#FFFFFF","#132257"),
    "PSV":                  ("PSV","🇳🇱","NED",1790,3,"#ED1C24","#FFFFFF"),
    "Ajax":                 ("AJA","🇳🇱","NED",1740,3,"#D2122E","#FFFFFF"),
    "Napoli":               ("NAP","🇮🇹","ITA",1870,3,"#12A0D7","#FFFFFF"),
    "Sporting CP":          ("SCP","🇵🇹","POR",1830,3,"#008057","#FFFFFF"),
    "Olympiacos":           ("OLY","🇬🇷","GRE",1720,3,"#E2001A","#FFFFFF"),
    "Slavia Prague":        ("SLA","🇨🇿","CZE",1720,3,"#E30613","#FFFFFF"),
    "Bodø/Glimt":           ("BOD","🇳🇴","NOR",1720,3,"#FFD200","#000000"),
    "Marseille":            ("OM","🇫🇷","FRA",1810,3,"#2FAEE0","#FFFFFF"),
    "Copenhagen":           ("FCK","🇩🇰","DEN",1690,4,"#FFFFFF","#0A2D6E"),
    "Monaco":               ("ASM","🇲🇨","FRA",1780,4,"#E7212D","#FFFFFF"),
    "Galatasaray":          ("GAL","🇹🇷","TUR",1745,4,"#A90432","#FDB912"),
    "Union Saint-Gilloise": ("USG","🇧🇪","BEL",1720,4,"#FFD700","#0033A0"),
    "Qarabağ":              ("QAR","🇦🇿","AZE",1650,4,"#000000","#FFFFFF"),
    "Athletic Club":        ("ATH","🇪🇸","ESP",1810,4,"#EE2523","#FFFFFF"),
    "Newcastle":            ("NEW","🏴󠁧󠁢󠁥󠁮󠁧󠁿","ENG",1880,4,"#241F20","#FFFFFF"),
    "Pafos":                ("PAF","🇨🇾","CYP",1610,4,"#1B3F8B","#FFFFFF"),
    "Kairat Almaty":        ("KAI","🇰🇿","KAZ",1560,4,"#FFD100","#000000"),
}

def ratings_from_elo(elo: float) -> tuple:
    atk = 1.50 + (elo - 1800) * 0.0035
    dfn = 0.84 - (elo - 1800) * 0.0009
    return round(atk, 3), round(dfn, 3)

def make_club(name: str, code: str, flag: str, assoc: str, elo: int, pot: int, c1: str, c2: str) -> dict:
    atk, dfn = ratings_from_elo(elo)
    return {"code":code,"flag":flag,"assoc":assoc,"elo":elo,"pot":pot,
            "c1":c1,"c2":c2,"atk":atk,"def":dfn}

BASE_TEAMS = {n: make_club(n, *v) for n, v in CLUB_DATA.items()}

def unknown_club(name: str, elo: int = 1700) -> dict:
    code = re.sub(r"[^A-Z]", "", name.upper())[:3] or "???"
    return make_club(name, code, "🏳️", "?", elo, 0, "#2a3566", "#c8d1f5")

# ── Player cards (approximate FC-style ratings, 2025/26 squads) ──────────────
# (name, club, nationality flag, position, overall)
# Face stats are derived from position + overall (see derive_stats); add a 6-tuple
# as a 6th element to override them.
PLAYER_CARDS_FALLBACK = [
    # Paris Saint-Germain
    ("O. Dembélé","Paris Saint-Germain","🇫🇷","RW",90),("Vitinha","Paris Saint-Germain","🇵🇹","CM",89),
    ("A. Hakimi","Paris Saint-Germain","🇲🇦","RB",89),("K. Kvaratskhelia","Paris Saint-Germain","🇬🇪","LW",88),
    ("Marquinhos","Paris Saint-Germain","🇧🇷","CB",87),("N. Mendes","Paris Saint-Germain","🇵🇹","LB",87),
    ("J. Neves","Paris Saint-Germain","🇵🇹","CM",86),("D. Doué","Paris Saint-Germain","🇫🇷","RW",85),
    ("W. Pacho","Paris Saint-Germain","🇪🇨","CB",84),("L. Chevalier","Paris Saint-Germain","🇫🇷","GK",83),
    ("G. Ramos","Paris Saint-Germain","🇵🇹","ST",81),
    # Real Madrid
    ("K. Mbappé","Real Madrid","🇫🇷","ST",91),("Vinícius Jr.","Real Madrid","🇧🇷","LW",89),
    ("T. Courtois","Real Madrid","🇧🇪","GK",89),("J. Bellingham","Real Madrid","🏴󠁧󠁢󠁥󠁮󠁧󠁿","CAM",88),
    ("F. Valverde","Real Madrid","🇺🇾","CM",88),("T. Alexander-Arnold","Real Madrid","🏴󠁧󠁢󠁥󠁮󠁧󠁿","RB",86),
    ("A. Rüdiger","Real Madrid","🇩🇪","CB",86),("É. Militão","Real Madrid","🇧🇷","CB",85),
    ("A. Tchouaméni","Real Madrid","🇫🇷","CDM",85),("Rodrygo","Real Madrid","🇧🇷","RW",85),
    ("A. Güler","Real Madrid","🇹🇷","CAM",83),("D. Huijsen","Real Madrid","🇪🇸","CB",82),
    # Manchester City
    ("E. Haaland","Manchester City","🇳🇴","ST",91),("G. Donnarumma","Manchester City","🇮🇹","GK",89),
    ("Rodri","Manchester City","🇪🇸","CDM",89),("P. Foden","Manchester City","🏴󠁧󠁢󠁥󠁮󠁧󠁿","CAM",86),
    ("Bernardo Silva","Manchester City","🇵🇹","CM",86),("R. Dias","Manchester City","🇵🇹","CB",86),
    ("T. Reijnders","Manchester City","🇳🇱","CM",85),("J. Gvardiol","Manchester City","🇭🇷","CB",85),
    ("J. Doku","Manchester City","🇧🇪","LW",84),("O. Marmoush","Manchester City","🇪🇬","ST",84),
    ("R. Cherki","Manchester City","🇫🇷","CAM",83),
    # Bayern Munich
    ("H. Kane","Bayern Munich","🏴󠁧󠁢󠁥󠁮󠁧󠁿","ST",90),("M. Olise","Bayern Munich","🇫🇷","RW",88),
    ("J. Kimmich","Bayern Munich","🇩🇪","CDM",88),("J. Musiala","Bayern Munich","🇩🇪","CAM",88),
    ("M. Neuer","Bayern Munich","🇩🇪","GK",86),("L. Díaz","Bayern Munich","🇨🇴","LW",86),
    ("D. Upamecano","Bayern Munich","🇫🇷","CB",85),("Kim Min-jae","Bayern Munich","🇰🇷","CB",84),
    ("A. Davies","Bayern Munich","🇨🇦","LB",84),("S. Gnabry","Bayern Munich","🇩🇪","RW",82),
    ("A. Pavlović","Bayern Munich","🇩🇪","CM",82),
    # Liverpool
    ("M. Salah","Liverpool","🇪🇬","RW",91),("V. van Dijk","Liverpool","🇳🇱","CB",90),
    ("Alisson","Liverpool","🇧🇷","GK",89),("F. Wirtz","Liverpool","🇩🇪","CAM",88),
    ("A. Isak","Liverpool","🇸🇪","ST",88),("A. Mac Allister","Liverpool","🇦🇷","CM",87),
    ("R. Gravenberch","Liverpool","🇳🇱","CDM",86),("D. Szoboszlai","Liverpool","🇭🇺","CM",85),
    ("I. Konaté","Liverpool","🇫🇷","CB",84),("J. Frimpong","Liverpool","🇳🇱","RB",84),
    ("C. Gakpo","Liverpool","🇳🇱","LW",84),("H. Ekitike","Liverpool","🇫🇷","ST",83),
    # Inter
    ("L. Martínez","Inter","🇦🇷","ST",89),("N. Barella","Inter","🇮🇹","CM",87),
    ("A. Bastoni","Inter","🇮🇹","CB",87),("Y. Sommer","Inter","🇨🇭","GK",85),
    ("H. Çalhanoğlu","Inter","🇹🇷","CDM",85),("F. Dimarco","Inter","🇮🇹","LWB",85),
    ("M. Thuram","Inter","🇫🇷","ST",85),("D. Dumfries","Inter","🇳🇱","RWB",84),
    ("F. Acerbi","Inter","🇮🇹","CB",83),("H. Mkhitaryan","Inter","🇦🇲","CM",82),
    # Chelsea
    ("C. Palmer","Chelsea","🏴󠁧󠁢󠁥󠁮󠁧󠁿","CAM",87),("M. Caicedo","Chelsea","🇪🇨","CDM",87),
    ("E. Fernández","Chelsea","🇦🇷","CM",85),("R. James","Chelsea","🏴󠁧󠁢󠁥󠁮󠁧󠁿","RB",84),
    ("João Pedro","Chelsea","🇧🇷","ST",83),("M. Cucurella","Chelsea","🇪🇸","LB",83),
    ("L. Colwill","Chelsea","🏴󠁧󠁢󠁥󠁮󠁧󠁿","CB",82),("P. Neto","Chelsea","🇵🇹","LW",82),
    ("R. Sánchez","Chelsea","🇪🇸","GK",81),("A. Garnacho","Chelsea","🇦🇷","LW",81),
    ("Estêvão","Chelsea","🇧🇷","RW",80),
    # Borussia Dortmund
    ("G. Kobel","Borussia Dortmund","🇨🇭","GK",87),("S. Guirassy","Borussia Dortmund","🇬🇳","ST",85),
    ("N. Schlotterbeck","Borussia Dortmund","🇩🇪","CB",84),("J. Brandt","Borussia Dortmund","🇩🇪","CAM",82),
    ("K. Adeyemi","Borussia Dortmund","🇩🇪","LW",81),("M. Sabitzer","Borussia Dortmund","🇦🇹","CM",81),
    ("R. Bensebaini","Borussia Dortmund","🇩🇿","LB",80),("M. Beier","Borussia Dortmund","🇩🇪","ST",80),
    # Barcelona
    ("L. Yamal","Barcelona","🇪🇸","RW",89),("Raphinha","Barcelona","🇧🇷","LW",89),
    ("Pedri","Barcelona","🇪🇸","CM",89),("R. Lewandowski","Barcelona","🇵🇱","ST",87),
    ("F. de Jong","Barcelona","🇳🇱","CM",86),("J. Koundé","Barcelona","🇫🇷","RB",85),
    ("Joan García","Barcelona","🇪🇸","GK",84),("D. Olmo","Barcelona","🇪🇸","CAM",84),
    ("P. Cubarsí","Barcelona","🇪🇸","CB",83),("Gavi","Barcelona","🇪🇸","CM",83),
    ("F. Torres","Barcelona","🇪🇸","ST",82),("M. Rashford","Barcelona","🏴󠁧󠁢󠁥󠁮󠁧󠁿","LW",82),
    # Arsenal
    ("B. Saka","Arsenal","🏴󠁧󠁢󠁥󠁮󠁧󠁿","RW",88),("W. Saliba","Arsenal","🇫🇷","CB",88),
    ("D. Rice","Arsenal","🏴󠁧󠁢󠁥󠁮󠁧󠁿","CM",87),("M. Ødegaard","Arsenal","🇳🇴","CAM",87),
    ("D. Raya","Arsenal","🇪🇸","GK",87),("Gabriel","Arsenal","🇧🇷","CB",86),
    ("V. Gyökeres","Arsenal","🇸🇪","ST",86),("M. Zubimendi","Arsenal","🇪🇸","CDM",85),
    ("E. Eze","Arsenal","🏴󠁧󠁢󠁥󠁮󠁧󠁿","CAM",84),("J. Timber","Arsenal","🇳🇱","RB",84),
    ("G. Martinelli","Arsenal","🇧🇷","LW",83),("L. Trossard","Arsenal","🇧🇪","LW",82),
    # Bayer Leverkusen
    ("A. Grimaldo","Bayer Leverkusen","🇪🇸","LWB",85),("P. Schick","Bayer Leverkusen","🇨🇿","ST",83),
    ("E. Tapsoba","Bayer Leverkusen","🇧🇫","CB",83),("A. García","Bayer Leverkusen","🇪🇸","CM",82),
    ("M. Flekken","Bayer Leverkusen","🇳🇱","GK",81),("R. Andrich","Bayer Leverkusen","🇩🇪","CDM",81),
    ("M. Tillman","Bayer Leverkusen","🇺🇸","CAM",80),
    # Atlético Madrid
    ("J. Álvarez","Atlético Madrid","🇦🇷","ST",87),("J. Oblak","Atlético Madrid","🇸🇮","GK",87),
    ("A. Griezmann","Atlético Madrid","🇫🇷","CAM",86),("M. Llorente","Atlético Madrid","🇪🇸","RB",83),
    ("Á. Baena","Atlético Madrid","🇪🇸","LW",83),("Koke","Atlético Madrid","🇪🇸","CM",82),
    ("A. Sørloth","Atlético Madrid","🇳🇴","ST",82),("R. Le Normand","Atlético Madrid","🇪🇸","CB",82),
    ("P. Barrios","Atlético Madrid","🇪🇸","CM",81),("G. Simeone","Atlético Madrid","🇦🇷","RW",80),
    # Benfica
    ("A. Trubin","Benfica","🇺🇦","GK",83),("V. Pavlidis","Benfica","🇬🇷","ST",82),
    ("N. Otamendi","Benfica","🇦🇷","CB",82),("F. Aursnes","Benfica","🇳🇴","CM",81),
    ("G. Sudakov","Benfica","🇺🇦","CAM",80),("D. Lukebakio","Benfica","🇧🇪","RW",80),
    # Atalanta
    ("A. Lookman","Atalanta","🇳🇬","LW",85),("C. De Ketelaere","Atalanta","🇧🇪","CAM",84),
    ("Ederson","Atalanta","🇧🇷","CM",83),("M. Carnesecchi","Atalanta","🇮🇹","GK",83),
    ("G. Scamacca","Atalanta","🇮🇹","ST",81),("I. Hien","Atalanta","🇸🇪","CB",81),
    ("N. Krstović","Atalanta","🇲🇪","ST",79),
    # Villarreal
    ("A. Moleiro","Villarreal","🇪🇸","CAM",81),("G. Moreno","Villarreal","🇪🇸","ST",81),
    ("D. Parejo","Villarreal","🇪🇸","CM",81),("J. Foyth","Villarreal","🇦🇷","CB",81),
    ("G. Mikautadze","Villarreal","🇬🇪","ST",80),("N. Pépé","Villarreal","🇨🇮","RW",79),
    # Juventus
    ("K. Yıldız","Juventus","🇹🇷","LW",85),("Bremer","Juventus","🇧🇷","CB",85),
    ("M. Di Gregorio","Juventus","🇮🇹","GK",83),("D. Vlahović","Juventus","🇷🇸","ST",82),
    ("J. David","Juventus","🇨🇦","ST",82),("K. Thuram","Juventus","🇫🇷","CM",82),
    ("F. Conceição","Juventus","🇵🇹","RW",81),("M. Locatelli","Juventus","🇮🇹","CDM",81),
    ("L. Openda","Juventus","🇧🇪","ST",81),
    # Eintracht Frankfurt
    ("J. Burkardt","Eintracht Frankfurt","🇩🇪","ST",79),("H. Larsson","Eintracht Frankfurt","🇸🇪","CM",79),
    ("R. Koch","Eintracht Frankfurt","🇩🇪","CB",79),("R. Doan","Eintracht Frankfurt","🇯🇵","RW",79),
    ("A. Knauff","Eintracht Frankfurt","🇩🇪","RW",78),("M. Götze","Eintracht Frankfurt","🇩🇪","CAM",78),
    # Club Brugge
    ("C. Tzolis","Club Brugge","🇬🇷","LW",79),("H. Vanaken","Club Brugge","🇧🇪","CAM",78),
    ("S. Mignolet","Club Brugge","🇧🇪","GK",78),("R. Onyedika","Club Brugge","🇳🇬","CDM",78),
    # Tottenham
    ("C. Romero","Tottenham","🇦🇷","CB",85),("G. Vicario","Tottenham","🇮🇹","GK",84),
    ("M. van de Ven","Tottenham","🇳🇱","CB",83),("M. Kudus","Tottenham","🇬🇭","RW",82),
    ("X. Simons","Tottenham","🇳🇱","CAM",82),("P. Porro","Tottenham","🇪🇸","RB",82),
    ("D. Solanke","Tottenham","🏴󠁧󠁢󠁥󠁮󠁧󠁿","ST",80),("B. Johnson","Tottenham","🏴󠁧󠁢󠁥󠁮󠁧󠁿","RW",80),
    ("Richarlison","Tottenham","🇧🇷","ST",79),
    # PSV
    ("G. Til","PSV","🇳🇱","CAM",79),("I. Perišić","PSV","🇭🇷","LW",79),
    ("I. Saibari","PSV","🇲🇦","CM",79),("J. Veerman","PSV","🇳🇱","CM",79),
    ("R. Pepi","PSV","🇺🇸","ST",77),
    # Ajax
    ("M. Godts","Ajax","🇧🇪","LW",78),("K. Taylor","Ajax","🇳🇱","CM",78),
    ("V. Jaroš","Ajax","🇨🇿","GK",78),("W. Weghorst","Ajax","🇳🇱","ST",77),
    # Napoli
    ("K. De Bruyne","Napoli","🇧🇪","CAM",86),("S. McTominay","Napoli","🏴󠁧󠁢󠁳󠁣󠁴󠁿","CM",85),
    ("S. Lobotka","Napoli","🇸🇰","CDM",83),("R. Lukaku","Napoli","🇧🇪","ST",82),
    ("F. Anguissa","Napoli","🇨🇲","CM",82),("G. Di Lorenzo","Napoli","🇮🇹","RB",82),
    ("A. Buongiorno","Napoli","🇮🇹","CB",82),("V. Milinković-Savić","Napoli","🇷🇸","GK",81),
    ("R. Højlund","Napoli","🇩🇰","ST",80),("D. Neres","Napoli","🇧🇷","LW",80),
    # Sporting CP
    ("F. Trincão","Sporting CP","🇵🇹","RW",82),("M. Hjulmand","Sporting CP","🇩🇰","CDM",82),
    ("Pote","Sporting CP","🇵🇹","CAM",81),("G. Inácio","Sporting CP","🇵🇹","CB",81),
    ("L. Suárez","Sporting CP","🇨🇴","ST",80),("Rui Silva","Sporting CP","🇵🇹","GK",80),
    # Olympiacos
    ("A. El Kaabi","Olympiacos","🇲🇦","ST",79),("Chiquinho","Olympiacos","🇵🇹","CAM",77),
    ("D. Podence","Olympiacos","🇵🇹","LW",77),("K. Tzolakis","Olympiacos","🇬🇷","GK",77),
    # Slavia Prague
    ("T. Chorý","Slavia Prague","🇨🇿","ST",76),("L. Provod","Slavia Prague","🇨🇿","LW",76),
    ("D. Zima","Slavia Prague","🇨🇿","CB",76),
    # Bodø/Glimt
    ("P. Berg","Bodø/Glimt","🇳🇴","CM",77),("J. P. Hauge","Bodø/Glimt","🇳🇴","LW",77),
    ("K. Høgh","Bodø/Glimt","🇳🇴","ST",76),
    # Marseille
    ("M. Greenwood","Marseille","🏴󠁧󠁢󠁥󠁮󠁧󠁿","RW",84),("G. Rulli","Marseille","🇦🇷","GK",82),
    ("P. Aubameyang","Marseille","🇬🇦","ST",81),("P. Højbjerg","Marseille","🇩🇰","CDM",81),
    ("A. Gouiri","Marseille","🇩🇿","ST",80),("L. Balerdi","Marseille","🇦🇷","CB",80),
    # Copenhagen
    ("D. Kotarski","Copenhagen","🇭🇷","GK",77),("M. Elyounoussi","Copenhagen","🇳🇴","LW",76),
    ("A. Cornelius","Copenhagen","🇩🇰","ST",75),("L. Lerager","Copenhagen","🇩🇰","CM",75),
    # Monaco
    ("M. Akliouche","Monaco","🇫🇷","RW",81),("D. Zakaria","Monaco","🇨🇭","CDM",81),
    ("L. Hrádecký","Monaco","🇫🇮","GK",81),("A. Golovin","Monaco","🇷🇺","CAM",80),
    ("T. Minamino","Monaco","🇯🇵","CAM",79),("F. Balogun","Monaco","🇺🇸","ST",79),
    ("P. Pogba","Monaco","🇫🇷","CM",79),("M. Biereth","Monaco","🇩🇰","ST",78),
    # Galatasaray
    ("V. Osimhen","Galatasaray","🇳🇬","ST",86),("L. Sané","Galatasaray","🇩🇪","RW",82),
    ("M. Icardi","Galatasaray","🇦🇷","ST",80),("L. Torreira","Galatasaray","🇺🇾","CDM",80),
    ("D. Sánchez","Galatasaray","🇨🇴","CB",80),("B. A. Yılmaz","Galatasaray","🇹🇷","LW",79),
    ("U. Çakır","Galatasaray","🇹🇷","GK",79),
    # Union Saint-Gilloise
    ("P. David","Union Saint-Gilloise","🇨🇦","ST",76),("A. Ait El Hadj","Union Saint-Gilloise","🇧🇪","CAM",75),
    ("C. Burgess","Union Saint-Gilloise","🏴󠁧󠁢󠁥󠁮󠁧󠁿","CB",75),
    # Qarabağ
    ("A. Zoubir","Qarabağ","🇫🇷","LW",74),("Juninho","Qarabağ","🇧🇷","ST",73),
    ("M. Kochalski","Qarabağ","🇵🇱","GK",74),
    # Athletic Club
    ("N. Williams","Athletic Club","🇪🇸","LW",85),("U. Simón","Athletic Club","🇪🇸","GK",84),
    ("I. Williams","Athletic Club","🇬🇭","RW",81),("O. Sancet","Athletic Club","🇪🇸","CAM",81),
    ("D. Vivian","Athletic Club","🇪🇸","CB",80),("G. Guruzeta","Athletic Club","🇪🇸","ST",78),
    # Newcastle
    ("B. Guimarães","Newcastle","🇧🇷","CM",86),("A. Gordon","Newcastle","🏴󠁧󠁢󠁥󠁮󠁧󠁿","LW",83),
    ("S. Tonali","Newcastle","🇮🇹","CM",83),("N. Pope","Newcastle","🏴󠁧󠁢󠁥󠁮󠁧󠁿","GK",83),
    ("S. Botman","Newcastle","🇳🇱","CB",81),("Joelinton","Newcastle","🇧🇷","CM",81),
    ("N. Woltemade","Newcastle","🇩🇪","ST",80),("Y. Wissa","Newcastle","🇨🇩","ST",80),
    # Pafos
    ("Jajá","Pafos","🇧🇷","LW",72),("I. Šunjić","Pafos","🇧🇦","CDM",72),
    ("D. Luckassen","Pafos","🇳🇱","CB",72),
    # Kairat Almaty
    ("D. Satpayev","Kairat Almaty","🇰🇿","ST",70),("T. Anarbekov","Kairat Almaty","🇰🇿","GK",70),
]

# Face-stat offsets relative to overall: PAC SHO PAS DRI DEF PHY (GK: DIV HAN KIC REF SPD POS)
POS_TEMPLATE = {
    "ST":  (-2,  3,-10, -2,-45, -3),
    "LW":  ( 4, -3, -5,  3,-48,-15), "RW": ( 4, -3, -5,  3,-48,-15),
    "CAM": (-6, -2,  2,  3,-35,-18),
    "CM":  (-8, -8,  1,  0, -8, -5),
    "CDM": (-14,-18,-3, -6,  1,  0),
    "CB":  (-12,-40,-16,-16, 2,  0),
    "LB":  ( 2,-20, -6, -5, -6, -8), "RB": ( 2,-20, -6, -5, -6, -8),
    "LWB": ( 3,-18, -4, -4, -8, -8), "RWB":( 3,-18, -4, -4, -8, -8),
    "GK":  ( 1, -2,-10,  2,-35, -1),
}

def derive_stats(name: str, pos: str, ovr: int) -> tuple:
    offs = POS_TEMPLATE.get(pos, POS_TEMPLATE["CM"])
    seed = zlib.crc32(name.encode("utf-8"))
    out = []
    for i, o in enumerate(offs):
        jitter = ((seed >> (i * 4)) & 0x7) - 3
        out.append(int(max(25, min(99, ovr + o + jitter))))
    return tuple(out)

# Shootout edge: 50% base, nudged by the club's best goalkeeper rating.
def pen_strength(club: str, cards: list) -> float:
    gks = [c["overall"] for c in cards if c["team"] == club and c["pos"] == "GK"]
    return float(np.clip(0.5 + ((max(gks) if gks else 78) - 80) * 0.01, 0.4, 0.6))

# Share of a club's goals by position (weighted by overall).
SCORER_POS_WEIGHT = {"ST":1.0,"LW":0.6,"RW":0.6,"CAM":0.45,"CM":0.2,"CDM":0.08,
                     "LB":0.07,"RB":0.07,"LWB":0.1,"RWB":0.1,"CB":0.06,"GK":0.0}
NAMED_SCORER_SHARE = 0.80   # cap on the share of a club's goals assigned to named players
SCORER_WEIGHT_REF  = 2.5    # weight of a full, strong squad (keeps thin squad lists from over-concentrating)

def build_squads(cards: list) -> dict:
    by: dict = defaultdict(list)
    for c in cards:
        w = SCORER_POS_WEIGHT.get(c["pos"], 0.1) * max(0.05, (c["overall"] - 60) / 30) ** 2
        if w > 0: by[c["team"]].append((c["name"], w))
    out = {}
    for team, ps in by.items():
        scale = 1.0 / max(SCORER_WEIGHT_REF, sum(w for _, w in ps) / NAMED_SCORER_SHARE)
        out[team] = [(n, round(w * scale, 4)) for n, w in ps]
    return out

UCL_LEAGUE_ID = 2
FIN_STATUS  = {"FT","AET","PEN","AWD","WO"}
LIVE_STATUS = {"1H","2H","HT","ET","BT","P","INT","LIVE"}

# Normalised API-Football club names → built-in names
CLUB_ALIASES = {
    "paris saint germain":"Paris Saint-Germain","psg":"Paris Saint-Germain","paris sg":"Paris Saint-Germain",
    "real madrid":"Real Madrid","manchester city":"Manchester City","man city":"Manchester City",
    "bayern munchen":"Bayern Munich","bayern munich":"Bayern Munich","fc bayern munchen":"Bayern Munich",
    "liverpool":"Liverpool","inter":"Inter","inter milan":"Inter","internazionale":"Inter",
    "chelsea":"Chelsea","borussia dortmund":"Borussia Dortmund","dortmund":"Borussia Dortmund",
    "barcelona":"Barcelona","arsenal":"Arsenal","bayer leverkusen":"Bayer Leverkusen",
    "atletico madrid":"Atlético Madrid","benfica":"Benfica","sl benfica":"Benfica",
    "atalanta":"Atalanta","villarreal":"Villarreal","juventus":"Juventus",
    "eintracht frankfurt":"Eintracht Frankfurt","club brugge kv":"Club Brugge","club brugge":"Club Brugge",
    "tottenham":"Tottenham","tottenham hotspur":"Tottenham","psv eindhoven":"PSV","psv":"PSV",
    "ajax":"Ajax","napoli":"Napoli","sporting cp":"Sporting CP","sporting lisbon":"Sporting CP",
    "olympiakos piraeus":"Olympiacos","olympiacos":"Olympiacos","olympiacos piraeus":"Olympiacos",
    "slavia praha":"Slavia Prague","slavia prague":"Slavia Prague","bodo glimt":"Bodø/Glimt",
    "marseille":"Marseille","olympique marseille":"Marseille","fc copenhagen":"Copenhagen",
    "kobenhavn":"Copenhagen","fc kobenhavn":"Copenhagen","copenhagen":"Copenhagen",
    "monaco":"Monaco","as monaco":"Monaco","galatasaray":"Galatasaray",
    "union st gilloise":"Union Saint-Gilloise","union saint gilloise":"Union Saint-Gilloise",
    "qarabag":"Qarabağ","qarabag fk":"Qarabağ","athletic club":"Athletic Club","athletic bilbao":"Athletic Club",
    "newcastle":"Newcastle","newcastle united":"Newcastle","pafos":"Pafos","pafos fc":"Pafos",
    "kairat almaty":"Kairat Almaty","kairat":"Kairat Almaty",
}

def norm_club(name: str) -> str:
    if not name: return ""
    s = name.replace("ø", "o").replace("Ø", "O")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in CLUB_ALIASES: return CLUB_ALIASES[s]
    s2 = re.sub(r"^(fc|ac|as|sc|sl) | (fc|cf|sc)$", "", s).strip()
    return CLUB_ALIASES.get(s2, name)

# ══════════════════════════════════════════════════════════════════════
# API (API-Football v3)
# ══════════════════════════════════════════════════════════════════════
API_BASE = "https://v3.football.api-sports.io"

def api_get(url: str, key: str) -> dict | None:
    try:
        r = requests.get(url, headers={"x-apisports-key": key}, timeout=12)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def fetch_status(key: str) -> dict | None:
    return api_get(f"{API_BASE}/status", key)

def parse_account_info(status: dict) -> tuple:
    resp = (status or {}).get("response") or {}
    if isinstance(resp, list):
        resp = resp[0] if resp else {}
    plan = (resp.get("subscription") or {}).get("plan", "unknown")
    reqs = (resp.get("requests") or {})
    return plan, str(reqs.get("current", "?")), str(reqs.get("limit_day", "?"))

def _parse_fixture(fix: dict) -> dict:
    f = fix.get("fixture") or {}; teams = fix.get("teams") or {}
    goals = fix.get("goals") or {}; score = fix.get("score") or {}
    status = (f.get("status") or {}).get("short", "NS")
    hn = norm_club((teams.get("home") or {}).get("name", ""))
    an = norm_club((teams.get("away") or {}).get("name", ""))
    gh = goals.get("home"); ga = goals.get("away")
    pen = score.get("penalty") or {}
    return {"home": hn, "away": an, "status": status,
            "gh": int(gh) if gh is not None else None, "ga": int(ga) if ga is not None else None,
            "pen_h": pen.get("home"), "pen_a": pen.get("away"),
            "round": (fix.get("league") or {}).get("round", "") or "",
            "date": (f.get("date") or "")[:10]}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ucl_fixtures(key: str, season: int) -> tuple:
    data = api_get(f"{API_BASE}/fixtures?league={UCL_LEAGUE_ID}&season={season}", key)
    raw = (data or {}).get("response") or []
    out = [_parse_fixture(x) for x in raw]
    lp = [m for m in out if "league" in m["round"].lower()]
    done = sum(1 for m in lp if m["status"] in FIN_STATUS)
    debug = [f"season={season}: {len(out)} fixtures, {len(lp)} league-phase, {done} finished"]
    for m in [x for x in lp if x["status"] in FIN_STATUS][:6]:
        debug.append(f"  {m['home']} {m['gh']}–{m['ga']} {m['away']} [{m['round']}]")
    return out, "\n".join(debug)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent_results(key: str, season: int) -> tuple:
    results, debug = [], []
    for s in (season, season - 1):
        data = api_get(f"{API_BASE}/fixtures?league={UCL_LEAGUE_ID}&season={s}&status=FT-AET-PEN", key)
        raw = (data or {}).get("response") or []
        n = 0
        for fix in raw:
            m = _parse_fixture(fix)
            if m["home"] and m["away"] and m["gh"] is not None and m["ga"] is not None and m["date"]:
                results.append(m); n += 1
        debug.append(f"UCL {s}/{str(s+1)[2:]}: {len(raw)} fixtures, {n} usable")
    debug.append(f"Total: {len(results)}")
    return results, debug

def compute_ratings(results: list, base: dict) -> dict:
    now = datetime.now()
    sc: dict = defaultdict(list); co: dict = defaultdict(list)
    for r in results:
        d = (now - datetime.strptime(r["date"], "%Y-%m-%d")).days
        w = 1.0 if d <= 180 else (0.7 if d <= 365 else 0.4)
        sc[r["home"]].append((r["gh"], w)); co[r["home"]].append((r["ga"], w))
        sc[r["away"]].append((r["ga"], w)); co[r["away"]].append((r["gh"], w))
    out = {k: dict(v) for k, v in base.items()}
    for t in sc:
        if t not in out: out[t] = unknown_club(t)
    for t, b in out.items():
        s = sc.get(t, [])
        if len(s) >= 3:
            tw = sum(w for _, w in s)
            gf = sum(g * w for g, w in s) / tw
            gc = sum(g * w for g, w in co[t]) / tw
            out[t] = {**b, "atk": round(0.6 * (gf / 0.93) + 0.4 * b["atk"], 3),
                           "def": round(0.6 * (gc / 1.65) + 0.4 * b["def"], 3)}
    return out

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_top_scorers(key: str, season: int) -> tuple:
    data = api_get(f"{API_BASE}/players/topscorers?league={UCL_LEAGUE_ID}&season={season}", key)
    resp = (data or {}).get("response") or []
    out: dict = defaultdict(list)
    for e in resp:
        p = e.get("player") or {}; stats = ((e.get("statistics") or [{}])[0])
        nm = p.get("name", ""); g = int((stats.get("goals") or {}).get("total") or 0)
        club = norm_club((stats.get("team") or {}).get("name", ""))
        if club and nm and g > 0: out[club].append((nm, g))
    return dict(out), f"/players/topscorers season={season}: {len(resp)} entries, {len(out)} clubs"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_cards_api(key: str, season: int) -> list:
    url = f"{API_BASE}/players?league={UCL_LEAGUE_ID}&season={season}&page=1"
    data = api_get(url, key)
    if not data or not data.get("response"): return []
    pages = (data.get("paging") or {}).get("total", 1)
    resp = list(data["response"])
    for pg in range(2, min(pages + 1, 8)):
        more = api_get(url.replace("page=1", f"page={pg}"), key)
        if more and more.get("response"): resp.extend(more["response"])
    pm = {"Goalkeeper": "GK", "Defender": "CB", "Midfielder": "CM", "Attacker": "ST"}
    out = []
    for e in resp:
        p = e.get("player") or {}; stats = ((e.get("statistics") or [{}])[0])
        nm = p.get("name", ""); club = norm_club((stats.get("team") or {}).get("name", ""))
        if not club or not nm: continue
        games = stats.get("games") or {}
        pos = pm.get(games.get("position") or "Midfielder", "CM")
        try: ovr = max(60, min(99, round(float(games.get("rating")) * 10))) if games.get("rating") else 72
        except (TypeError, ValueError): ovr = 72
        out.append({"name": nm, "team": club, "nat": "🏳️", "pos": pos, "overall": ovr,
                    "stats": derive_stats(nm, pos, ovr)})
    return out

# ══════════════════════════════════════════════════════════════════════
# LEAGUE-PHASE DRAW (used when no live fixtures are available)
# ══════════════════════════════════════════════════════════════════════
def _draw_ok(fx: list, assoc: dict, max_same: int) -> bool:  # used by tests/diagnostics
    opp: dict = defaultdict(list)
    for h, a in fx:
        if assoc[h] == assoc[a] and assoc[h] != "?": return False
        opp[h].append(assoc[a]); opp[a].append(assoc[h])
    if max_same:
        for lst in opp.values():
            cnt: dict = defaultdict(int)
            for c in lst:
                cnt[c] += 1
                if cnt[c] > max_same and c != "?": return False
    return True

@st.cache_data(show_spinner=False)
def generate_draw(clubs_spec: tuple, seed: int = 2025) -> tuple:
    """Each club plays 2 opponents from every pot (1 home, 1 away): 144 matches.
    Built block by block (pot × pot); avoids same-association ties and >2 opponents per association."""
    rng = random.Random(seed)
    clubs = sorted(clubs_spec, key=lambda x: (x[1] or 9))
    pots = {p: [c[0] for c in clubs[(p - 1) * 9:p * 9]] for p in (1, 2, 3, 4)}
    assoc = {c[0]: c[2] for c in clubs_spec}
    blocks = [(i, i) for i in (1, 2, 3, 4)] + [(i, j) for i in (1, 2, 3, 4) for j in range(i + 1, 5)]

    def sample_block(i, j):
        A = pots[i]
        if i == j:
            cyc = rng.sample(A, 9)
            return [(cyc[k], cyc[(k + 1) % 9]) for k in range(9)]
        B = pots[j]
        while True:
            s1 = rng.sample(B, 9); s2 = rng.sample(B, 9)
            if all(x != y for x, y in zip(s1, s2)): break
        return [(A[k], s1[k]) for k in range(9)] + [(s2[k], A[k]) for k in range(9)]

    def fits(block, cnt):
        add: dict = defaultdict(int)
        for h, a in block:
            if assoc[h] == assoc[a] and assoc[h] != "?": return False
            add[(h, assoc[a])] += 1; add[(a, assoc[h])] += 1
        return all(cnt[k] + v <= 2 or k[1] == "?" for k, v in add.items())

    for strict in (True, False):
        for _ in range(2000):
            cnt: dict = defaultdict(int); fx: list = []
            for i, j in blocks:
                for _ in range(300):
                    b = sample_block(i, j)
                    if not strict or fits(b, cnt): break
                else:
                    break
                for h, a in b:
                    cnt[(h, assoc[a])] += 1; cnt[(a, assoc[h])] += 1
                fx.extend(b)
            if len(fx) == 144:
                return tuple(fx)
    return tuple(fx)

# ══════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE (vectorised Monte Carlo)
# ══════════════════════════════════════════════════════════════════════
CHUNK = 10_000
PO_PAIRS = ((9, 10, 23, 24), (11, 12, 21, 22), (13, 14, 19, 20), (15, 16, 17, 18))
R16_PAIRS = (((1, 2), 15), ((3, 4), 13), ((5, 6), 11), ((7, 8), 9))

def _two_leg(rng, H, V, XN, ha, pen, TG, rows):
    """H hosts the second leg. Aggregate → extra time (2nd leg) → penalties."""
    v1 = rng.poisson(XN[V, H] * (1 + ha)); h1 = rng.poisson(XN[H, V])
    h2 = rng.poisson(XN[H, V] * (1 + ha)); v2 = rng.poisson(XN[V, H])
    th = h1 + h2; tv = v1 + v2; tie = th == tv
    th = th + rng.poisson(XN[H, V] * (1 + ha) / 3) * tie
    tv = tv + rng.poisson(XN[V, H] / 3) * tie
    p = pen[H] / (pen[H] + pen[V])
    hw = (th > tv) | ((th == tv) & (rng.random(len(H)) < p))
    TG[rows, H] += th; TG[rows, V] += tv
    return np.where(hw, H, V)

def _final(rng, A, B, XN, pen, TG, rows):
    ga = rng.poisson(XN[A, B]); gb = rng.poisson(XN[B, A]); tie = ga == gb
    ga = ga + rng.poisson(XN[A, B] / 3) * tie; gb = gb + rng.poisson(XN[B, A] / 3) * tie
    p = pen[A] / (pen[A] + pen[B])
    aw = (ga > gb) | ((ga == gb) & (rng.random(len(A)) < p))
    TG[rows, A] += ga; TG[rows, B] += gb
    return np.where(aw, A, B)

@st.cache_data(show_spinner=False, max_entries=8)
def run_sims(n_sims: int, seed: int, spec_key: str, _spec: dict) -> dict:
    S = _spec
    T = len(S["names"])
    atk = np.array(S["atk"]); dfn = np.array(S["def"]); pen = np.array(S["pen"]); ha = S["ha"]
    XN = np.outer(atk, dfn)
    fh = np.array(S["fh"], dtype=int); fa = np.array(S["fa"], dtype=int); M = len(fh)
    gh0 = np.array(S["gh0"]); ga0 = np.array(S["ga0"]); played = gh0 >= 0
    Hinc = np.zeros((M, T), np.float32); Hinc[np.arange(M), fh] = 1
    Ainc = np.zeros((M, T), np.float32); Ainc[np.arange(M), fa] = 1
    lamH = XN[fh, fa] * (1 + ha); lamA = XN[fa, fh]
    pl = S["players"]; P = len(pl)
    p_t = np.array([p[1] for p in pl], dtype=int); p_s = np.array([p[2] for p in pl])
    p_c = np.array([p[3] for p in pl], dtype=np.int64)
    rng = np.random.default_rng(seed)
    keys = ("pts", "pos", "gd", "top8", "top24", "r16", "qf", "sf", "final", "win")
    acc = {k: np.zeros(T) for k in keys}
    pos_hist = np.zeros(T * T); cut8 = cut24 = 0.0
    ts_win = np.zeros(P); ts_goals = np.zeros(P)
    bc = lambda x: np.bincount(np.ravel(x), minlength=T)
    done = 0
    while done < n_sims:
        n = min(CHUNK, n_sims - done); rows = np.arange(n)
        GH = rng.poisson(lamH, (n, M)); GA = rng.poisson(lamA, (n, M))
        if played.any():
            GH[:, played] = gh0[played]; GA[:, played] = ga0[played]
        GHf = GH.astype(np.float32); GAf = GA.astype(np.float32)
        hw = (GH > GA).astype(np.float32); dr = (GH == GA).astype(np.float32); aw = (GA > GH).astype(np.float32)
        PTS = (3 * hw + dr) @ Hinc + (3 * aw + dr) @ Ainc
        GF = GHf @ Hinc + GAf @ Ainc
        GD = GF - (GAf @ Hinc + GHf @ Ainc)
        un = (~played).astype(np.float32)
        TG = ((GHf * un) @ Hinc + (GAf * un) @ Ainc).astype(np.int64)
        score = PTS.astype(np.float64) * 1e6 + (GD + 500) * 1e3 + GF + rng.random((n, T)) * 0.9
        order = np.argsort(-score, axis=1)
        pos = np.empty_like(order); pos[rows[:, None], order] = np.arange(T)[None, :]
        seed_at = lambda k: order[:, k - 1]

        po_w = {}
        for s1, s2, u1, u2 in PO_PAIRS:
            f = rng.random(n) < .5
            va = np.where(f, seed_at(u1), seed_at(u2)); vb = np.where(f, seed_at(u2), seed_at(u1))
            po_w[s1] = (_two_leg(rng, seed_at(s1), va, XN, ha, pen, TG, rows),
                        _two_leg(rng, seed_at(s2), vb, XN, ha, pen, TG, rows))
        halves = []
        for (a, b), grp in R16_PAIRS:
            w1, w2 = po_w[grp]; f = rng.random(n) < .5
            ra = _two_leg(rng, seed_at(a), np.where(f, w1, w2), XN, ha, pen, TG, rows)
            rb = _two_leg(rng, seed_at(b), np.where(f, w2, w1), XN, ha, pen, TG, rows)
            g = rng.random(n) < .5
            halves.append((np.where(g, ra, rb), np.where(g, rb, ra)))

        def ko(a, b):
            H = np.where(pos[rows, a] < pos[rows, b], a, b); V = np.where(H == a, b, a)
            return _two_leg(rng, H, V, XN, ha, pen, TG, rows)
        qa1 = ko(halves[0][0], halves[3][0]); qa2 = ko(halves[1][0], halves[2][0])
        qb1 = ko(halves[0][1], halves[3][1]); qb2 = ko(halves[1][1], halves[2][1])
        sa = ko(qa1, qa2); sb = ko(qb1, qb2)
        ch = _final(rng, sa, sb, XN, pen, TG, rows)

        acc["pts"] += PTS.sum(0); acc["pos"] += (pos + 1).sum(0); acc["gd"] += GD.sum(0)
        acc["top8"] += (pos < 8).sum(0); acc["top24"] += (pos < 24).sum(0)
        acc["r16"] += bc(order[:, :8]) + sum(bc(w) for pr in po_w.values() for w in pr)
        acc["qf"] += sum(bc(t) for pr in halves for t in pr)
        acc["sf"] += bc(np.concatenate([qa1, qa2, qb1, qb2]))
        acc["final"] += bc(np.concatenate([sa, sb])); acc["win"] += bc(ch)
        pos_hist += np.bincount((np.arange(T)[None, :] * T + pos).ravel(), minlength=T * T)
        srt = -np.sort(-PTS, axis=1); cut8 += srt[:, 7].sum(); cut24 += srt[:, 23].sum()
        if P:
            PG = rng.binomial(TG[:, p_t], p_s) + p_c
            mx = PG.max(1, keepdims=True); m = (PG == mx) & (mx > 0)
            ts_win += (m / np.maximum(m.sum(1, keepdims=True), 1)).sum(0)
            ts_goals += PG.sum(0)
        done += n
    N = float(n_sims)
    return {"names": list(S["names"]), "n": n_sims,
            **{k: (v / N).tolist() for k, v in acc.items()},
            "pos_hist": (pos_hist.reshape(T, T) / N).tolist(),
            "cut8": cut8 / N, "cut24": cut24 / N,
            "players": [p[0] for p in pl], "player_club": [S["names"][p[1]] for p in pl],
            "player_now": p_c.tolist(), "ts_win": (ts_win / N).tolist(), "ts_goals": (ts_goals / N).tolist()}

# ── Closed-form match maths (used by H2H, fixtures and the bracket) ──────────
def get_team(n: str) -> dict:
    return TEAMS.get(n) or unknown_club(n)

def xg_pair(home: str, away: str, neutral: bool = False) -> tuple:
    th = get_team(home); ta = get_team(away)
    lh = th["atk"] * TEAM_BOOST.get(home, 1.0) * ta["def"] * (1.0 if neutral else 1 + HOME_ADV)
    la = ta["atk"] * TEAM_BOOST.get(away, 1.0) * th["def"]
    return lh, la

def outcome_probs(l1: float, l2: float, k: int = 14) -> tuple:
    g = np.outer(poisson.pmf(np.arange(k), l1), poisson.pmf(np.arange(k), l2))
    w = float(np.tril(g, -1).sum()); d = float(np.trace(g)); l = float(np.triu(g, 1).sum())
    t = w + d + l
    return w / t, d / t, l / t

def top_scores(l1: float, l2: float, n: int = 5) -> list:
    k = np.arange(9)
    g = np.outer(poisson.pmf(k, l1), poisson.pmf(k, l2))
    flat = sorted(((int(i), int(j), float(g[i, j]) * 100) for i in k for j in k), key=lambda x: -x[2])
    return flat[:n]

def pen_share(a: str, b: str) -> float:
    pa = PEN.get(a, .5); pb = PEN.get(b, .5)
    return pa / (pa + pb)

def match_probs(home: str, away: str, neutral: bool = False) -> dict:
    lh, la = xg_pair(home, away, neutral)
    w, d, l = outcome_probs(lh, la)
    return {"home": w * 100, "draw": d * 100, "away": l * 100, "xh": lh, "xa": la, "scores": top_scores(lh, la)}

def tie_advance(H: str, V: str) -> float:
    """P(H advances) over two legs, H hosting the second leg."""
    h2, v2 = xg_pair(H, V)       # second leg at H
    v1, h1 = xg_pair(V, H)       # first leg at V
    w, d, _ = outcome_probs(h1 + h2, v1 + v2, k=18)
    ew, ed, _ = outcome_probs(h2 / 3, v2 / 3)
    return w + d * (ew + ed * pen_share(H, V))

def final_win(a: str, b: str) -> float:
    la, lb = xg_pair(a, b, neutral=True)
    w, d, _ = outcome_probs(la, lb)
    ew, ed, _ = outcome_probs(la / 3, lb / 3)
    return w + d * (ew + ed * pen_share(a, b))

# ══════════════════════════════════════════════════════════════════════
# CARD RENDER
# ══════════════════════════════════════════════════════════════════════
TIER_STYLE = {
    "ICON":   {"acc": "#ffe08a",
               "rim": "linear-gradient(135deg,#fff7cf 0%,#f3c64e 22%,#9c6a12 45%,#ffe9a0 62%,#c98d1f 80%,#fff3c0 100%)",
               "bg":  "linear-gradient(165deg,#2a0f4f 0%,#5b2ca0 42%,#1a0b3d 70%,#0b0620 100%)"},
    "GOLD":   {"acc": "#f1cf6b",
               "rim": "linear-gradient(135deg,#fbe8a6 0%,#d4a537 30%,#8a6416 55%,#f5d77a 78%,#b8862a 100%)",
               "bg":  "linear-gradient(165deg,#081a55 0%,#1f4fc4 45%,#0a1a5c 72%,#050d33 100%)"},
    "SILVER": {"acc": "#dfe7f7",
               "rim": "linear-gradient(135deg,#ffffff 0%,#aab6cc 30%,#5f6c84 55%,#e8eef9 78%,#8793a9 100%)",
               "bg":  "linear-gradient(165deg,#0c2142 0%,#31588f 45%,#0e2448 72%,#081529 100%)"},
    "BRONZE": {"acc": "#e7b07e",
               "rim": "linear-gradient(135deg,#f6c9a0 0%,#b0703e 32%,#6b3f1d 55%,#e2a571 78%,#8f5a2e 100%)",
               "bg":  "linear-gradient(165deg,#1d1622 0%,#43344f 45%,#1e1724 72%,#100c14 100%)"},
}

CARD_CSS = """
<style>
@keyframes ushine { 0% { background-position:-180% 0; } 55%,100% { background-position:280% 0; } }
@keyframes uholo  { from { filter:hue-rotate(0deg); } to { filter:hue-rotate(360deg); } }
@keyframes upop   { 0% { transform:scale(.6) rotateY(90deg); opacity:0; } 100% { transform:scale(1) rotateY(0); opacity:1; } }
@keyframes uroll  { 0% { transform:rotate(0) scale(.4); opacity:.2; } 65% { transform:rotate(600deg) scale(1.15); opacity:1; } 100% { transform:rotate(720deg) scale(1); } }
@keyframes uglow  { 0%,100% { box-shadow:0 0 6px rgba(255,224,138,.5); } 50% { box-shadow:0 0 18px rgba(255,224,138,.95); } }
.uwrap { transition:transform .25s ease, filter .25s ease; perspective:600px; }
.uwrap:hover { transform:translateY(-6px) scale(1.05); filter:drop-shadow(0 0 12px var(--acc)) brightness(1.08); }
.ucard { font-size:10px; position:relative; width:100%; max-width:17em; margin:0 auto; padding:.32em;
         clip-path:polygon(50% 0,100% 6%,100% 87%,50% 100%,0 87%,0 6%); background:var(--rim);
         animation:upop .45s ease-out; }
.ucard.sm { font-size:6.4px; }
.ubody { position:relative; clip-path:polygon(50% 0,100% 6%,100% 87%,50% 100%,0 87%,0 6%); overflow:hidden;
         background:radial-gradient(circle at 18% 20%,rgba(255,255,255,.45) 0 .08em,transparent .2em),
                    radial-gradient(circle at 82% 30%,rgba(255,255,255,.35) 0 .08em,transparent .2em),
                    radial-gradient(circle at 64% 9%,rgba(255,255,255,.3) 0 .08em,transparent .2em),
                    repeating-linear-gradient(135deg,rgba(255,255,255,.035) 0 .5em,transparent .5em 1.2em),
                    var(--bg);
         min-height:26em; padding:1.6em 1em 3.2em; display:flex; flex-direction:column; align-items:center; box-sizing:border-box; }
.ubody::before { content:""; position:absolute; inset:0; pointer-events:none; z-index:3;
         background:linear-gradient(115deg,transparent 35%,rgba(255,255,255,.32) 47%,rgba(255,255,255,.06) 53%,transparent 64%);
         background-size:250% 100%; animation:ushine 4.5s ease-in-out infinite; }
.ucard.icon .ubody::after { content:""; position:absolute; inset:0; pointer-events:none; z-index:2; mix-blend-mode:screen;
         background:conic-gradient(from 0deg,rgba(255,0,170,.22),rgba(0,210,255,.22),rgba(0,255,140,.22),rgba(255,230,0,.22),rgba(255,0,170,.22));
         animation:uholo 6s linear infinite; }
.utop { display:flex; width:100%; justify-content:space-between; align-items:flex-start; position:relative; z-index:4; }
.uovr { font-size:3.2em; font-weight:900; line-height:.9; color:var(--acc); text-shadow:0 .06em .2em rgba(0,0,0,.6); }
.upos { font-size:1.1em; font-weight:800; color:#fff; letter-spacing:.1em; margin-top:.2em; text-align:center; }
.uflag { font-size:1.6em; margin-top:.2em; text-align:center; }
.ubadge { width:3.6em; height:3.6em; border-radius:50%; display:flex; align-items:center; justify-content:center;
          font-size:1em; font-weight:900; box-shadow:0 0 0 .15em var(--acc),0 .2em .5em rgba(0,0,0,.5); }
.uavatar { position:relative; z-index:4; width:7.4em; height:7.4em; border-radius:50%; margin:-.6em 0 .5em;
           display:flex; align-items:center; justify-content:center; font-size:1em;
           box-shadow:0 0 0 .22em var(--acc),0 .4em 1em rgba(0,0,0,.55),inset 0 -1em 1.6em rgba(0,0,0,.35); }
.uavatar span { font-size:2.9em; font-weight:900; color:#fff; letter-spacing:-.03em; text-shadow:0 .08em .25em rgba(0,0,0,.7); }
.uname { position:relative; z-index:4; width:100%; text-align:center; font-size:1.25em; font-weight:900; color:#fff;
         text-transform:uppercase; letter-spacing:.05em; padding:.3em 0; margin-bottom:.5em; word-break:break-word;
         border-top:1px solid var(--acc); border-bottom:1px solid var(--acc);
         background:linear-gradient(90deg,transparent,rgba(0,0,0,.35),transparent); }
.ustats { position:relative; z-index:4; display:grid; grid-template-columns:1fr 1fr; gap:.25em 1.1em; width:100%; padding:0 .5em; box-sizing:border-box; }
.ustat { display:flex; gap:.45em; align-items:baseline; }
.ustat b { font-size:1.35em; font-weight:900; color:#fff; min-width:1.4em; }
.ustat i { font-style:normal; font-size:.9em; font-weight:800; color:var(--acc); letter-spacing:.05em; }
.utier { position:absolute; bottom:1.5em; left:0; right:0; text-align:center; z-index:4; font-size:.8em; font-weight:900;
         letter-spacing:.25em; color:var(--acc); }
.unew { position:absolute; top:1.2em; left:50%; z-index:5; transform:translateX(-50%) rotate(-6deg); background:#ff3d71; color:#fff;
        font-size:.9em; font-weight:900; padding:.15em .6em; border-radius:.4em; letter-spacing:.08em; }
.ucard.icon .uovr { animation:uholo 6s linear infinite; }
.dice { width:72px; height:72px; border-radius:14px; display:grid; grid-template-columns:repeat(3,1fr); grid-template-rows:repeat(3,1fr);
        padding:10px; box-sizing:border-box; margin:6px auto; animation:uroll .8s cubic-bezier(.2,.8,.3,1);
        box-shadow:inset 0 -6px 0 rgba(0,0,0,.18),0 6px 16px rgba(0,0,0,.5); }
.dice.six { animation:uroll .8s cubic-bezier(.2,.8,.3,1), uglow 1.6s ease-in-out .8s infinite; }
.pip { width:12px; height:12px; border-radius:50%; align-self:center; justify-self:center; }
.pitch { position:relative; border-radius:10px; padding:14px 6px; border:2px solid rgba(255,255,255,.25);
         background:radial-gradient(circle at 50% 50%,transparent 0 58px,rgba(255,255,255,.22) 59px 61px,transparent 62px),
                    linear-gradient(rgba(255,255,255,.22),rgba(255,255,255,.22)) center/100% 2px no-repeat,
                    repeating-linear-gradient(180deg,#0f6b34 0 60px,#127a3b 60px 120px); }
.prow { display:flex; justify-content:space-evenly; align-items:center; margin:6px 0; gap:4px; }
.pslot { flex:0 1 110px; min-width:0; }
.pempty { height:118px; border:2px dashed rgba(255,255,255,.45); border-radius:10px; display:flex; flex-direction:column;
          align-items:center; justify-content:center; color:rgba(255,255,255,.8); font-weight:800; font-size:13px; background:rgba(0,0,0,.15); }
.pempty.active { border-color:#ffe08a; color:#ffe08a; box-shadow:0 0 14px rgba(255,224,138,.6); }
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

def _lum(hexc: str) -> float:
    h = hexc.lstrip("#")
    if len(h) != 6: return 128.0
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 0.299 * r + 0.587 * g + 0.114 * b

def card_tier(ovr: int) -> str:
    if ovr >= 90: return "ICON"
    if ovr >= 86: return "GOLD"
    if ovr >= 82: return "SILVER"
    return "BRONZE"

def _initials(name: str) -> str:
    parts = [p for p in re.split(r"[\s.]+", name) if p]
    if not parts: return "?"
    if len(parts) == 1: return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()

def render_card(p: dict, size: str = "md", badge: str = "") -> str:
    tier = card_tier(p["overall"]); ts = TIER_STYLE[tier]
    club = get_team(p["team"])
    c1, c2, code = club["c1"], club["c2"], club["code"]
    btc = "#111" if _lum(c1) > 150 else "#fff"
    labs = ["DIV","HAN","KIC","REF","SPD","POS"] if p["pos"] == "GK" else ["PAC","SHO","PAS","DRI","DEF","PHY"]
    cells = "".join(f'<div class="ustat"><b>{v}</b><i>{l}</i></div>' for l, v in zip(labs, p["stats"]))
    cls = "ucard" + (" sm" if size == "sm" else "") + (" icon" if tier == "ICON" else "")
    new = f'<div class="unew">{badge}</div>' if badge else ""
    return (
        f'<div class="uwrap" style="--acc:{ts["acc"]};"><div class="{cls}" style="--rim:{ts["rim"]};--bg:{ts["bg"]};">'
        f'<div class="ubody">{new}'
        f'<div class="utop"><div><div class="uovr">{p["overall"]}</div><div class="upos">{p["pos"]}</div>'
        f'<div class="uflag">{p.get("nat", "")}</div></div>'
        f'<div class="ubadge" style="background:{c1};border:.25em solid {c2};color:{btc};">{code}</div></div>'
        f'<div class="uavatar" style="background:radial-gradient(circle at 35% 30%,{c1} 0%,{c2} 75%,#000 120%);">'
        f'<span>{_initials(p["name"])}</span></div>'
        f'<div class="uname">{p["name"]}</div>'
        f'<div class="ustats">{cells}</div>'
        f'<div class="utier">★ {tier} ★</div>'
        f'</div></div></div>')

def tie_card(label: str, H: str, V: str, p_h: float, neutral: bool = False) -> str:
    th, tv = get_team(H), get_team(V)
    fav_h = p_h >= 0.5
    col_h = "#facc15" if abs(p_h - .5) < .05 else ("#4ade80" if fav_h else "#f87171")
    col_v = "#facc15" if abs(p_h - .5) < .05 else ("#4ade80" if not fav_h else "#f87171")
    if neutral:
        lh, lv = xg_pair(H, V, neutral=True); s = top_scores(lh, lv, 1)[0]
        sub = f'Neutral venue · pred {s[0]}–{s[1]} ({s[2]:.1f}%)'
    else:
        v1, h1 = xg_pair(V, H); h2, v2 = xg_pair(H, V)
        a = top_scores(v1, h1, 1)[0]; b = top_scores(h2, v2, 1)[0]
        sub = f'Leg 1 @ {tv["code"]} {a[1]}–{a[0]} · Leg 2 @ {th["code"]} {b[0]}–{b[1]}'
    row = lambda t, n, col, p, crown: (
        f'<div style="display:flex;align-items:center;gap:6px;padding:2px 3px;">'
        f'<span style="width:22px;height:14px;border-radius:3px;background:{t["c1"]};border:2px solid {t["c2"]};'
        f'display:inline-block;"></span>'
        f'<span style="flex:1;font-size:12px;font-weight:600;color:#dfe5ff;">{n}{crown}</span>'
        f'<span style="font-size:12px;font-weight:800;color:{col};">{p*100:.1f}%</span></div>')
    return (f'<div style="border:1px solid #1a2554;border-radius:6px;padding:7px 9px;background:#0c1430;margin-bottom:6px;">'
            f'<div style="font-size:9px;color:#7d8bbd;margin-bottom:3px;">{label}</div>'
            + row(th, H, col_h, p_h, " ▲" if fav_h else "")
            + f'<div style="display:flex;height:3px;border-radius:2px;overflow:hidden;margin:2px 0;">'
              f'<div style="width:{p_h*100:.0f}%;background:{col_h};"></div>'
              f'<div style="width:{(1-p_h)*100:.0f}%;background:{col_v};"></div></div>'
            + row(tv, V, col_v, 1 - p_h, " ▲" if not fav_h else "")
            + f'<div style="font-size:9px;color:#56628f;margin-top:2px;">{sub}</div></div>')

# ══════════════════════════════════════════════════════════════════════
# RUNTIME STATE
# ══════════════════════════════════════════════════════════════════════
TEAMS = {k: dict(v) for k, v in BASE_TEAMS.items()}
TEAM_BOOST: dict = {}
HOME_ADV = 0.18

def cards_from_fallback() -> list:
    out = []
    for r in PLAYER_CARDS_FALLBACK:
        name, club, nat, pos, ovr = r[:5]
        out.append({"name": name, "team": club, "nat": nat, "pos": pos, "overall": ovr,
                    "stats": tuple(r[5]) if len(r) > 5 else derive_stats(name, pos, ovr)})
    return out

def fmt_club(n: str) -> str:
    return f'{get_team(n)["flag"]} {n}'

# ══════════════════════════════════════════════════════════════════════
# UI — Header
# ══════════════════════════════════════════════════════════════════════
st.title("⭐ UEFA Champions League Predictor")
st.caption("36-club league phase · knockout play-offs · two-legged ties · Poisson model · Monte Carlo · Live API")

ratings_source = "Built-in (Elo-derived)"
squads_source  = "Built-in (from player cards)"
cards_source   = "Built-in"
api_log: list  = []
fix_debug = scorer_debug = ""
live_fixtures: list = []
api_cards: list = []
api_scorers: dict = {}
n_matches_used = 0

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    api_key = st.text_input("API-Football key", type="password", help="api-football.com key for live data")
    use_api = st.checkbox("Enable live API", value=False)
    season = st.selectbox("API season", [2026, 2025], format_func=lambda s: f"{s}/{str(s+1)[2:]}",
                          help="Season used for live fixtures, results, scorers and players")

    if use_api and api_key:
        with st.spinner("Connecting…"):
            status = fetch_status(api_key)
        if not status:
            st.warning("⚠️ Connection failed.")
            api_log.append(("❌", "API connection", "Returned None — check key"))
        else:
            plan, used, limit = parse_account_info(status)
            api_log.append(("✅", "Connected", f"Plan: {plan} · Reqs: {used}/{limit}"))
            st.success(f"✅ {plan} · {used}/{limit} reqs")

            with st.spinner("Ratings…"):
                res_data, rat_debug = fetch_recent_results(api_key, season)
            if res_data:
                TEAMS = compute_ratings(res_data, TEAMS)
                n_matches_used = len(res_data)
                ratings_source = f"Live ({n_matches_used} UCL matches)"
                api_log.append(("✅", "Ratings", f"{n_matches_used} matches\n" + "\n".join(rat_debug)))
            else:
                api_log.append(("⚠️", "Ratings", "0 matches\n" + "\n".join(rat_debug)))

            with st.spinner("Fixtures…"):
                live_fixtures, fix_debug = fetch_ucl_fixtures(api_key, season)
            api_log.append(("✅" if live_fixtures else "⚠️", "Fixtures", fix_debug or "none"))

            with st.spinner("Scorers…"):
                api_scorers, scorer_debug = fetch_top_scorers(api_key, season)
            api_log.append(("✅" if api_scorers else "⚠️", "Scorers", scorer_debug))

            with st.spinner("Players…"):
                api_cards = fetch_player_cards_api(api_key, season)
            api_log.append(("✅" if api_cards else "⚠️", "Player cards", f"{len(api_cards)} players"))
    else:
        st.info("🔵 Add key for live data")
        api_log.append(("🔵", "Offline", "Using built-in data"))

    st.divider()
    st.markdown("**🏟️ Home advantage**")
    HOME_ADV = st.slider("Home xG boost", 0.0, 0.5, 0.18, 0.01, key="home_adv",
                         help="Multiplier on the home side's expected goals (final is neutral)")

    st.divider()
    st.markdown("**⚡ Club boost**")
    boost_team = st.selectbox("Club", ["(none)"] + sorted(TEAMS.keys()), key="boost_team")
    boost_val = st.slider("Attack ×", 0.5, 2.0, 1.0, 0.05, key="boost_val")
    if boost_team != "(none)" and boost_val != 1.0:
        TEAM_BOOST[boost_team] = boost_val
        st.caption(f"{fmt_club(boost_team)} ×{boost_val:.2f}")

# ── Field + league-phase fixtures ─────────────────────────────────────
lp_live = [m for m in live_fixtures if "league" in m["round"].lower() and m["home"] and m["away"]]
lp_teams = {m["home"] for m in lp_live} | {m["away"] for m in lp_live}
if len(lp_live) == 144 and len(lp_teams) == 36:
    FIELD = sorted(lp_teams)
    FIXTURES = [(m["home"], m["away"], m["gh"], m["ga"], m["status"], m["date"]) for m in lp_live]
    fixture_source = f"Live API ({season}/{str(season+1)[2:]})"
else:
    FIELD = list(BASE_TEAMS)
    draw = generate_draw(tuple((n, BASE_TEAMS[n]["pot"], BASE_TEAMS[n]["assoc"]) for n in FIELD))
    FIXTURES = [(h, a, None, None, "NS", "") for h, a in draw]
    fixture_source = "Simulated draw (built-in field)"
    if use_api and api_key and live_fixtures:
        api_log.append(("⚠️", "League phase", f"API returned {len(lp_live)} league-phase fixtures / "
                        f"{len(lp_teams)} clubs — need 144 / 36, using simulated draw"))
for n in FIELD:
    if n not in TEAMS: TEAMS[n] = unknown_club(n)
IDX = {n: i for i, n in enumerate(FIELD)}

CARDS = cards_from_fallback()
if api_cards:
    CARDS = api_cards
    cards_source = f"Live API ({len(api_cards)} players)"
PEN = {n: pen_strength(n, CARDS) for n in TEAMS}
SQUADS = build_squads(CARDS)
CURRENT_GOALS: dict = {}
def _surname(n: str) -> str:
    s = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
    return (re.findall(r"[a-z]+", s) or [""])[-1]

if api_scorers:
    for club, lst in api_scorers.items():
        squad = SQUADS.setdefault(club, [])
        by_surname = {_surname(nm): nm for nm, _ in squad}
        for nm, g in lst:
            match = by_surname.get(_surname(nm))
            if match:
                CURRENT_GOALS[match] = g
            else:
                squad.append((nm, 0.05))
                CURRENT_GOALS[nm] = g
    squads_source = f"Built-in shares + live UCL goals ({len(api_scorers)} clubs)"

def build_spec() -> tuple:
    played = [(gh is not None and ga is not None and stt in FIN_STATUS) for _, _, gh, ga, stt, _ in FIXTURES]
    spec = {
        "names": FIELD,
        "atk": [round(TEAMS[n]["atk"] * TEAM_BOOST.get(n, 1.0), 4) for n in FIELD],
        "def": [round(TEAMS[n]["def"], 4) for n in FIELD],
        "pen": [round(PEN.get(n, .5), 3) for n in FIELD],
        "ha": float(HOME_ADV),
        "fh": [IDX[f[0]] for f in FIXTURES], "fa": [IDX[f[1]] for f in FIXTURES],
        "gh0": [f[2] if p else -1 for f, p in zip(FIXTURES, played)],
        "ga0": [f[3] if p else -1 for f, p in zip(FIXTURES, played)],
        "players": [(nm, IDX[c], sh, int(CURRENT_GOALS.get(nm, 0)))
                    for c in FIELD for nm, sh in SQUADS.get(c, [])],
    }
    return spec, hashlib.md5(repr(spec).encode()).hexdigest()

SPEC, SPEC_KEY = build_spec()
N_PLAYED = sum(1 for g in SPEC["gh0"] if g >= 0)

def get_sim() -> tuple:
    s = st.session_state.get("sim")
    if s and s["key"] == SPEC_KEY:
        return s["res"], False
    return run_sims(5000, 1, SPEC_KEY, SPEC), True

top_elo = max(FIELD, key=lambda k: TEAMS[k]["elo"])
st.markdown(f"""<div class="dbar">
<div><div class="dl">Clubs</div><div class="dv">{len(FIELD)}</div></div>
<div><div class="dl">League matches</div><div class="dv">{len(FIXTURES)} · {N_PLAYED} played</div></div>
<div><div class="dl">Fixtures</div><div class="dv">{fixture_source}</div></div>
<div><div class="dl">Player cards</div><div class="dv">{len(CARDS)}</div></div>
<div><div class="dl">Top Elo</div><div class="dv">{fmt_club(top_elo)}</div></div>
<div><div class="dl">Format</div><div class="dv">8 games · top 8 → R16</div></div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.divider()
    n_boost = sum(1 for v in TEAM_BOOST.values() if v != 1.0)
    for line in [f"Ratings: {ratings_source}", f"Fixtures: {fixture_source}", f"Scorers: {squads_source}",
                 f"Cards: {cards_source}", f"Boosts active: {n_boost} · Home adv: +{HOME_ADV*100:.0f}%"]:
        st.caption(line)

# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab_sim, tab_lp, tab_ko, tab_ts, tab_cards, tab_draft, tab_h2h, tab_model, tab_api = st.tabs([
    "📊 Simulation", "📋 League Phase", "🗓️ Knockouts", "👟 Top Scorer",
    "🃏 Cards", "🎲 Draft", "⚔️ H2H", "🔬 Model", "🔌 API",
])
PCT = lambda label, mx=100.0: st.column_config.ProgressColumn(label, format="%.1f%%", min_value=0.0, max_value=mx)

# ─────────────────────────────────────────────────────────────────────
# TAB 1 — Simulation
# ─────────────────────────────────────────────────────────────────────
with tab_sim:
    st.subheader("Monte Carlo season simulation")
    c1, c2 = st.columns(2)
    with c1: n_sims = st.slider("Simulations", 1_000, 200_000, 20_000, 1_000)
    with c2: seed = st.number_input("Seed (0=random)", value=0, step=1)
    if st.button("▶ Run", type="primary", width="stretch", key="run_sim"):
        s = int(seed) if int(seed) != 0 else random.randint(1, 10_000_000)
        with st.spinner(f"Simulating {n_sims:,} seasons…"):
            st.session_state["sim"] = {"key": SPEC_KEY, "res": run_sims(n_sims, s, SPEC_KEY, SPEC)}

    R, preview = get_sim()
    if preview:
        st.caption("Showing a quick 5,000-season preview — press ▶ Run for a full simulation.")
    rows = [{"Club": fmt_club(n), "Pot": TEAMS[n].get("pot") or "—", "Elo": TEAMS[n]["elo"],
             "Avg pts": R["pts"][i], "Avg pos": R["pos"][i],
             "Top 8": R["top8"][i] * 100, "Top 24": R["top24"][i] * 100, "R16": R["r16"][i] * 100,
             "QF": R["qf"][i] * 100, "SF": R["sf"][i] * 100, "Final": R["final"][i] * 100,
             "Win": R["win"][i] * 100, "_n": n} for i, n in enumerate(R["names"])]
    df = pd.DataFrame(rows).sort_values(["Win", "Final", "SF"], ascending=False).reset_index(drop=True)
    df.insert(0, "#", df.index + 1)
    top = df.iloc[0]
    gap = f" · +{top['Win'] - df.iloc[1]['Win']:.1f}pp over {df.iloc[1]['_n']}" if len(df) > 1 else ""
    st.success(f"**{top['Club']}** favourite · {top['Win']:.1f}%{gap}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Simulations", f"{R['n']:,}")
    m2.metric("Clubs w/ title", int((df["Win"] > 0).sum()))
    m3.metric("Top-8 line (pts)", f"{R['cut8']:.1f}")
    m4.metric("Top-24 line (pts)", f"{R['cut24']:.1f}")
    m5.metric("Favourite odds", f"{100/top['Win']:.1f}:1" if top["Win"] > 0 else "—")

    cl, cr = st.columns([3, 2], gap="medium")
    with cl:
        st.markdown("#### Full field — stage reached")
        st.dataframe(df.drop(columns="_n"), width="stretch", hide_index=True, height=520,
                     column_config={"Avg pts": st.column_config.NumberColumn(format="%.1f"),
                                    "Avg pos": st.column_config.NumberColumn(format="%.1f"),
                                    **{c: st.column_config.NumberColumn(format="%.1f%%")
                                       for c in ("Top 8", "Top 24", "R16", "QF", "SF", "Final")},
                                    "Win": PCT("Win", max(1.0, float(df["Win"].max())))})
    with cr:
        st.markdown("#### Top 12 — title %")
        st.bar_chart(df.head(12).set_index("_n")[["Win"]], horizontal=True, sort="-Win")
        st.markdown("#### By association")
        df["A"] = df["_n"].map(lambda n: get_team(n)["assoc"])
        st.bar_chart(df.groupby("A")["Win"].sum().sort_values(ascending=False))

# ─────────────────────────────────────────────────────────────────────
# TAB 2 — League phase
# ─────────────────────────────────────────────────────────────────────
with tab_lp:
    R, preview = get_sim()
    st.subheader("📋 League phase — projected table")
    st.caption(f"{R['n']:,} simulated seasons · fixtures: {fixture_source} · "
               "1–8 → Round of 16 · 9–24 → knockout play-offs · 25–36 eliminated")
    ph = np.array(R["pos_hist"])
    lrows = [{"Club": fmt_club(n), "Avg pts": R["pts"][i], "Avg GD": R["gd"][i], "Avg pos": R["pos"][i],
              "Likeliest pos": int(ph[i].argmax()) + 1,
              "Top 8": R["top8"][i] * 100, "Play-off": (R["top24"][i] - R["top8"][i]) * 100,
              "Out": (1 - R["top24"][i]) * 100} for i, n in enumerate(R["names"])]
    ldf = pd.DataFrame(lrows).sort_values("Avg pos").reset_index(drop=True)
    ldf.insert(0, "#", ldf.index + 1)
    st.dataframe(ldf, width="stretch", hide_index=True, height=560,
                 column_config={"Avg pts": st.column_config.NumberColumn(format="%.1f"),
                                "Avg GD": st.column_config.NumberColumn(format="%+.1f"),
                                "Avg pos": st.column_config.NumberColumn(format="%.1f"),
                                "Top 8": PCT("Top 8"), "Play-off": PCT("Play-off"), "Out": PCT("Out")})

    if N_PLAYED:
        st.divider()
        st.markdown(f"#### Current standings ({N_PLAYED} of {len(FIXTURES)} played)")
        tbl = {n: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0} for n in FIELD}
        for h, a, gh, ga, stt, _ in FIXTURES:
            if gh is None or ga is None or stt not in FIN_STATUS: continue
            for t, f, g in ((h, gh, ga), (a, ga, gh)):
                r = tbl[t]; r["P"] += 1; r["GF"] += f; r["GA"] += g
                r["W" if f > g else ("D" if f == g else "L")] += 1
        cdf = pd.DataFrame([{"Club": fmt_club(n), **v, "GD": v["GF"] - v["GA"], "Pts": 3 * v["W"] + v["D"]}
                            for n, v in tbl.items()]).sort_values(["Pts", "GD", "GF"], ascending=False)
        cdf.insert(0, "#", range(1, len(cdf) + 1))
        st.dataframe(cdf, width="stretch", hide_index=True, height=420)

    st.divider()
    st.markdown("#### Club fixtures")
    sel = st.selectbox("Club", FIELD, index=0, format_func=fmt_club, key="fx_club")
    frows = []
    for h, a, gh, ga, stt, dt in FIXTURES:
        if sel not in (h, a): continue
        home = h == sel; opp = a if home else h
        mp = match_probs(h, a)
        w, d, l = (mp["home"], mp["draw"], mp["away"]) if home else (mp["away"], mp["draw"], mp["home"])
        res = f"{gh}–{ga}" if stt in FIN_STATUS and gh is not None else ("LIVE" if stt in LIVE_STATUS else "")
        s0 = mp["scores"][0]
        frows.append({"Date": dt, "Venue": "H" if home else "A", "Opponent": fmt_club(opp),
                      "Pot": get_team(opp).get("pot") or "—", "Result": res,
                      "Win": w, "Draw": d, "Loss": l, "Pred": f"{s0[0]}–{s0[1]}" if home else f"{s0[1]}–{s0[0]}"})
    st.dataframe(pd.DataFrame(frows), width="stretch", hide_index=True,
                 column_config={"Win": PCT("Win"), "Draw": st.column_config.NumberColumn(format="%.1f%%"),
                                "Loss": st.column_config.NumberColumn(format="%.1f%%")})
    st.caption("Pred = most likely score from the club's perspective (club goals first)")

# ─────────────────────────────────────────────────────────────────────
# TAB 3 — Knockouts
# ─────────────────────────────────────────────────────────────────────
with tab_ko:
    R, preview = get_sim()
    st.subheader("🗓️ Knockout bracket — projected path")
    st.caption("Seeded by average league position · favourite advances each round · "
               "% = chance to win the tie (two legs, ET + penalties) · the real draw pairs seeds in twos at random")
    order_ = [R["names"][i] for i in np.argsort(R["pos"])]
    S_ = {k + 1: n for k, n in enumerate(order_)}

    def adv(H, V):
        p = tie_advance(H, V)
        return (H if p >= .5 else V), p

    po = {}
    po_html = []
    for s, u in ((9, 24), (10, 23), (11, 22), (12, 21), (13, 20), (14, 19), (15, 18), (16, 17)):
        w, p = adv(S_[s], S_[u]); po[s] = w
        po_html.append(tie_card(f"Play-off · {s} v {u} · 2nd leg at {get_team(S_[s])['code']}", S_[s], S_[u], p))
    st.markdown("### Knockout play-offs")
    a_, b_ = st.columns(2)
    with a_: st.markdown("".join(po_html[:4]), unsafe_allow_html=True)
    with b_: st.markdown("".join(po_html[4:]), unsafe_allow_html=True)

    r16_map = {1: po[16], 2: po[15], 3: po[14], 4: po[13], 5: po[12], 6: po[11], 7: po[10], 8: po[9]}
    r16w, r16_html = {}, {}
    for s, opp in r16_map.items():
        w, p = adv(S_[s], opp); r16w[s] = w
        r16_html[s] = tie_card(f"Round of 16 · seed {s}", S_[s], opp, p)

    def ko_card(lbl, a, b):
        pa, pb = R["pos"][R["names"].index(a)], R["pos"][R["names"].index(b)]
        H, V = (a, b) if pa <= pb else (b, a)
        w, p = adv(H, V)
        return w, tie_card(lbl, H, V, p)

    qa1, qa1h = ko_card("QF 1", r16w[1], r16w[8]); qa2, qa2h = ko_card("QF 2", r16w[4], r16w[5])
    qb1, qb1h = ko_card("QF 3", r16w[2], r16w[7]); qb2, qb2h = ko_card("QF 4", r16w[3], r16w[6])
    sa, sah = ko_card("SF 1", qa1, qa2); sb, sbh = ko_card("SF 2", qb1, qb2)
    pf = final_win(sa, sb)

    st.markdown("### Round of 16 → Final")
    k1, k2, k3, k4 = st.columns([3, 3, 3, 3])
    with k1:
        st.markdown("**Round of 16**")
        st.markdown("".join(r16_html[s] for s in (1, 8, 4, 5, 2, 7, 3, 6)), unsafe_allow_html=True)
    with k2:
        st.markdown("**Quarter-finals**")
        st.markdown(qa1h + qa2h + qb1h + qb2h, unsafe_allow_html=True)
    with k3:
        st.markdown("**Semi-finals**")
        st.markdown(sah + sbh, unsafe_allow_html=True)
    with k4:
        st.markdown("**Final**")
        st.markdown(tie_card("Final · single match", sa, sb, pf, neutral=True), unsafe_allow_html=True)
        champ = sa if pf >= .5 else sb
        st.success(f"🏆 Projected winner: **{fmt_club(champ)}**")
        st.caption(f"Overall title chance from simulation: {R['win'][R['names'].index(champ)]*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────
# TAB 4 — Top scorer
# ─────────────────────────────────────────────────────────────────────
with tab_ts:
    R, preview = get_sim()
    st.subheader("👟 Top scorer — predicted")
    if not R["players"]:
        st.warning("No player data.")
    else:
        trows = [{"Player": p, "Club": fmt_club(c), "Goals so far": now_, "Exp. goals": g, "Top scorer": w * 100}
                 for p, c, now_, g, w in zip(R["players"], R["player_club"], R["player_now"], R["ts_goals"], R["ts_win"])]
        tdf = pd.DataFrame(trows).sort_values(["Top scorer", "Exp. goals"], ascending=False).head(25).reset_index(drop=True)
        tdf.insert(0, "#", tdf.index + 1)
        if not N_PLAYED and not CURRENT_GOALS: tdf = tdf.drop(columns="Goals so far")
        t = tdf.iloc[0]
        st.success(f"**{t['Player']}** ({t['Club']}) — top scorer {t['Top scorer']:.1f}% · {t['Exp. goals']:.1f} exp. goals")
        cl, cr = st.columns([3, 2], gap="large")
        with cl:
            st.dataframe(tdf, width="stretch", hide_index=True, height=600,
                         column_config={"Exp. goals": st.column_config.NumberColumn(format="%.2f"),
                                        "Top scorer": PCT("Top scorer", max(1.0, float(tdf["Top scorer"].max())))})
        with cr:
            st.markdown("#### Top 10")
            st.bar_chart(tdf.head(10).set_index("Player")[["Top scorer"]], horizontal=True, sort="-Top scorer")
        st.caption(f"{R['n']:,} seasons · club goals split by player goal share · ties for top spot split evenly")

# ─────────────────────────────────────────────────────────────────────
# TAB 5 — Player cards
# ─────────────────────────────────────────────────────────────────────
with tab_cards:
    st.subheader("🃏 Champions League Player Cards")
    st.caption(f"Source: {cards_source} · Icon 90+ (holo) · Gold 86–89 · Silver 82–85 · Bronze <82 · hover a card")
    cards = sorted(CARDS, key=lambda x: -x["overall"])
    f1, f2, f3, f4 = st.columns([3, 2, 2, 2])
    with f1: ft = st.selectbox("Club", ["All"] + sorted({c["team"] for c in cards}), key="ct")
    with f2: fp = st.selectbox("Pos", ["All"] + sorted({c["pos"] for c in cards}), key="cp")
    with f3: fo = st.slider("Min OVR", 60, 95, 60, 1, key="co")
    with f4: fc = st.selectbox("Sort", ["Overall ↓", "Overall ↑", "Name", "Club"], key="cs")
    fil = [c for c in cards if (ft == "All" or c["team"] == ft) and (fp == "All" or c["pos"] == fp) and c["overall"] >= fo]
    if fc == "Overall ↑": fil.sort(key=lambda x: x["overall"])
    elif fc == "Name":    fil.sort(key=lambda x: x["name"])
    elif fc == "Club":    fil.sort(key=lambda x: (x["team"], -x["overall"]))
    st.caption(f"**{len(fil)}** players · {sum(c['overall'] >= 90 for c in fil)} icon · "
               f"{sum(86 <= c['overall'] < 90 for c in fil)} gold · {sum(82 <= c['overall'] < 86 for c in fil)} silver · "
               f"{sum(c['overall'] < 82 for c in fil)} bronze")
    CPR = 6
    for i in range(0, len(fil), CPR):
        cols = st.columns(CPR)
        for col, card in zip(cols, fil[i:i + CPR]):
            with col:
                st.markdown(render_card(card), unsafe_allow_html=True)
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if not fil: st.info("No players match filters.")

# ─────────────────────────────────────────────────────────────────────
# TAB — Dice draft
# ─────────────────────────────────────────────────────────────────────
FORMATIONS = {
    "4-3-3":   [["LW", "ST", "RW"], ["CM", "CM", "CM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "4-2-3-1": [["ST"], ["LW", "CAM", "RW"], ["CDM", "CDM"], ["LB", "CB", "CB", "RB"], ["GK"]],
    "3-5-2":   [["ST", "ST"], ["LWB", "CM", "CAM", "CM", "RWB"], ["CB", "CB", "CB"], ["GK"]],
    "4-4-2":   [["ST", "ST"], ["LW", "CM", "CM", "RW"], ["LB", "CB", "CB", "RB"], ["GK"]],
}
SLOT_POOL = {"GK": ["GK"], "LB": ["LB", "LWB"], "RB": ["RB", "RWB"], "LWB": ["LWB", "LB"], "RWB": ["RWB", "RB"],
             "CB": ["CB"], "CM": ["CM", "CDM", "CAM"], "CDM": ["CDM", "CM"], "CAM": ["CAM", "CM"],
             "LW": ["LW", "RW"], "RW": ["RW", "LW"], "ST": ["ST"]}
DICE_BANDS = {1: (60, 79), 2: (77, 81), 3: (80, 84), 4: (83, 86), 5: (86, 88), 6: (89, 99)}
DICE_LABEL = {1: "Bronze pull", 2: "Bronze+ pull", 3: "Silver pull", 4: "Silver+ pull", 5: "Gold pull", 6: "ICON pull!"}
ATT_W = {"ST": 1.0, "LW": 1.0, "RW": 1.0, "CAM": 1.0, "CM": 0.6, "CDM": 0.25, "LWB": 0.35, "RWB": 0.35,
         "LB": 0.2, "RB": 0.2, "CB": 0.05, "GK": 0.0}
DEF_W = {"GK": 2.0, "CB": 1.0, "LB": 1.0, "RB": 1.0, "LWB": 0.8, "RWB": 0.8, "CDM": 1.0, "CM": 0.4,
         "CAM": 0.15, "LW": 0.1, "RW": 0.1, "ST": 0.05}
STARTING_REROLLS = 5
PIPS = {1: [4], 2: [0, 8], 3: [0, 4, 8], 4: [0, 2, 6, 8], 5: [0, 2, 4, 6, 8], 6: [0, 2, 3, 5, 6, 8]}

def formation_slots(form: str) -> list:
    return [pos for line in FORMATIONS[form] for pos in line]

def dice_html(v: int, nonce: int) -> str:
    bg, pip = ("linear-gradient(145deg,#fff3c4,#f3c64e)", "#5b2ca0") if v == 6 else ("linear-gradient(145deg,#ffffff,#d9def0)", "#0b1440")
    cells = "".join(f'<div class="pip" style="background:{pip if i in PIPS[v] else "transparent"};"></div>' for i in range(9))
    return (f'<div data-n="{nonce}" style="text-align:center;"><div class="dice{" six" if v == 6 else ""}" style="background:{bg};">{cells}</div>'
            f'<div style="font-size:12px;font-weight:800;margin:4px 0 14px;color:{"#ffe08a" if v >= 5 else "#c8d1f5"};">Rolled {v} · {DICE_LABEL[v]}</div></div>')

def draw_offers(slot: str, roll: int, taken: set, k: int = 3) -> list:
    pool = [c for c in CARDS if c["pos"] in SLOT_POOL[slot] and c["name"] not in taken]
    lo, hi = DICE_BANDS[roll]
    cands = [c for c in pool if lo <= c["overall"] <= hi]
    while len(cands) < k and (lo > 40 or hi < 99):
        lo -= 1; hi += 1
        cands = [c for c in pool if lo <= c["overall"] <= hi]
    return random.sample(cands, min(k, len(cands)))

def xi_ratings(xi: list) -> dict:
    """xi: list of (slot, card). Returns unit ratings, chemistry and model atk/def."""
    def wavg(W):
        tot = sum(W.get(s, 0) for s, _ in xi)
        return sum(W.get(s, 0) * c["overall"] for s, c in xi) / tot if tot else 0
    att, dfn = wavg(ATT_W), wavg(DEF_W)
    chem = 0.0
    for i in range(len(xi)):
        for j in range(i + 1, len(xi)):
            a, b = xi[i][1], xi[j][1]
            if a["team"] == b["team"]: chem += 0.3
            if a.get("nat") and a.get("nat") == b.get("nat") and a["nat"] != "🏳️": chem += 0.15
    chem = min(3.0, chem)
    elo_of = lambda ovr: float(np.clip(-766 + 31.9 * ovr, 1450, 2250))
    a_elo, d_elo = elo_of(att + chem), elo_of(dfn + chem)
    gk = max([c["overall"] for s, c in xi if s == "GK"] or [75])
    return {"ovr": float(np.mean([c["overall"] for _, c in xi])) if xi else 0, "att": att, "def": dfn, "chem": chem,
            "elo": int((a_elo + d_elo) / 2), "atk_r": ratings_from_elo(a_elo)[0],
            "def_r": max(0.45, ratings_from_elo(d_elo)[1]), "pen": float(np.clip(0.5 + (gk - 80) * 0.01, 0.4, 0.6))}

def pitch_html(form: str, xi: dict, active: int) -> str:
    rows, idx = [], 0
    for line in FORMATIONS[form]:
        cells = []
        for pos in line:
            c = xi.get(idx)
            if c:
                cells.append(f'<div class="pslot">{render_card(c, "sm")}</div>')
            else:
                cells.append(f'<div class="pslot"><div class="pempty{" active" if idx == active else ""}">'
                             f'<div style="font-size:22px;">🎲</div>{pos}</div></div>')
            idx += 1
        rows.append(f'<div class="prow">{"".join(cells)}</div>')
    return f'<div class="pitch">{"".join(rows)}</div>'

def _draft() -> dict:
    if "draft" not in st.session_state:
        st.session_state["draft"] = {"form": "4-3-3", "xi": {}, "offers": None, "roll": None,
                                     "rerolls": STARTING_REROLLS, "nonce": 0}
    return st.session_state["draft"]

def _next_empty(D: dict, start: int = 0) -> int:
    n = len(formation_slots(D["form"]))
    for k in range(n):
        i = (start + k) % n
        if i not in D["xi"]: return i
    return start

def _slot_label(form: str, k: int) -> str:
    return f"{k+1} · {formation_slots(form)[k]}"

def _active_idx() -> int:
    D = _draft(); n = len(formation_slots(D["form"]))
    try: k = int(str(st.session_state.get("draft_slot", "1")).split("·")[0].strip()) - 1
    except ValueError: k = 0
    return k if 0 <= k < n else 0

def _set_active(D: dict, k: int):
    st.session_state["draft_slot"] = _slot_label(D["form"], k)

def _cb_reset():
    st.session_state.pop("draft", None); st.session_state.pop("draft_res", None)
    st.session_state["draft_slot"] = _slot_label("4-3-3", 0)

def _cb_form():
    D = _draft(); D.update({"form": st.session_state["draft_form"], "xi": {}, "offers": None, "roll": None,
                            "rerolls": STARTING_REROLLS})
    _set_active(D, 0); st.session_state.pop("draft_res", None)

def _cb_roll():
    D = _draft(); i = _active_idx(); slot = formation_slots(D["form"])[i]
    if i in D["xi"]:
        if D["rerolls"] <= 0: return
        D["rerolls"] -= 1
    taken = {c["name"] for k, c in D["xi"].items() if k != i}
    r = random.randint(1, 6); D["nonce"] += 1
    D["roll"] = (i, r, D["nonce"]); D["offers"] = (i, draw_offers(slot, r, taken))

def _cb_pick(j: int):
    D = _draft(); i, offers = D["offers"]
    D["xi"][i] = offers[j]; D["offers"] = None
    _set_active(D, _next_empty(D, i))
    st.session_state.pop("draft_res", None)

def _cb_auto():
    D = _draft()
    for i, slot in enumerate(formation_slots(D["form"])):
        if i in D["xi"]: continue
        taken = {c["name"] for c in D["xi"].values()}
        offers = draw_offers(slot, random.randint(1, 6), taken)
        if offers: D["xi"][i] = max(offers, key=lambda c: c["overall"])
    D["offers"] = None; D["roll"] = None
    st.session_state.pop("draft_res", None)

def xi_spec(club_out: str, name: str, xi_cards: list, rt: dict) -> tuple:
    i = IDX[club_out]
    spec = {k: (list(v) if isinstance(v, list) else v) for k, v in SPEC.items()}
    spec["names"][i] = name
    spec["atk"][i] = round(rt["atk_r"], 4); spec["def"][i] = round(rt["def_r"], 4); spec["pen"][i] = round(rt["pen"], 3)
    for k, (h, a) in enumerate(zip(spec["fh"], spec["fa"])):
        if i in (h, a): spec["gh0"][k] = -1; spec["ga0"][k] = -1
    xi_names = {c["name"] for c in xi_cards}
    mine = build_squads([{**c, "team": name} for c in xi_cards]).get(name, [])
    spec["players"] = [p for p in SPEC["players"] if p[1] != i and p[0] not in xi_names] + [(n, i, s, 0) for n, s in mine]
    return spec, hashlib.md5(repr(spec).encode()).hexdigest()

with tab_draft:
    D = _draft()
    slots = formation_slots(D["form"])
    labels = [_slot_label(D["form"], k) for k in range(len(slots))]
    if st.session_state.get("draft_slot") not in labels:
        st.session_state["draft_slot"] = labels[0]
    st.subheader("🎲 Dice Draft — build your Champions League XI")
    st.caption("Roll the die for each position: 1 = bronze pull … 6 = ICON pull. Pick 1 of 3 players. "
               f"Re-rolling a filled slot costs a token ({STARTING_REROLLS} per draft). Then send your XI into the UCL.")
    h1, h2, h3, h4 = st.columns([2, 2, 2, 2])
    with h1: st.selectbox("Formation", list(FORMATIONS), index=list(FORMATIONS).index(D["form"]),
                          key="draft_form", on_change=_cb_form)
    with h2: st.metric("Re-roll tokens", D["rerolls"])
    with h3: st.button("⚡ Auto-draft empty slots", on_click=_cb_auto, width="stretch", key="draft_auto")
    with h4: st.button("♻️ New draft", on_click=_cb_reset, width="stretch", key="draft_reset")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(pitch_html(D["form"], D["xi"], _active_idx()), unsafe_allow_html=True)
    with right:
        st.selectbox("Position", labels, key="draft_slot")
        act = _active_idx()
        filled = act in D["xi"]
        lbl = f"🎲 Re-roll {slots[act]} ({D['rerolls']} left)" if filled else f"🎲 Roll for {slots[act]}"
        st.button(lbl, type="primary", width="stretch", key="draft_roll", on_click=_cb_roll,
                  disabled=filled and D["rerolls"] <= 0)
        if D["roll"] and D["roll"][0] == act:
            st.markdown(dice_html(D["roll"][1], D["roll"][2]), unsafe_allow_html=True)
        if D["offers"] and D["offers"][0] == act:
            offers = D["offers"][1]
            if not offers:
                st.warning("No players left for this position.")
            oc = st.columns(3)
            for j, c in enumerate(offers):
                with oc[j]:
                    st.markdown(render_card(c, badge="NEW" if c["overall"] >= 89 else ""), unsafe_allow_html=True)
                    st.button("Pick", key=f"draft_pick_{j}", on_click=_cb_pick, args=(j,), width="stretch")
        elif filled:
            st.caption("Current pick:")
            st.markdown(render_card(D["xi"][act]), unsafe_allow_html=True)

    xi_pairs = [(slots[k], D["xi"][k]) for k in sorted(D["xi"]) if k < len(slots)]
    st.divider()
    if xi_pairs:
        rt = xi_ratings(xi_pairs)
        rank = 1 + sum(1 for n in FIELD if TEAMS[n]["elo"] > rt["elo"])
        m = st.columns(6)
        m[0].metric("Players", f"{len(xi_pairs)}/11"); m[1].metric("Team OVR", f"{rt['ovr']:.1f}")
        m[2].metric("Attack", f"{rt['att']:.1f}"); m[3].metric("Defence", f"{rt['def']:.1f}")
        m[4].metric("Chemistry", f"+{rt['chem']:.1f}"); m[5].metric("Elo equiv.", f"{rt['elo']:,}", f"#{rank} of {len(FIELD)}")
        st.caption("Chemistry: +0.3 per pair of club-mates, +0.15 per pair of compatriots (max +3 to attack and defence)")

    if len(xi_pairs) < len(slots):
        st.info(f"Fill all {len(slots)} positions to simulate your XI in the Champions League.")
    else:
        st.markdown("### 🏆 Send your XI into the Champions League")
        s1, s2, s3 = st.columns([3, 3, 2])
        weakest = min(FIELD, key=lambda n: TEAMS[n]["elo"])
        with s1: team_name = st.text_input("Team name", "Dream XI", key="draft_name").strip() or "Dream XI"
        with s2: club_out = st.selectbox("Takes the place (and fixtures) of", FIELD, index=FIELD.index(weakest),
                                         format_func=fmt_club, key="draft_out")
        with s3: dn = st.select_slider("Simulations", [2_000, 5_000, 10_000, 20_000, 50_000], 10_000, key="draft_n")
        if team_name in TEAMS and team_name != club_out: team_name += " XI"
        if st.button("▶ Simulate my XI", type="primary", width="stretch", key="draft_go"):
            spec_x, key_x = xi_spec(club_out, team_name, [c for _, c in xi_pairs], rt)
            with st.spinner(f"Simulating {dn:,} seasons with {team_name}…"):
                st.session_state["draft_res"] = {"res": run_sims(dn, random.randint(1, 10_000_000), key_x, spec_x),
                                                 "name": team_name, "out": club_out, "rt": rt}
        DR = st.session_state.get("draft_res")
        if DR:
            R2, nm = DR["res"], DR["name"]; i = R2["names"].index(nm)
            win_rank = 1 + sum(1 for w in R2["win"] if w > R2["win"][i])
            verdict = ("🏆 Title favourites!" if win_rank == 1 else "🔥 Genuine contender" if R2["win"][i] > .05
                       else "💪 Knockout regular" if R2["r16"][i] > .6 else "😬 Group-stage struggle" if R2["top24"][i] < .5
                       else "🎟️ Play-off battlers")
            st.success(f"**{nm}** — {verdict} · title {R2['win'][i]*100:.1f}% (#{win_rank} of {len(R2['names'])}) · "
                       f"avg {R2['pts'][i]:.1f} pts, finish {R2['pos'][i]:.1f}")
            mm = st.columns(7)
            for col, (lbl_, k) in zip(mm, [("Top 8", "top8"), ("Top 24", "top24"), ("R16", "r16"), ("QF", "qf"),
                                          ("SF", "sf"), ("Final", "final"), ("Win", "win")]):
                col.metric(lbl_, f"{R2[k][i]*100:.1f}%")
            c_l, c_r = st.columns([3, 2], gap="medium")
            with c_l:
                st.markdown(f"#### {nm}'s league-phase fixtures")
                rows_ = []
                for h, a, _, _, _, dt in FIXTURES:
                    if DR["out"] not in (h, a): continue
                    home = h == DR["out"]; opp = a if home else h; to = get_team(opp)
                    me_x = DR["rt"]["atk_r"] * to["def"] * ((1 + HOME_ADV) if home else 1)
                    op_x = to["atk"] * TEAM_BOOST.get(opp, 1.0) * DR["rt"]["def_r"] * (1 if home else (1 + HOME_ADV))
                    w, d, l = outcome_probs(me_x, op_x); s0 = top_scores(me_x, op_x, 1)[0]
                    rows_.append({"Venue": "H" if home else "A", "Opponent": fmt_club(opp), "Win": w * 100,
                                  "Draw": d * 100, "Loss": l * 100, "Pred": f"{s0[0]}–{s0[1]}",
                                  "xG": f"{me_x:.2f} – {op_x:.2f}"})
                st.dataframe(pd.DataFrame(rows_), width="stretch", hide_index=True,
                             column_config={"Win": PCT("Win"), "Draw": st.column_config.NumberColumn(format="%.1f%%"),
                                            "Loss": st.column_config.NumberColumn(format="%.1f%%")})
            with c_r:
                st.markdown(f"#### {nm} scorers")
                sc = [{"Player": p, "Exp. goals": g, "Top scorer": w * 100}
                      for p, c, g, w in zip(R2["players"], R2["player_club"], R2["ts_goals"], R2["ts_win"]) if c == nm]
                st.dataframe(pd.DataFrame(sc).sort_values("Exp. goals", ascending=False), width="stretch", hide_index=True,
                             column_config={"Exp. goals": st.column_config.NumberColumn(format="%.2f"),
                                            "Top scorer": st.column_config.NumberColumn(format="%.1f%%")})
                rivals = sorted(range(len(R2["names"])), key=lambda k: -R2["win"][k])[:5]
                st.markdown("#### Title race")
                st.dataframe(pd.DataFrame([{"Club": (f"⭐ {R2['names'][k]}" if k == i else fmt_club(R2["names"][k])),
                                            "Win": R2["win"][k] * 100} for k in sorted(set(rivals) | {i}, key=lambda k: -R2["win"][k])]),
                             width="stretch", hide_index=True, column_config={"Win": PCT("Win", max(1.0, max(R2["win"]) * 100))})
            st.caption(f"{R2['n']:,} seasons · {nm} replaces {fmt_club(DR['out'])} and plays its 8 league fixtures")

# ─────────────────────────────────────────────────────────────────────
# TAB 6 — Head-to-head
# ─────────────────────────────────────────────────────────────────────
with tab_h2h:
    st.subheader("⚔️ Head-to-head")
    tns = sorted(TEAMS.keys())
    ca, cvs, cb = st.columns([5, 1, 5])
    with ca: ta = st.selectbox("Club A", tns, index=tns.index("Real Madrid") if "Real Madrid" in tns else 0,
                               format_func=fmt_club, key="h2h_a")
    with cvs: st.markdown("<br><br>**vs**", unsafe_allow_html=True)
    with cb: tb = st.selectbox("Club B", tns, index=tns.index("Arsenal") if "Arsenal" in tns else 1,
                               format_func=fmt_club, key="h2h_b")
    mode = st.radio("Format", ["A at home", "B at home", "Final (neutral)", "Two legs — A hosts 2nd leg",
                               "Two legs — B hosts 2nd leg"], horizontal=True, key="h2h_mode")

    if st.button("▶ Calculate", type="primary", width="stretch", key="h2h_go"):
        if ta == tb:
            st.warning("Pick different clubs.")
        else:
            fa, fb = get_team(ta)["code"], get_team(tb)["code"]
            pa_pen = pen_share(ta, tb)
            ea, eb = get_team(ta)["elo"], get_team(tb)["elo"]
            if mode.startswith("Two legs"):
                H, V = (ta, tb) if "A hosts" in mode else (tb, ta)
                p = tie_advance(H, V); pA = p if H == ta else 1 - p
                v1, h1 = xg_pair(V, H); h2, v2 = xg_pair(H, V)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"{fa} advance", f"{pA*100:.1f}%"); c2.metric(f"{fb} advance", f"{(1-pA)*100:.1f}%")
                c3.metric(f"{fa} pens", f"{pa_pen*100:.0f}%"); c4.metric(f"{fb} pens", f"{(1-pa_pen)*100:.0f}%")
                st.divider()
                l1, l2 = st.columns(2)
                for col, (hh, aa, lh, la, lbl) in zip((l1, l2), ((V, H, v1, h1, "Leg 1"), (H, V, h2, v2, "Leg 2"))):
                    with col:
                        w, d, l = outcome_probs(lh, la); s = top_scores(lh, la)
                        st.markdown(f"#### {lbl}: {fmt_club(hh)} v {fmt_club(aa)}")
                        st.caption(f"xG {lh:.2f} – {la:.2f} · home {w*100:.1f}% · draw {d*100:.1f}% · away {l*100:.1f}%")
                        st.dataframe(pd.DataFrame([{"Score": f"{hh} {i}–{j} {aa}", "Prob": f"{q:.2f}%"} for i, j, q in s]),
                                     width="stretch", hide_index=True)
            else:
                neutral = mode.startswith("Final")
                home, away = (tb, ta) if mode == "B at home" else (ta, tb)
                mp = match_probs(home, away, neutral)
                pA, dr, pB = (mp["home"], mp["draw"], mp["away"]) if home == ta else (mp["away"], mp["draw"], mp["home"])
                xa_, xb_ = (mp["xh"], mp["xa"]) if home == ta else (mp["xa"], mp["xh"])
                c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                c1.metric(f"{fa} win 90'", f"{pA:.1f}%"); c2.metric("Draw 90'", f"{dr:.1f}%")
                c3.metric(f"{fb} win 90'", f"{pB:.1f}%"); c4.metric(f"{fa} xG", f"{xa_:.2f}")
                c5.metric(f"{fb} xG", f"{xb_:.2f}"); c6.metric(f"{fa} pens", f"{pa_pen*100:.0f}%")
                c7.metric(f"{fb} pens", f"{(1-pa_pen)*100:.0f}%")
                st.divider()
                col_sc, col_elo = st.columns([3, 2], gap="medium")
                with col_sc:
                    s0 = mp["scores"][0]
                    st.markdown(f"#### Predicted: **{home} {s0[0]}–{s0[1]} {away}** ({s0[2]:.1f}%)")
                    st.dataframe(pd.DataFrame([{"Score": f"{home} {i}–{j} {away}", "Prob": f"{q:.2f}%",
                                                "Result": "Home" if i > j else ("Away" if j > i else "Draw")}
                                               for i, j, q in mp["scores"]]), width="stretch", hide_index=True)
                with col_elo:
                    d1, d2 = st.columns(2)
                    d1.metric(f"{fa} Elo", f"{ea:,}", f"{ea-eb:+,}"); d2.metric(f"{fb} Elo", f"{eb:,}")
                    if neutral:
                        st.metric(f"{fa} lifts trophy", f"{final_win(ta, tb)*100:.1f}%")
                    st.metric("Model margin", f"{pA-pB:+.1f}pp")
            notes = [f"{fmt_club(t)} boost ×{TEAM_BOOST[t]:.2f}" for t in (ta, tb) if TEAM_BOOST.get(t, 1) != 1]
            if notes: st.caption(" · ".join(notes))
            st.caption(f"Closed-form Poisson · home xG +{HOME_ADV*100:.0f}% · draws in ties → ET → penalties")

# ─────────────────────────────────────────────────────────────────────
# TAB 7 — Model
# ─────────────────────────────────────────────────────────────────────
with tab_model:
    if use_api and api_key and n_matches_used > 0:
        st.success(f"🟢 Live ratings — {n_matches_used} UCL matches")
    else:
        st.info("🔵 Built-in ratings (derived from club Elo)")
    with st.expander("Format", expanded=True):
        st.markdown("""
- **League phase:** 36 clubs, 8 matches each (2 opponents per pot, 1 home + 1 away). Ranked on points → goal difference → goals scored.
- **1st–8th** go straight to the Round of 16. **9th–24th** play two-legged knockout play-offs (9/10 v 23/24, 11/12 v 21/22, 13/14 v 19/20, 15/16 v 17/18). **25th–36th** are out.
- **Round of 16 → semi-finals:** two legs, the higher league finisher hosts the 2nd leg. Level on aggregate → 30 min extra time → penalties.
- **Final:** single match at a neutral venue.
- Without live fixtures, a league-phase draw is generated from the pots (no same-association ties, max 2 opponents per association).
""")
    with st.expander("Poisson goal model"):
        st.markdown("""
```
xG_home = attack_home × defence_away × (1 + home_adv) × boost_home
xG_away = attack_away × defence_home × boost_away
```
Goals ~ Poisson(xG). Extra time = 1/3 of 90-min xG. Penalties weighted by each club's best goalkeeper rating.
Top scorer: each club's simulated goals are split between players by a goal share derived from position and overall rating
(or real UCL goal tallies when the API is on); 20% of goals go to "other" players.
""")
    with st.expander("Club ratings"):
        edf = pd.DataFrame([{"Club": fmt_club(k), "Pot": v.get("pot") or "—", "Atk": v["atk"], "Def": v["def"],
                             "Boost": TEAM_BOOST.get(k, 1.0), "Elo": v["elo"], "Pens": PEN.get(k, .5)}
                            for k, v in sorted(TEAMS.items(), key=lambda x: -x[1]["elo"])])
        st.dataframe(edf, width="stretch", hide_index=True,
                     column_config={"Atk": st.column_config.NumberColumn(format="%.3f"),
                                    "Def": st.column_config.NumberColumn(format="%.3f"),
                                    "Boost": st.column_config.NumberColumn(format="%.2f×"),
                                    "Pens": st.column_config.NumberColumn(format="%.2f")})
    with st.expander("vs professional models"):
        st.markdown("""
|Feature|This|Pro|
|---|---|---|
|Goal dist|Poisson ✅|Poisson / bivariate / neg-binomial|
|Ratings|Elo + live UCL results ✅|MLE on domestic + European matches|
|Home adv|Adjustable ✅|Per-club + travel|
|Manual boost|Any club ✅|Automated|
|Injuries / rotation|Not included|Live feeds|
|Penalties|GK-rating edge ✅|Player-level records|
|Top scorer|Goal-share model ✅|Shot-level xG + minutes|
""")

# ─────────────────────────────────────────────────────────────────────
# TAB 8 — API Status
# ─────────────────────────────────────────────────────────────────────
with tab_api:
    st.subheader("🔌 API Status")
    if not use_api or not api_key:
        st.info("Enable API and add key in sidebar.")
    else:
        ok_all = all(i == "✅" for i, _, _ in api_log)
        ok_any = any(i == "✅" for i, _, _ in api_log)
        if ok_all:   st.success("✅ All checks passed")
        elif ok_any: st.warning("⚠️ Partially working")
        else:        st.error("❌ No live data")
        st.divider()
        for icon, label, detail in api_log:
            ci, cb_ = st.columns([1, 10])
            with ci: st.markdown(f"## {icon}")
            with cb_:
                st.markdown(f"**{label}**")
                lines = detail.strip().split("\n")
                st.caption(lines[0])
                if len(lines) > 1:
                    with st.expander("Detail"): st.code("\n".join(lines[1:]))
        st.divider()
        st.markdown(f"""**If something returns 0:**
1. The Champions League is `league={UCL_LEAGUE_ID}` on API-Football; `season` is the year the season starts (2026 = 2026/27).
2. Free plans may not include the current season or player stats.
3. Check requests used vs your daily limit in the connection step.
4. Live fixtures are only used once all 144 league-phase matches are published; until then a simulated draw is used.
""")
        if st.button("🔄 Clear cache + retry", key="clr"):
            for fn in [fetch_recent_results, fetch_ucl_fixtures, fetch_top_scorers, fetch_player_cards_api]:
                fn.clear()
            st.success("Cleared — reloading.")
            st.rerun()
