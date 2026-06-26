import streamlit as st
import numpy as np
import pandas as pd
import random
from collections import defaultdict

# ==============================================================================
# 1. APPLICATION SYSTEM ARCHITECTURE & INTERFACE
# ==============================================================================
st.set_page_config(
    page_title="Professional 48-Team WC Predictor Engine", 
    page_icon="🏆", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏆 Professional 48-Team World Cup Simulation Dashboard")
st.write(
    "An enterprise-level data simulation framework executing the exact "
    "104-match format. This system processes a full group-stage round-robin, "
    "calculates wildcards dynamically, and routes results through the official knockout trees."
)

# ==============================================================================
# 2. DEFINITIVE 48-TEAM INTERNATIONALLY CALIBRATED DATABASE
# ==============================================================================
# Baseline parameters map team capabilities. 
# ATT: Attacking Force, MID: Midfield Transition, DEF: Defensive Rigidity.
# HOST: Territorial bonus coefficient (1.05 for USA, Mexico, Canada).
# COMP: Composure/Experience rating used for extra-time and shootout matrices.
TEAM_METRICS = {
    # GROUP A
    "Mexico": {"ATT": 80, "MID": 79, "DEF": 78, "HOST": 1.05, "COMP": 78, "GROUP": "A"},
    "South Africa": {"ATT": 71, "MID": 72, "DEF": 71, "HOST": 1.00, "COMP": 70, "GROUP": "A"},
    "South Korea": {"ATT": 77, "MID": 76, "DEF": 75, "HOST": 1.00, "COMP": 76, "GROUP": "A"},
    "Czechia": {"ATT": 76, "MID": 76, "DEF": 75, "HOST": 1.00, "COMP": 74, "GROUP": "A"},
    # GROUP B
    "Canada": {"ATT": 76, "MID": 75, "DEF": 74, "HOST": 1.05, "COMP": 73, "GROUP": "B"},
    "Bosnia-Herzegovina": {"ATT": 73, "MID": 73, "DEF": 72, "HOST": 1.00, "COMP": 71, "GROUP": "B"},
    "Qatar": {"ATT": 70, "MID": 71, "DEF": 69, "HOST": 1.00, "COMP": 72, "GROUP": "B"},
    "Switzerland": {"ATT": 78, "MID": 80, "DEF": 79, "HOST": 1.00, "COMP": 81, "GROUP": "B"},
    # GROUP C
    "Brazil": {"ATT": 89, "MID": 86, "DEF": 85, "HOST": 1.00, "COMP": 87, "GROUP": "C"},
    "Morocco": {"ATT": 83, "MID": 85, "DEF": 84, "HOST": 1.00, "COMP": 84, "GROUP": "C"},
    "Haiti": {"ATT": 65, "MID": 64, "DEF": 63, "HOST": 1.00, "COMP": 62, "GROUP": "C"},
    "Scotland": {"ATT": 74, "MID": 76, "DEF": 75, "HOST": 1.00, "COMP": 73, "GROUP": "C"},
    # GROUP D
    "USA": {"ATT": 82, "MID": 81, "DEF": 80, "HOST": 1.05, "COMP": 80, "GROUP": "D"},
    "Paraguay": {"ATT": 74, "MID": 75, "DEF": 76, "HOST": 1.00, "COMP": 75, "GROUP": "D"},
    "Australia": {"ATT": 73, "MID": 74, "DEF": 74, "HOST": 1.00, "COMP": 76, "GROUP": "D"},
    "Türkiye": {"ATT": 80, "MID": 79, "DEF": 77, "HOST": 1.00, "COMP": 77, "GROUP": "D"},
    # GROUP E
    "Germany": {"ATT": 86, "MID": 87, "DEF": 84, "HOST": 1.00, "COMP": 85, "GROUP": "E"},
    "Curaçao": {"ATT": 63, "MID": 64, "DEF": 65, "HOST": 1.00, "COMP": 60, "GROUP": "E"},
    "Ivory Coast": {"ATT": 78, "MID": 77, "DEF": 76, "HOST": 1.00, "COMP": 75, "GROUP": "E"},
    "Ecuador": {"ATT": 77, "MID": 78, "DEF": 79, "HOST": 1.00, "COMP": 77, "GROUP": "E"},
    # GROUP F
    "Netherlands": {"ATT": 84, "MID": 85, "DEF": 86, "HOST": 1.00, "COMP": 84, "GROUP": "F"},
    "Japan": {"ATT": 80, "MID": 81, "DEF": 79, "HOST": 1.00, "COMP": 82, "GROUP": "F"},
    "Sweden": {"ATT": 78, "MID": 77, "DEF": 77, "HOST": 1.00, "COMP": 76, "GROUP": "F"},
    "Tunisia": {"ATT": 72, "MID": 73, "DEF": 73, "HOST": 1.00, "COMP": 72, "GROUP": "F"},
    # GROUP G
    "Belgium": {"ATT": 84, "MID": 83, "DEF": 81, "HOST": 1.00, "COMP": 82, "GROUP": "G"},
    "Egypt": {"ATT": 76, "MID": 74, "DEF": 74, "HOST": 1.00, "COMP": 75, "GROUP": "G"},
    "Iran": {"ATT": 75, "MID": 74, "DEF": 75, "HOST": 1.00, "COMP": 74, "GROUP": "G"},
    "New Zealand": {"ATT": 68, "MID": 67, "DEF": 68, "HOST": 1.00, "COMP": 69, "GROUP": "G"},
    # GROUP H
    "Spain": {"ATT": 89, "MID": 92, "DEF": 86, "HOST": 1.00, "COMP": 90, "GROUP": "H"},
    "Cape Verde": {"ATT": 70, "MID": 71, "DEF": 70, "HOST": 1.00, "COMP": 68, "GROUP": "H"},
    "Saudi Arabia": {"ATT": 71, "MID": 72, "DEF": 70, "HOST": 1.00, "COMP": 73, "GROUP": "H"},
    "Uruguay": {"ATT": 82, "MID": 83, "DEF": 82, "HOST": 1.00, "COMP": 83, "GROUP": "H"},
    # GROUP I
    "France": {"ATT": 92, "MID": 88, "DEF": 88, "HOST": 1.00, "COMP": 89, "GROUP": "I"},
    "Senegal": {"ATT": 78, "MID": 78, "DEF": 78, "HOST": 1.00, "COMP": 77, "GROUP": "I"},
    "Iraq": {"ATT": 71, "MID": 71, "DEF": 70, "HOST": 1.00, "COMP": 71, "GROUP": "I"},
    "Norway": {"ATT": 81, "MID": 77, "DEF": 76, "HOST": 1.00, "COMP": 75, "GROUP": "I"},
    # GROUP J
    "Argentina": {"ATT": 91, "MID": 89, "DEF": 87, "HOST": 1.00, "COMP": 91, "GROUP": "J"},
    "Algeria": {"ATT": 76, "MID": 76, "DEF": 74, "HOST": 1.00, "COMP": 74, "GROUP": "J"},
    "Austria": {"ATT": 78, "MID": 80, "DEF": 78, "HOST": 1.00, "COMP": 79, "GROUP": "J"},
    "Jordan": {"ATT": 68, "MID": 67, "DEF": 68, "HOST": 1.00, "COMP": 66, "GROUP": "J"},
    # GROUP K
    "Portugal": {"ATT": 88, "MID": 87, "DEF": 84, "HOST": 1.00, "COMP": 86, "GROUP": "K"},
    "Congo DR": {"ATT": 73, "MID": 72, "DEF": 71, "HOST": 1.00, "COMP": 70, "GROUP": "K"},
    "Uzbekistan": {"ATT": 72, "MID": 73, "DEF": 72, "HOST": 1.00, "COMP": 72, "GROUP": "K"},
    "Colombia": {"ATT": 82, "MID": 81, "DEF": 81, "HOST": 1.00, "COMP": 80, "GROUP": "K"},
    # GROUP L
    "England": {"ATT": 89, "MID": 88, "DEF": 86, "HOST": 1.00, "COMP": 85, "GROUP": "L"},
    "Croatia": {"ATT": 79, "MID": 83, "DEF": 80, "HOST": 1.00, "COMP": 86, "GROUP": "L"},
    "Ghana": {"ATT": 75, "MID": 75, "DEF": 73, "HOST": 1.00, "COMP": 73, "GROUP": "L"},
    "Panama": {"ATT": 71, "MID": 71, "DEF": 72, "HOST": 1.00, "COMP": 71, "GROUP": "L"}
}

# ==============================================================================
# 3. SIDEBAR SIMULATION RHEOSTAT CONTROLS
# ==============================================================================
st.sidebar.header("⚙️ Simulation Settings")
sim_volume = st.sidebar.slider("Number of Tournaments", min_value=50, max_value=1000, value=200, step=50)

st.sidebar.subheader("🚑 Dynamic Engine Toggles")
enable_fatigue = st.sidebar.checkbox("Squad Fatigue Accumulation", value=True)
enable_injuries = st.sidebar.checkbox("In-Match Injury Factors", value=True)
enable_cards = st.sidebar.checkbox("Suspension Limits", value=True)

# Helper function to review database state cleanly
with st.expander("🔍 View Engine Dataset Configurations for All 48 Nations"):
    st.dataframe(pd.DataFrame.from_dict(TEAM_METRICS, orient='index'), use_container_width=True)

# ==============================================================================
# 4. INTER-MATCH MATRIX & STATISTICAL COUPLING
# ==============================================================================
def run_poisson_match(t1, t2, tournament_state, is_knockout=False):
    """Calculates match scores using historical attribute balancing and Poisson distributions."""
    m1, m2 = TEAM_METRICS[t1], TEAM_METRICS[t2]
    
    # Isolate base strengths
    att1, mid1, def1 = m1["ATT"], m1["MID"], m1["DEF"]
    att2, mid2, def2 = m2["ATT"], m2["MID"], m2["DEF"]
    
    # Calculate Host Country Coefficient boosts
    h1 = m1["HOST"] if not is_knockout else 1.01 if m1["HOST"] > 1.0 else 1.0
    h2 = m2["HOST"] if not is_knockout else 1.01 if m2["HOST"] > 1.0 else 1.0
    
    # Apply Cumulative Tournament Attrition
    f1 = tournament_state["fatigue"].get(t1, 1.0) if enable_fatigue else 1.0
    f2 = tournament_state["fatigue"].get(t2, 1.0) if enable_fatigue else 1.0
    
    # Apply Active Roster Suspensions (2 Yellow Cards = 5% tactical penalty)
    s1 = 0.95 if (enable_cards and tournament_state["cards"].get(t1, 0) >= 2) else 1.0
    s2 = 0.95 if (enable_cards and tournament_state["cards"].get(t2, 0) >= 2) else 1.0
    
    # Check for unexpected in-match physical degradation (4% odds)
    if enable_injuries and random.random() < 0.04:
        if random.choice([True, False]): f1 *= 0.90
        else: f2 *= 0.90
        
    # Scale Expected Goals (xG) parameters
    t1_xg = max(0.2, ((mid1 * 0.5 + att1 * 0.5) / (def2 * 1.15)) * 1.45 * h1 * f1 * s1)
    t2_xg = max(0.2, ((mid2 * 0.5 + att2 * 0.5) / (def1 * 1.15)) * 1.45 * h2 * f2 * s2)
    
    g1 = np.random.poisson(t1_xg)
    g2 = np.random.poisson(t2_xg)
    
    # Append physical stress records for the subsequent tier
    tournament_state["fatigue"][t1] = max(0.75, tournament_state["fatigue"].get(t1, 1.0) - 0.03)
    tournament_state["fatigue"][t2] = max(0.75, tournament_state["fatigue"].get(t2, 1.0) - 0.03)
    
    # Append disciplinary records randomly (60% match occurrence)
    if enable_cards:
        if random.random() < 0.60: tournament_state["cards"][t1] = tournament_state["cards"].get(t1, 0) + 1
        if random.random() < 0.60: tournament_state["cards"][t2] = tournament_state["cards"].get(t2, 0) + 1
        
    if not is_knockout:
        return g1, g2, "FT"
        
    if g1 != g2:
        return g1, g2, "FT"
        
    # Extra Time (AET) Resolution Matrix (30% breakthrough capacity)
    if random.random() < 0.30:
        if m1["COMP"] + random.randint(-4, 4) > m2["COMP"] + random.randint(-4, 4):
            return g1 + 1, g2, "AET"
        else:
            return g1, g2 + 1, "AET"
            
    # Penalty Shootout (PEN) Composure Weighting Equation
    t1_pen_weight = m1["COMP"] / (m1["COMP"] + m2["COMP"])
    if random.random() < t1_pen_weight:
        return g1, g2, "PEN_T1"
    else:
        return g1, g2, "PEN_T2"

# ==============================================================================
# 5. CORE 104-MATCH DYNAMIC TOURNAMENT PIPELINE
# ==============================================================================
def run_world_cup_matrix():
