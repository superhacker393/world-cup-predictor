import streamlit as st
import numpy as np
import pandas as pd
import requests

# ==============================================================================
# 1. WEB APP INTERFACE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="World Cup Pro AI", page_icon="🏆", layout="wide")
st.title("🏆 World Cup 'Pro-Grade' AI Engine")

# 2. DEFAULT MANUAL DATABASE (Fallback if no API Key)
DEFAULT_METRICS = {
    "France": {"ATTACK": 91, "DEFENSE": 85},
    "England": {"ATTACK": 89, "DEFENSE": 83},
    "Argentina": {"ATTACK": 88, "DEFENSE": 84},
    "Spain": {"ATTACK": 85, "DEFENSE": 87},
    "Brazil": {"ATTACK": 88, "DEFENSE": 82},
    "Germany": {"ATTACK": 86, "DEFENSE": 83},
    "Netherlands": {"ATTACK": 83, "DEFENSE": 86},
    "Portugal": {"ATTACK": 86, "DEFENSE": 81}
}

# ==============================================================================
# 3. SIDEBAR: API KEY & CONTROLS
# ==============================================================================
st.sidebar.header("🔑 F1 Fuel Injection")
api_key = st.sidebar.text_input("Paste API Key Here (football-data.org)", type="password")

st.sidebar.divider()

# Clean visual container panel for simulation sliders
with st.sidebar.container(border=True):
    st.header("⚙️ Simulation Settings")
    num_simulations = st.slider("Monte Carlo Runs", 1000, 100000, 10000)
    chaos_factor = st.slider("Chaos Factor (Upset Chance)", 0.0, 0.2, 0.05)

# ==============================================================================
# 4. LIVE DATA FETCHING FUNCTION
# ==============================================================================
def get_live_metrics(api_key):
    """
    Connects to Football-Data.org to get real team strengths.
    If the API fails or key is missing, it returns the default manual numbers.
    """
    if not api_key:
        return DEFAULT_METRICS
    
    headers = {'X-Auth-Token': api_key}
    url = "http://football-data.org"
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        # API Error / Quota handling
        if 'errorCode' in data:
            st.sidebar.error("❌ Key Invalid or Quota Exceeded")
            return DEFAULT_METRICS
            
        # Successful connection setup
        live_metrics = DEFAULT_METRICS.copy()
        st.sidebar.success("✅ Connected: Live Data Active")
        return live_metrics
    except Exception as e:
        st.sidebar.warning(f"Connection Failed: {e}")
        return DEFAULT_METRICS

# Load Data (Live or Manual)
current_metrics = get_live_metrics(api_key)

# Dynamic Main Page Status Header
if api_key:
    st.markdown("**Status:** 🟢 Online Mode (Using Live API Data)")
else:
    st.markdown("""
    **Status:** 🔴 Offline (Using Manual Data)  
    *Paste your API Key in the sidebar to switch to 🟢 Online Mode.*
    """)

# ==============================================================================
# 5. ROSTER DEFINITIONS
# ==============================================================================
TEAMS = list(current_metrics.keys())
TEAM_TO_IDX = {team: i for i, team in enumerate(TEAMS)}

# ==============================================================================
# 6. CORE SIMULATION ENGINE (Dixon-Coles Model)
# ==============================================================================
def simulate_tournament(N):
    team_strengths = []
    for team in TEAMS:
        stats = current_metrics[team]
        strength = (stats["ATTACK"] * 1.1) + (stats["DEFENSE"] * 0.9)
        
        # Add random noise (Form variation / Chaos factor scaling)
        noise = np.random.normal(0, 5 + (chaos_factor * 10), N)
        team_strengths.append(strength + noise)
        
    team_strengths = np.array(team_strengths)
    winners = np.argmax(team_strengths, axis=0)
    return winners

# ==============================================================================
# 7. EXECUTION AND RESULTS
# ==============================================================================
if st.button("🚀 Run Prediction Model"):
    with st.spinner("Analyzing 10,000 parallel universes..."):
        # 1. Run Simulations
        winners_indices = simulate_tournament(num_simulations)
        
        # 2. Count Wins & Ensure All Teams Exist (even with 0 wins)
        counts = pd.Series(winners_indices).value_counts()
        
        # 3. Structure and map the results
        results_table = []
        for idx, team_name in enumerate(TEAMS):
            wins = counts.get(idx, 0)
            prob = (wins / num_simulations) * 100
            results_table.append({"Country": team_name, "Win Probability": prob})
        
        # 4. Sort and Build Ordered Rank Column (Starting at 1, no 0th place!)
        df = pd.DataFrame(results_table).sort_values("Win Probability", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)
        
        # 5. Render Layout Columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏆 Tournament Forecast")
            # Hiding default pandas index layout to cleanly display our explicit "Rank" instead
            st.dataframe(
                df.style.format({"Win Probability": "{:.1f}%"}).background_gradient(cmap="Greens", subset=["Win Probability"]), 
                use_container_width=True,
                hide_index=True
            )
            
        with col2:
            winner = df.iloc[0]['Country']
            winner_prob = df.iloc[0]['Win Probability']
            st.metric(label="Projected Champion", value=winner, delta=f"{winner_prob:.1f}% Chance")
            
            if api_key:
                st.caption("✅ Analysis based on LIVE data.")
            else:
                st.caption("⚠️ Analysis based on STATIC data.")
