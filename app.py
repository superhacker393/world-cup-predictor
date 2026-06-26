import streamlit as st
import numpy as np
import pandas as pd

# 1. WEB APP INTERFACE CONFIGURATION
st.set_page_config(page_title="World Cup 48-Team AI Predictor", page_icon="🏆", layout="centered")
st.title("🏆 World Cup 48-Team Monte Carlo AI Predictor")
st.write(
    "This app uses a complete 48-team database, Poisson goal distributions, "
    "and Monte Carlo simulation models to predict the tournament champion."
)

# 2. GROUND TRUTH DATABASE (All 48 World Cup Teams)
TEAM_METRICS = {
    "Argentina": {"POINTS": 1877.27, "OFFENSE": 2.20, "DEFENSE": 0.80},
    "France": {"POINTS": 1870.70, "OFFENSE": 2.18, "DEFENSE": 0.81},
    "Spain": {"POINTS": 1874.71, "OFFENSE": 2.15, "DEFENSE": 0.82},
    "England": {"POINTS": 1828.02, "OFFENSE": 2.10, "DEFENSE": 0.84},
    "Brazil": {"POINTS": 1765.86, "OFFENSE": 2.05, "DEFENSE": 0.85},
    "Portugal": {"POINTS": 1767.85, "OFFENSE": 2.00, "DEFENSE": 0.87},
    "Netherlands": {"POINTS": 1753.57, "OFFENSE": 1.95, "DEFENSE": 0.88},
    "Belgium": {"POINTS": 1748.20, "OFFENSE": 1.90, "DEFENSE": 0.89},
    "Italy": {"POINTS": 1724.30, "OFFENSE": 1.88, "DEFENSE": 0.86},
    "Germany": {"POINTS": 1646.32, "OFFENSE": 1.92, "DEFENSE": 0.90},
    "Croatia": {"POINTS": 1721.10, "OFFENSE": 1.82, "DEFENSE": 0.89},
    "Morocco": {"POINTS": 1755.10, "OFFENSE": 1.80, "DEFENSE": 0.88},
    "Uruguay": {"POINTS": 1660.20, "OFFENSE": 1.85, "DEFENSE": 0.91},
    "USA": {"POINTS": 1642.10, "OFFENSE": 1.80, "DEFENSE": 0.93},
    "Colombia": {"POINTS": 1654.40, "OFFENSE": 1.82, "DEFENSE": 0.90},
    "Mexico": {"POINTS": 1632.20, "OFFENSE": 1.75, "DEFENSE": 0.94},
    "Japan": {"POINTS": 1628.40, "OFFENSE": 1.84, "DEFENSE": 0.92},
    "Senegal": {"POINTS": 1622.10, "OFFENSE": 1.74, "DEFENSE": 0.92},
    "Iran": {"POINTS": 1611.20, "OFFENSE": 1.72, "DEFENSE": 0.93},
    "Denmark": {"POINTS": 1608.50, "OFFENSE": 1.76, "DEFENSE": 0.94},
    "Switzerland": {"POINTS": 1602.10, "OFFENSE": 1.70, "DEFENSE": 0.91},
    "South Korea": {"POINTS": 1589.40, "OFFENSE": 1.74, "DEFENSE": 0.95},
    "Australia": {"POINTS": 1571.20, "OFFENSE": 1.68, "DEFENSE": 0.94},
    "Ukraine": {"POINTS": 1565.30, "OFFENSE": 1.72, "DEFENSE": 0.96},
    "Austria": {"POINTS": 1560.10, "OFFENSE": 1.70, "DEFENSE": 0.95},
    "Sweden": {"POINTS": 1545.20, "OFFENSE": 1.71, "DEFENSE": 0.97},
    "Hungary": {"POINTS": 1532.10, "OFFENSE": 1.68, "DEFENSE": 0.96},
    "Nigeria": {"POINTS": 1520.40, "OFFENSE": 1.73, "DEFENSE": 0.99},
    "Wales": {"POINTS": 1515.20, "OFFENSE": 1.62, "DEFENSE": 0.96},
    "Poland": {"POINTS": 1510.30, "OFFENSE": 1.66, "DEFENSE": 0.98},
    "Ecuador": {"POINTS": 1508.20, "OFFENSE": 1.65, "DEFENSE": 0.95},
    "Peru": {"POINTS": 1495.10, "OFFENSE": 1.58, "DEFENSE": 0.96},
    "Chile": {"POINTS": 1488.40, "OFFENSE": 1.60, "DEFENSE": 0.98},
    "Tunisia": {"POINTS": 1475.30, "OFFENSE": 1.52, "DEFENSE": 0.95},
    "Algeria": {"POINTS": 1468.20, "OFFENSE": 1.62, "DEFENSE": 0.99},
    "Egypt": {"POINTS": 1460.10, "OFFENSE": 1.64, "DEFENSE": 1.00},
    "Canada": {"POINTS": 1458.50, "OFFENSE": 1.66, "DEFENSE": 1.02},
    "Scotland": {"POINTS": 1450.20, "OFFENSE": 1.56, "DEFENSE": 0.99},
    "Costa Rica": {"POINTS": 1435.40, "OFFENSE": 1.50, "DEFENSE": 1.01},
    "Turkey": {"POINTS": 1428.10, "OFFENSE": 1.62, "DEFENSE": 1.04},
    "Cameroon": {"POINTS": 1420.30, "OFFENSE": 1.56, "DEFENSE": 1.02},
    "Mali": {"POINTS": 1412.10, "OFFENSE": 1.50, "DEFENSE": 1.01},
    "Saudi Arabia": {"POINTS": 1405.40, "OFFENSE": 1.52, "DEFENSE": 1.05},
    "Qatar": {"POINTS": 1398.20, "OFFENSE": 1.54, "DEFENSE": 1.06},
    "Ghana": {"POINTS": 1392.10, "OFFENSE": 1.56, "DEFENSE": 1.08},
    "Panama": {"POINTS": 1375.40, "OFFENSE": 1.46, "DEFENSE": 1.05},
    "Jamaica": {"POINTS": 1360.20, "OFFENSE": 1.48, "DEFENSE": 1.07},
    "New Zealand": {"POINTS": 1248.60, "OFFENSE": 1.40, "DEFENSE": 1.15}
}

# 3. INTERACTIVE SIDEBAR CONTROLS
st.sidebar.header("🛠️ Simulation Engine Configuration")
num_simulations = st.sidebar.slider("Number of Tournaments", min_value=100, max_value=5000, value=1000, step=100)

st.sidebar.subheader("Adjust Team Form Modifiers")
selected_team = st.sidebar.selectbox("Select Team to Modify", list(TEAM_METRICS.keys()))

# Maintain form selections persistent in Streamlit session state
if "modifiers" not in st.session_state:
    st.session_state.modifiers = {team: 1.0 for team in TEAM_METRICS.keys()}

st.session_state.modifiers[selected_team] = st.sidebar.slider(
    f"Form Boost for {selected_team}", min_value=0.8, max_value=1.2, value=st.session_state.modifiers[selected_team], step=0.05
)

# 4. MATH & ENGINE SIMULATION LOGIC
def simulate_poisson_match(team1, team2):
    t1, t2 = TEAM_METRICS[team1], TEAM_METRICS[team2]
    
    t1_form = st.session_state.modifiers[team1]
    t2_form = st.session_state.modifiers[team2]
    
    # Calculate expected goals (xG) based on form adjusted offense vs defense
    t1_xg = max(0.4, (t1["OFFENSE"] * t1_form) * t2["DEFENSE"])
    t2_xg = max(0.4, (t2["OFFENSE"] * t2_form) * t1["DEFENSE"])
    
    g1 = np.random.poisson(t1_xg)
    g2 = np.random.poisson(t2_xg)
    
    if g1 != g2:
        return team1 if g1 > g2 else team2
    
    # Standard Knockout Tiebreaker using weighted probabilities based on FIFA Points
    if np.random.rand() < 0.3:
        return team1 if np.random.rand() > 0.5 else team2
    else:
        t1_weight = 0.55 if t1["POINTS"] > t2["POINTS"] else 0.45
        return team1 if np.random.rand() < t1_weight else team2

def run_single_tournament():
    # Fetch all 48 teams and randomly seed them into a knockout structure
    current_round = list(TEAM_METRICS.keys())
    np.random.shuffle(current_round)
    
    # Process knockout rounds: 48 -> 24 -> 12 -> 6 -> 3 -> Final Group / Tiebreaker progression
    while len(current_round) > 1:
        next_round = []
        # If odd number of teams left (e.g. 3 teams), the last one gets a bye to speed up logic
        for i in range(0, len(current_round), 2):
            if i + 1 < len(current_round):
                winner = simulate_poisson_match(current_round[i], current_round[i+1])
                next_round.append(winner)
            else:
                next_round.append(current_round[i])
        current_round = next_round
        
    return current_round[0]

# 5. EXECUTION CORE RUNNER
if st.button("🚀 Run AI Tournament Simulation"):
    championship_counts = {team: 0 for team in TEAM_METRICS.keys()}
    
    # Create progress visual containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Perform Monte Carlo simulation passes
    for i in range(num_simulations):
        winner = run_single_tournament()
        championship_counts[winner] += 1
        
        # Smoothly update progress bars without overloading the frontend UI
        step = max(1, num_simulations // 10)
        if i % step == 0:
            progress_bar.progress((i + 1) / num_simulations)
            status_text.text(f"Simulating tournament run {i:,} of {num_simulations:,}...")
            
    progress_bar.empty()
    status_text.empty()
    
    # Structure data into a summary dataframe
    results_df = pd.DataFrame(list(championship_counts.items()), columns=["Country", "Simulated Wins"])
    results_df["Win Probability"] = (results_df["Simulated Wins"] / num_simulations) * 100
    results_df = results_df.sort_values(by="Win Probability", ascending=False).reset_index(drop=True)
    
    # 6. RESULTS LAYOUT RENDERING
    st.subheader("📊 Mathematical Probability Results")
    
    # Interactive data presentation matrix
    st.dataframe(
        results_df[results_df["Simulated Wins"] > 0].style.format({"Win Probability": "{:.2f}%"}), 
        use_container_width=True
    )
    
    # Highlight presentation of the statistical favorite
    top_team = results_df.iloc[0]["Country"]
    top_prob = results_df.iloc[0]["Win Probability"]
    
    st.success(
        f"🏆 The AI Predictor determines **{top_team.upper()}** has the highest statistical probability "
        f"of winning the Cup, capturing the trophy in **{top_prob:.2f}%** of simulation realities!"
    )
    
    # Interactive bar chart layout
    st.bar_chart(data=results_df[results_df["Win Probability"] > 0.5], x="Country", y="Win Probability")
