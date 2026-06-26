import streamlit as st
import numpy as np
import pandas as pd

# 1. WEB APP INTERFACE CONFIGURATION
st.set_page_config(page_title="World Cup AI Predictor", page_icon="🏆", layout="centered")

st.title("🏆 World Cup Monte Carlo AI Predictor")
st.write("This app uses official FIFA Elo points, Poisson goal distributions, and macro-level data crunching to simulate the World Cup thousands of times.")

# 2. GROUND TRUTH DATABASE (Pre-WC June 2026 Official FIFA Points)
TEAM_METRICS = {
    "Argentina":   {"POINTS": 1889.06, "OFFENSE": 2.10, "DEFENSE": 0.85},
    "France":      {"POINTS": 1887.11, "OFFENSE": 2.15, "DEFENSE": 0.90},
    "Spain":       {"POINTS": 1856.03, "OFFENSE": 1.95, "DEFENSE": 0.80},
    "England":     {"POINTS": 1828.02, "OFFENSE": 1.90, "DEFENSE": 0.88},
    "Brazil":      {"POINTS": 1765.34, "OFFENSE": 1.85, "DEFENSE": 0.92},
    "Portugal":    {"POINTS": 1755.09, "OFFENSE": 1.80, "DEFENSE": 0.95},
    "Netherlands": {"POINTS": 1743.20, "OFFENSE": 1.70, "DEFENSE": 0.91},
    "Germany":     {"POINTS": 1736.00, "OFFENSE": 1.75, "DEFENSE": 0.96}
}

# INTERACTIVE SIDEBAR: Allow users to adjust baseline parameters on the website
st.sidebar.header("🛠️ Simulation Controls")
num_simulations = st.sidebar.slider("Number of Tournaments to Simulate", min_value=500, max_value=20000, value=10000, step=500)

st.sidebar.subheader("Adjust Team Form Modifiers")
form_modifiers = {}
for team in TEAM_METRICS.keys():
    form_modifiers[team] = st.sidebar.slider(f"{team} Form Boost", min_value=0.8, max_value=1.2, value=1.0, step=0.05)

# 3. SIMULATION CORE LOGIC
def simulate_poisson_match(team1, team2):
    t1, t2 = TEAM_METRICS[team1], TEAM_METRICS[team2]
    
    # Apply user-controlled form modifications from the website interface
    t1_offense = t1["OFFENSE"] * form_modifiers[team1]
    t2_offense = t2["OFFENSE"] * form_modifiers[team2]
    
    # Calculate baseline expected goals (xG) based on offensive power vs defensive resistance
    t1_xg = max(0.5, t1_offense * t2["DEFENSE"])
    t2_xg = max(0.5, t2_offense * t1["DEFENSE"])
    
    # Generate random match goals utilizing standard soccer distribution math
    g1 = np.random.poisson(t1_xg)
    g2 = np.random.poisson(t2_xg)
    
    if g1 != g2:
        return team1 if g1 > g2 else team2
        
    # Standard Knockout Tiebreakers
    if np.random.rand() < 0.3:
        return team1 if np.random.rand() > 0.5 else team2
    else:
        t1_weight = 0.54 if t1["POINTS"] > t2["POINTS"] else 0.46
        return team1 if np.random.rand() < t1_weight else team2

def run_single_tournament():
    q1 = simulate_poisson_match("Argentina", "Portugal")
    q2 = simulate_poisson_match("France", "Germany")
    q3 = simulate_poisson_match("Spain", "Netherlands")
    q4 = simulate_poisson_match("England", "Brazil")
    
    sf1 = simulate_poisson_match(q1, q2)
    sf2 = simulate_poisson_match(q3, q4)
    
    return simulate_poisson_match(sf1, sf2)

# 4. EXECUTION TRIGGER
if st.button("🚀 Run AI Tournament Simulation"):
    championship_counts = {team: 0 for team in TEAM_METRICS.keys()}
    
    # Progress indicator for the website interface
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run loop
    for i in range(num_simulations):
        winner = run_single_tournament()
        championship_counts[winner] += 1
        
        # Update progress bar occasionally to preserve processing speed
        if i % (num_simulations // 10) == 0:
            progress_bar.progress((i + 1) / num_simulations)
            status_text.text(f"Simulating tournament vector {i:,} of {num_simulations:,}...")
            
    progress_bar.empty()
    status_text.empty()
    
    # Compile Data Frame for Display
    results_df = pd.DataFrame(list(championship_counts.items()), columns=["Country", "Simulated Wins"])
    results_df["Win Probability"] = (results_df["Simulated Wins"] / num_simulations) * 100
    results_df = results_df.sort_values(by="Win Probability", ascending=False).reset_index(drop=True)
    
    # Display Results via Interactive Web Layout
    st.subheader("📊 Mathematical Probability Results")
    
    # Nicely formatted table output
    st.dataframe(
        results_df.style.format({"Win Probability": "{:.2f}%"}),
        use_container_width=True
    )
    
    # Graphic display of the top favorite
    top_team = results_df.iloc[0]["Country"]
    top_prob = results_df.iloc[0]["Win Probability"]
    st.success(f"🏆 The AI predicts **{top_team.upper()}** has the highest statistical chance of winning the tournament, taking the trophy in **{top_prob:.2f}%** of simulation realities!")
    
    # Interactive bar chart visual
    st.bar_chart(data=results_df, x="Country", y="Win Probability")
