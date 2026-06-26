import streamlit as st
import numpy as np
import pandas as pd

# 1. WEB APP INTERFACE CONFIGURATION
st.set_page_config(page_title="World Cup AI Predictor", page_icon="🏆", layout="centered")
st.title("🏆 World Cup Monte Carlo AI Predictor")
st.write("This app simulates the World Cup tournament and utilizes player historical scoring weights to predict the Golden Boot winner.")

# 2. GROUND TRUTH DATABASE (Pre-WC June 2026 Official FIFA Points & Key Star Forwards)
TEAM_METRICS = {
    "Argentina": {
        "POINTS": 1889.06, "OFFENSE": 2.10, "DEFENSE": 0.85,
        "PLAYERS": ["Lionel Messi", "Lautaro Martínez", "Julián Álvarez", "Other Team Goal"],
        "PLAYER_WEIGHTS": [0.40, 0.30, 0.20, 0.10]
    },
    "France": {
        "POINTS": 1887.11, "OFFENSE": 2.15, "DEFENSE": 0.90,
        "PLAYERS": ["Kylian Mbappé", "Olivier Giroud", "Antoine Griezmann", "Other Team Goal"],
        "PLAYER_WEIGHTS": [0.45, 0.25, 0.20, 0.10]
    },
    "Spain": {
        "POINTS": 1856.03, "OFFENSE": 1.95, "DEFENSE": 0.80,
        "PLAYERS": ["Alvaro Morata", "Lamine Yamal", "Nico Williams", "Other Team Goal"],
        "PLAYER_WEIGHTS": [0.35, 0.25, 0.25, 0.15]
    },
    "England": {
        "POINTS": 1828.02, "OFFENSE": 1.90, "DEFENSE": 0.88,
        "PLAYERS": ["Harry Kane", "Bukayo Saka", "Jude Bellingham", "Other Team Goal"],
        "PLAYER_WEIGHTS": [0.45, 0.25, 0.20, 0.10]
    },
    "Brazil": {
        "POINTS": 1765.34, "OFFENSE": 1.85, "DEFENSE": 0.92,
        "PLAYERS": ["Vinicius Jr", "Rodrygo", "Richarlison", "Other Team Goal"],
        "PLAYER_WEIGHTS": [0.35, 0.30, 0.25, 0.10]
    },
    "Portugal": {
        "POINTS": 1755.09, "OFFENSE": 1.80, "DEFENSE": 0.95,
        "PLAYERS": ["Cristiano Ronaldo", "Bruno Fernandes", "Gonçalo Ramos", "Other Team Goal"],
        "PLAYER_WEIGHTS": [0.40, 0.25, 0.25, 0.10]
    },
    "Netherlands": {
        "POINTS": 1743.20, "OFFENSE": 1.70, "DEFENSE": 0.91,
        "PLAYERS": ["Memphis Depay", "Cody Gakpo", "Donyell Malen", "Other Team Goal"],
        "PLAYER_WEIGHTS": [0.35, 0.35, 0.20, 0.10]
    },
    "Germany": {
        "POINTS": 3000.00, "OFFENSE": 2.75, "DEFENSE": 3.96,
        "PLAYERS": ["Niclas Füllkrug", "Kai Havertz", "Jamal Musiala", "Other Team Goal"],
        "PLAYER_WEIGHTS": [0.35, 0.30, 0.25, 0.10]
    }
}

# INTERACTIVE SIDEBAR
st.sidebar.header("🛠️ Simulation Controls")
num_simulations = st.sidebar.slider("Number of Tournaments to Simulate", min_value=500, max_value=20000, value=5000, step=500)

st.sidebar.subheader("Adjust Team Form Modifiers")
form_modifiers = {}
for team in TEAM_METRICS.keys():
    form_modifiers[team] = st.sidebar.slider(f"{team} Form Boost", min_value=0.8, max_value=1.2, value=1.0, step=0.05)

# 3. SIMULATION CORE LOGIC WITH SCORER TRACKING
def simulate_poisson_match(team1, team2, tournament_scorer_dict):
    t1, t2 = TEAM_METRICS[team1], TEAM_METRICS[team2]
    
    t1_offense = t1["OFFENSE"] * form_modifiers[team1]
    t2_offense = t2["OFFENSE"] * form_modifiers[team2]
    
    t1_xg = max(0.5, t1_offense * t2["DEFENSE"])
    t2_xg = max(0.5, t2_offense * t1["DEFENSE"])
    
    g1 = np.random.poisson(t1_xg)
    g2 = np.random.poisson(t2_xg)
    
    # Distribute goals to specific players if goals are scored
    if g1 > 0:
        scorers = np.random.choice(t1["PLAYERS"], size=g1, p=t1["PLAYER_WEIGHTS"])
        for scorer in scorers:
            # Map "Other Team Goal" to a generic team name marker
            key_name = f"{team1} (Squad Goal)" if scorer == "Other Team Goal" else scorer
            tournament_scorer_dict[key_name] = tournament_scorer_dict.get(key_name, 0) + 1
            
    if g2 > 0:
        scorers = np.random.choice(t2["PLAYERS"], size=g2, p=t2["PLAYER_WEIGHTS"])
        for scorer in scorers:
            key_name = f"{team2} (Squad Goal)" if scorer == "Other Team Goal" else scorer
            tournament_scorer_dict[key_name] = tournament_scorer_dict.get(key_name, 0) + 1
    
    if g1 != g2:
        return team1 if g1 > g2 else team2
        
    t1_weight = 0.54 if t1["POINTS"] > t2["POINTS"] else 0.46
    return team1 if np.random.rand() < t1_weight else team2

def run_single_tournament(tournament_scorer_dict):
    q1 = simulate_poisson_match("Argentina", "Portugal", tournament_scorer_dict)
    q2 = simulate_poisson_match("France", "Germany", tournament_scorer_dict)
    q3 = simulate_poisson_match("Spain", "Netherlands", tournament_scorer_dict)
    q4 = simulate_poisson_match("England", "Brazil", tournament_scorer_dict)
    
    sf1 = simulate_poisson_match(q1, q2, tournament_scorer_dict)
    sf2 = simulate_poisson_match(q3, q4, tournament_scorer_dict)
    
    return simulate_poisson_match(sf1, sf2, tournament_scorer_dict)

# 4. EXECUTION TRIGGER
if st.button("🚀 Run AI Tournament Simulation"):
    championship_counts = {team: 0 for team in TEAM_METRICS.keys()}
    golden_boot_wins = {}  # Tracks how many times a player wins the golden boot across simulations
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    update_step = max(1, num_simulations // 10)
    
    # Run loop
    for i in range(num_simulations):
        # Fresh tracking dictionary for this single tournament iteration
        current_tournament_goals = {}
        
        winner = run_single_tournament(current_tournament_goals)
        championship_counts[winner] += 1
        
        # Determine Golden Boot winner for this single tournament run
        if current_tournament_goals:
            max_goals = max(current_tournament_goals.values())
            top_scorers = [player for player, goals in current_tournament_goals.items() if goals == max_goals]
            
            # Handle ties by dividing the Golden Boot victory proportionally
            for player in top_scorers:
                golden_boot_wins[player] = golden_boot_wins.get(player, 0) + (1 / len(top_scorers))
        
        if i % update_step == 0:
            progress_bar.progress((i + 1) / num_simulations)
            status_text.text(f"Simulating tournament vector {i:,} of {num_simulations:,}...")
            
    progress_bar.empty()
    status_text.empty()
    
    # --- PROCESS TEAM RESULTS ---
    results_df = pd.DataFrame(list(championship_counts.items()), columns=["Country", "Simulated Wins"])
    results_df["Win Probability"] = (results_df["Simulated Wins"] / num_simulations) * 100
    results_df = results_df.sort_values(by="Win Probability", ascending=False).reset_index(drop=True)
    
    # --- PROCESS GOLDEN BOOT RESULTS ---
    boot_df = pd.DataFrame(list(golden_boot_wins.items()), columns=["Player", "Boot Wins"])
    boot_df["Golden Boot Probability"] = (boot_df["Boot Wins"] / num_simulations) * 100
    boot_df = boot_df.sort_values(by="Golden Boot Probability", ascending=False).reset_index(drop=True)
    
    # Filter out generic team squad markers from displaying as individual player winners
    player_boot_df = boot_df[~boot_df["Player"].str.contains("Squad Goal")].reset_index(drop=True)
    
    # --- DISPLAY UI RESULTS ---
    st.subheader("📊 Tournament Victory Probabilities")
    st.dataframe(results_df.style.format({"Win Probability": "{:.2f}%"}), use_container_width=True)
    
    # Display Top Team
    top_team = results_df.iloc[0]["Country"]
    top_team_prob = results_df.iloc[0]["Win Probability"]
    st.success(f"🏆 Trophy favorite: **{top_team}** with a **{top_team_prob:.2f}%** chance.")
    
    # Display Golden Boot Standings
    st.subheader("🥾 Golden Boot Winner Probabilities")
    st.dataframe(player_boot_df.style.format({"Golden Boot Probability": "{:.2f}%"}), use_container_width=True)
    
    # Display Top Player
    top_player = player_boot_df.iloc[0]["Player"]
    top_player_prob = player_boot_df.iloc[0]["Golden Boot Probability"]
    st.info(f"⚽ Golden Boot favorite: **{top_player}** with a **{top_player_prob:.2f}%** probability of claiming the award.")
    
    # Chart visual
    st.bar_chart(data=player_boot_df.head(10), x="Player", y="Golden Boot Probability")
