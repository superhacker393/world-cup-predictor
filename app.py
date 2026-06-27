import streamlit as st
import numpy as np
import pandas as pd

# 1. WEB APP INTERFACE CONFIGURATION
st.set_page_config(page_title="World Cup AI Predictor", page_icon="🏆", layout="centered")
st.title("🏆 World Cup Monte Carlo AI Predictor")
st.write("This app simulates the World Cup tournament and utilizes player historical scoring weights to predict the Golden Boot winner.")

# 2. GROUND TRUTH DATABASE
TEAM_METRICS = {
    "Argentina": { "POINTS": 1889.06, "OFFENSE": 2.10, "DEFENSE": 0.85, "PLAYERS": ["Lionel Messi", "Lautaro Martínez", "Julián Álvarez", "Argentina (Squad Goal)"], "PLAYER_WEIGHTS": [0.40, 0.30, 0.20, 0.10] },
    "France": { "POINTS": 1887.11, "OFFENSE": 2.15, "DEFENSE": 0.90, "PLAYERS": ["Kylian Mbappé", "Olivier Giroud", "Antoine Griezmann", "France (Squad Goal)"], "PLAYER_WEIGHTS": [0.45, 0.25, 0.20, 0.10] },
    "Spain": { "POINTS": 1856.03, "OFFENSE": 1.95, "DEFENSE": 0.80, "PLAYERS": ["Alvaro Morata", "Lamine Yamal", "Nico Williams", "Spain (Squad Goal)"], "PLAYER_WEIGHTS": [0.35, 0.25, 0.25, 0.15] },
    "England": { "POINTS": 1828.02, "OFFENSE": 1.90, "DEFENSE": 0.88, "PLAYERS": ["Harry Kane", "Bukayo Saka", "Jude Bellingham", "England (Squad Goal)"], "PLAYER_WEIGHTS": [0.45, 0.25, 0.20, 0.10] },
    "Brazil": { "POINTS": 1765.34, "OFFENSE": 1.85, "DEFENSE": 0.92, "PLAYERS": ["Vinicius Jr", "Rodrygo", "Richarlison", "Brazil (Squad Goal)"], "PLAYER_WEIGHTS": [0.35, 0.30, 0.25, 0.10] },
    "Portugal": { "POINTS": 1755.09, "OFFENSE": 1.80, "DEFENSE": 0.95, "PLAYERS": ["Cristiano Ronaldo", "Bruno Fernandes", "Gonçalo Ramos", "Portugal (Squad Goal)"], "PLAYER_WEIGHTS": [0.40, 0.25, 0.25, 0.10] },
    "Netherlands": { "POINTS": 1743.20, "OFFENSE": 1.70, "DEFENSE": 0.91, "PLAYERS": ["Memphis Depay", "Cody Gakpo", "Donyell Malen", "Netherlands (Squad Goal)"], "PLAYER_WEIGHTS": [0.35, 0.35, 0.20, 0.10] },
    "Germany": { "POINTS": 1736.00, "OFFENSE": 1.75, "DEFENSE": 0.96, "PLAYERS": ["Niclas Füllkrug", "Kai Havertz", "Jamal Musiala", "Germany (Squad Goal)"], "PLAYER_WEIGHTS": [0.35, 0.30, 0.25, 0.10] }
}

# Map teams and players to integer IDs for lightning-fast matrix tracking
TEAMS = list(TEAM_METRICS.keys())
TEAM_TO_IDX = {team: i for i, team in enumerate(TEAMS)}

ALL_PLAYERS = []
PLAYER_TO_IDX = {}
player_team_mask = [] # Tracks which team each player belongs to

for t_idx, team in enumerate(TEAMS):
    for p_name in TEAM_METRICS[team]["PLAYERS"]:
        ALL_PLAYERS.append(p_name)
        PLAYER_TO_IDX[p_name] = len(ALL_PLAYERS) - 1
        player_team_mask.append(t_idx)

player_team_mask = np.array(player_team_mask)
num_players = len(ALL_PLAYERS)

# INTERACTIVE SIDEBAR
st.sidebar.header("🛠️ Simulation Controls")
num_simulations = st.sidebar.slider("Number of Tournaments to Simulate", min_value=500, max_value=1000000, value=100000, step=500)

st.sidebar.subheader("Adjust Team Form Modifiers")
form_modifiers = {}
for team in TEAMS:
    form_modifiers[team] = st.sidebar.slider(f"{team} Form Boost", min_value=0.8, max_value=1.2, value=1.0, step=0.05)

# 3. VECTORIZED SIMULATION ENGINE
def simulate_vector_match(team1_ids, team2_ids, player_goals, N):
    """Simulates N matches simultaneously using vectorized array operations"""
    # Fetch team stats based on current match setups
    t1_off = np.array([TEAM_METRICS[TEAMS[i]]["OFFENSE"] * form_modifiers[TEAMS[i]] for i in team1_ids])
    t2_def = np.array([TEAM_METRICS[TEAMS[i]]["DEFENSE"] for i in team2_ids])
    t2_off = np.array([TEAM_METRICS[TEAMS[i]]["OFFENSE"] * form_modifiers[TEAMS[i]] for i in team2_ids])
    t1_def = np.array([TEAM_METRICS[TEAMS[i]]["DEFENSE"] for i in team1_ids])
    
    t1_xg = np.maximum(0.5, t1_off * t2_def)
    t2_xg = np.maximum(0.5, t2_off * t1_def)
    
    # Generate all goals for all N matches instantly
    g1 = np.random.poisson(t1_xg)
    g2 = np.random.poisson(t2_xg)
    
    # Fast player goal distribution
    for i in range(N):
        idx1, idx2 = team1_ids[i], team2_ids[i]
        
        if g1[i] > 0:
            p_indices = np.where(player_team_mask == idx1)[0]
            weights = TEAM_METRICS[TEAMS[idx1]]["PLAYER_WEIGHTS"]
            chosen = np.random.choice(p_indices, size=g1[i], p=weights)
            for cp in chosen:
                player_goals[i, cp] += 1
                
        if g2[i] > 0:
            p_indices = np.where(player_team_mask == idx2)[0]
            weights = TEAM_METRICS[TEAMS[idx2]]["PLAYER_WEIGHTS"]
            chosen = np.random.choice(p_indices, size=g2[i], p=weights)
            for cp in chosen:
                player_goals[i, cp] += 1

    # Resolve winners and handle tie-breaking rules
    winners = np.where(g1 > g2, team1_ids, team2_ids)
    ties = (g1 == g2)
    if np.any(ties):
        t1_pts = np.array([TEAM_METRICS[TEAMS[i]]["POINTS"] for i in team1_ids])
        t2_pts = np.array([TEAM_METRICS[TEAMS[i]]["POINTS"] for i in team2_ids])
        t1_weight = np.where(t1_pts > t2_pts, 0.54, 0.46)
        
        rand_vals = np.random.rand(N)
        tie_winners = np.where(rand_vals < t1_weight, team1_ids, team2_ids)
        winners = np.where(ties, tie_winners, winners)
        
    return winners

# 4. EXECUTION TRIGGER
if st.button("🚀 Run AI Tournament Simulation"):
    with st.spinner(f"Processing {num_simulations:,} parallel universes..."):
        # Matrix to hold player goals: [Simulation Index, Player Index]
        player_goals = np.zeros((num_simulations, num_players), dtype=np.int32)
        
        # Quarterfinals
        init_t1 = np.full(num_simulations, TEAM_TO_IDX["Argentina"])
        init_t2 = np.full(num_simulations, TEAM_TO_IDX["Portugal"])
        q1 = simulate_vector_match(init_t1, init_t2, player_goals, num_simulations)
        
        init_t3 = np.full(num_simulations, TEAM_TO_IDX["France"])
        init_t4 = np.full(num_simulations, TEAM_TO_IDX["Germany"])
        q2 = simulate_vector_match(init_t3, init_t4, player_goals, num_simulations)
        
        init_t5 = np.full(num_simulations, TEAM_TO_IDX["Spain"])
        init_t6 = np.full(num_simulations, TEAM_TO_IDX["Netherlands"])
        q3 = simulate_vector_match(init_t5, init_t6, player_goals, num_simulations)
        
        init_t7 = np.full(num_simulations, TEAM_TO_IDX["England"])
        init_t8 = np.full(num_simulations, TEAM_TO_IDX["Brazil"])
        q4 = simulate_vector_match(init_t7, init_t8, player_goals, num_simulations)
        
        # Semifinals
        sf1 = simulate_vector_match(q1, q2, player_goals, num_simulations)
        sf2 = simulate_vector_match(q3, q4, player_goals, num_simulations)
        
        # Finals
        champions = simulate_vector_match(sf1, sf2, player_goals, num_simulations)
        
        # --- PROCESS CHAMPIONSHIP RESULTS ---
        unique_teams, counts = np.unique(champions, return_counts=True)
        championship_counts = {team: 0 for team in TEAMS}
        for ut, ct in zip(unique_teams, counts):
            championship_counts[TEAMS[ut]] = ct
            
        results_df = pd.DataFrame(list(championship_counts.items()), columns=["Country", "Simulated Wins"])
        results_df["Win Probability"] = (results_df["Simulated Wins"] / num_simulations) * 100
        results_df = results_df.sort_values(by="Win Probability", ascending=False).reset_index(drop=True)

        # --- PROCESS GOLDEN BOOT RESULTS ---
        # Find maximum goals scored per simulation universe
        max_goals_per_sim = np.max(player_goals, axis=1, keepdims=True)
        
        # Prevent division by zero if an entire tournament goes goalless
        max_goals_per_sim = np.where(max_goals_per_sim == 0, 1, max_goals_per_sim) 
        
        # Boolean matrix of who matched the maximum goals in each simulation run
        is_top_scorer = (player_goals == max_goals_per_sim) & (player_goals > 0)
        
        # Handle sharing metrics proportionally for ties
        scorers_per_sim = np.sum(is_top_scorer, axis=1, keepdims=True)
        scorers_per_sim = np.where(scorers_per_sim == 0, 1, scorers_per_sim)
        boot_shares = is_top_scorer / scorers_per_sim
        
        # Aggregate totals across all simulation lines
        total_boot_wins = np.sum(boot_shares, axis=0)
        
        boot_df = pd.DataFrame({
            "Player": ALL_PLAYERS,
            "Boot Wins": total_boot_wins
        })
        boot_df["Golden Boot Probability"] = (boot_df["Boot Wins"] / num_simulations) * 100
        boot_df = boot_df.sort_values(by="Golden Boot Probability", ascending=False).reset_index(drop=True)
        
        # Filter out generic squad markers from final display
        player_boot_df = boot_df[~boot_df["Player"].str.contains("Squad Goal")].reset_index(drop=True)

    # --- DISPLAY UI RESULTS ---
    st.subheader("📊 Tournament Victory Probabilities")
    st.dataframe(results_df.style.format({"Win Probability": "{:.2f}%"}), use_container_width=True)
    
    top_team = results_df.iloc[0]["Country"]
    top_team_prob = results_df.iloc[0]["Win Probability"]
    st.success(f"🏆 Trophy favorite: **{top_team}** with a **{top_team_prob:.2f}%** chance.")

    st.subheader("🥾 Golden Boot Winner Probabilities")
    st.dataframe(player_boot_df.style.format({"Golden Boot Probability": "{:.2f}%"}), use_container_width=True)
    
    top_player = player_boot_df.iloc[0]["Player"]
    top_player_prob = player_boot_df.iloc[0]["Golden Boot Probability"]
    st.info(f"⚽ Golden Boot favorite: **{top_player}** with a **{top_player_prob:.2f}%** probability of claiming the award.")
    
    st.bar_chart(data=player_boot_df.head(10), x="Player", y="Golden Boot Probability")
