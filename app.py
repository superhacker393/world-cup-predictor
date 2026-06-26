import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. THE TRAINING DATA (Teaching the AI how football matches work)
# We provide historical match data including Team ratings and who won (0=Draw, 1=Team1 Win, 2=Team2 Win)
training_data = [
    {"t1_att": 92, "t1_mid": 88, "t1_def": 89, "t2_att": 88, "t2_mid": 86, "t2_def": 84, "result": 1}, # Fr vs Ar (Fr Win)
    {"t1_att": 87, "t1_mid": 91, "t1_def": 86, "t2_att": 89, "t2_mid": 87, "t2_def": 85, "result": 1}, # Sp vs En (Sp Win)
    {"t1_att": 86, "t1_mid": 88, "t1_def": 84, "t2_att": 92, "t2_mid": 88, "t2_def": 89, "result": 2}, # Ge vs Fr (Fr Win)
    {"t1_att": 88, "t1_mid": 86, "t1_def": 84, "t2_att": 87, "t2_mid": 87, "t2_def": 83, "result": 1}, # Ar vs Po (Ar Win)
    {"t1_att": 89, "t1_mid": 87, "t1_def": 85, "t2_att": 88, "t2_mid": 84, "t2_def": 86, "result": 0}, # En vs Br (Draw)
    {"t1_att": 87, "t1_mid": 91, "t1_def": 86, "t2_att": 83, "t2_mid": 85, "t2_def": 85, "result": 1}, # Sp vs Ne (Sp Win)
    {"t1_att": 88, "t1_mid": 84, "t1_def": 86, "t2_att": 92, "t2_mid": 88, "t2_def": 89, "result": 2}, # Br vs Fr (Fr Win)
    {"t1_att": 86, "t1_mid": 88, "t1_def": 84, "t2_att": 87, "t2_mid": 91, "t2_def": 86, "result": 2}, # Ge vs Sp (Sp Win)
]

df = pd.DataFrame(training_data)
X_train = df[["t1_att", "t1_mid", "t1_def", "t2_att", "t2_mid", "t2_def"]]
y_train = df["result"]

# 2. TRAIN THE AI MODEL
# Random Forest uses an ensemble of decision trees to calculate mathematical probabilities [1]
ai_model = RandomForestClassifier(n_estimators=100, random_state=42)
ai_model.fit(X_train, y_train)

# 3. NATION DATABASE FOR TARGET TOURNAMENT
NATIONS = {
    "Spain":       {"ATT": 87, "MID": 91, "DEF": 86},
    "France":      {"ATT": 92, "MID": 88, "DEF": 89},
    "Argentina":   {"ATT": 88, "MID": 86, "DEF": 84},
    "England":     {"ATT": 89, "MID": 87, "DEF": 85},
    "Germany":     {"ATT": 86, "MID": 88, "DEF": 84},
    "Portugal":    {"ATT": 87, "MID": 87, "DEF": 83},
    "Brazil":      {"ATT": 88, "MID": 84, "DEF": 86},
    "Netherlands": {"ATT": 83, "MID": 85, "DEF": 85},
}

def ai_predict_match(team1, team2):
    """Feeds team metrics into the trained AI model to predict a winner."""
    t1, t2 = NATIONS[team1], NATIONS[team2]
    
    # Create the data shape the AI expects
    match_features = pd.DataFrame([{
        "t1_att": t1["ATT"], "t1_mid": t1["MID"], "t1_def": t1["DEF"],
        "t2_att": t2["ATT"], "t2_mid": t2["MID"], "t2_def": t2["DEF"]
    }])
    
    # AI predicts probabilities for [Draw, Team 1 Win, Team 2 Win]
    probabilities = ai_model.predict_proba(match_features)[0]
    
    # Introduce an intelligent random choice based strictly on the AI's calculated probabilities
    # This ensures heavy favorites usually win, but realistic upsets can still happen
    outcomes = [f"{team1} (via tiebreaker)", team1, team2]
    winner = np.random.choice(outcomes, p=probabilities)
    
    print(f"   🤖 AI Prediction: {team1} vs {team2} ➔ Winner: {winner}")
    return winner

# 4. RUN THE AI TOURNAMENT
print("🤖 --- AI-POWERED WORLD CUP SIMULATION --- 🤖\n")

print("▶️ QUARTERFINALS:")
q1 = ai_predict_match("Argentina", "Portugal")
q2 = ai_predict_match("France", "Germany")
q3 = ai_predict_match("Spain", "Netherlands")
q4 = ai_predict_match("England", "Brazil")

print("\n▶️ SEMIFINALS:")
sf1 = ai_predict_match(q1, q2)
sf2 = ai_predict_match(q3, q4)

print("\n👑 THE WORLD CUP FINAL:")
champion = ai_predict_match(sf1, sf2)

print(f"\n🎉 THE AI PREDICTS {champion.upper()} TO WIN THE WORLD CUP! 🎉")
