import random

# 1. DEFINE NATION RATINGS (FC 26 Estimated Database Logic)
# ATT: Attacking, MID: Midfield, DEF: Defensive depth. 
# Form Modifier represents recent competitive momentum.
NATIONS = {
    "Spain":      {"ATT": 87, "MID": 91, "DEF": 86, "FORM": 1.05},
    "France":     {"ATT": 92, "MID": 88, "DEF": 89, "FORM": 1.02},
    "Argentina":  {"ATT": 88, "MID": 86, "DEF": 84, "FORM": 1.00},
    "England":    {"ATT": 89, "MID": 87, "DEF": 85, "FORM": 0.98},
    "Germany":    {"ATT": 86, "MID": 88, "DEF": 84, "FORM": 1.01},
    "Portugal":   {"ATT": 87, "MID": 87, "DEF": 83, "FORM": 1.00},
    "Brazil":     {"ATT": 88, "MID": 84, "DEF": 86, "FORM": 0.97},
    "Netherlands":{"ATT": 83, "MID": 85, "DEF": 85, "FORM": 0.99},
}

def calculate_match_score(team1, team2):
    """Calculates a realistic match outcome using attributes and RNG."""
    t1_data = NATIONS[team1]
    t2_data = NATIONS[team2]
    
    # Base power score combining midfield control and attacking threat against opponent defense
    t1_power = ((t1_data["MID"] * 0.6) + (t1_data["ATT"] * 0.4) - (t2_data["DEF"] * 0.3)) * t1_data["FORM"]
    t2_power = ((t2_data["MID"] * 0.6) + (t2_data["ATT"] * 0.4) - (t1_data["DEF"] * 0.3)) * t2_data["FORM"]
    
    # Volatility / Luck factor (Simulates real-world randomness and referee decisions)
    t1_rng = random.uniform(0.8, 1.3)
    t2_rng = random.uniform(0.8, 1.3)
    
    t1_final = t1_power * t1_rng
    t2_final = t2_power * t2_rng
    
    # Convert numerical power scores into soccer goals
    t1_goals = int(max(0, (t1_final - 40) / 10))
    t2_goals = int(max(0, (t2_final - 40) / 10))
    
    # Resolve Deadlocks for Knockout Matches
    if t1_goals == t2_goals:
        # 30% chance a team wins in extra time, otherwise it goes to a penalty shootout coin-flip
        if random.random() > 0.7:
            if random.choice([True, False]): t1_goals += 1
            else: t2_goals += 1
            return t1_goals, t2_goals, "AET"
        else:
            return t1_goals, t2_goals, "PEN"
            
    return t1_goals, t2_goals, "FT"

def run_knockout_fixture(team1, team2):
    """Executes the match and formats the text printout."""
    g1, g2, condition = calculate_match_score(team1, team2)
    
    if condition == "FT":
        suffix = ""
    elif condition == "AET":
        suffix = " (After Extra Time)"
    else:
        # Simulate a dramatic penalty shootout score
        p1, p2 = (5, random.randint(3, 4)) if random.choice([True, False]) else (random.randint(3, 4), 5)
        suffix = f" (PENS: {p1}-{p2})"
        
    winner = team1 if (g1 > g2 or (condition == "PEN" and p1 == 5)) else team2
    print(f"   {team1} {g1} - {g2} {team2}{suffix}")
    return winner

# 2. RUN THE TOURNAMENT BRACKET
print("🏆 --- SIMULATED WORLD CUP KNOCKOUTS --- 🏆\n")

print("▶️ QUARTERFINALS:")
q1_winner = run_knockout_fixture("Argentina", "Portugal")
q2_winner = run_knockout_fixture("France", "Germany")
q3_winner = run_knockout_fixture("Spain", "Netherlands")
q4_winner = run_knockout_fixture("England", "Brazil")

print("\n▶️ SEMIFINALS:")
sf1_winner = run_knockout_fixture(q1_winner, q2_winner)
sf2_winner = run_knockout_fixture(q3_winner, q4_winner)

print("\n👑 THE WORLD CUP FINAL:")
champion = run_knockout_fixture(sf1_winner, sf2_winner)

print(f"\n🎉 {champion.upper()} HAS WON THE WORLD CUP! 🎉")
