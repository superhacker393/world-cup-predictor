import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

# ==============================================================================
# 1. CORE MACHINE LEARNING ENGINE (Poisson Regression Engine)
# ==============================================================================
class WorldCupPredictor:
    def __init__(self):
        self.team_parameters = {}
        self.intercept = 0.0
        self.teams = []
        
    def _poisson_loss(self, params, df, teams):
        """Calculates negative log-likelihood for Poisson regression."""
        team_map = {team: i for i, team in enumerate(teams)}
        num_teams = len(teams)
        
        # Split flat params array into attack, defense, and baseline intercept
        att = params[:num_teams]
        df_style = params[num_teams:2*num_teams]
        intercept = params[2*num_teams]
        
        log_likelihood = 0
        for _, row in df.iterrows():
            if row['home_team'] not in team_map or row['away_team'] not in team_map:
                continue
            home_idx = team_map[row['home_team']]
            away_idx = team_map[row['away_team']]
            
            # Predict lambda (expected goals) using exponential link function
            home_lambda = np.exp(att[home_idx] + df_style[away_idx] + intercept)
            away_lambda = np.exp(att[away_idx] + df_style[home_idx])
            
            # Sum up the log probabilities of the actual match outcomes
            log_likelihood += poisson.logpmf(row['home_score'], home_lambda)
            log_likelihood += poisson.logpmf(row['away_score'], away_lambda)
            
        return -log_likelihood

    def fit(self, match_history_df):
        """Trains the model on historical match data to isolate team parameters."""
        self.teams = sorted(list(set(match_history_df['home_team']).union(set(match_history_df['away_team']))))
        num_teams = len(self.teams)
        
        # Initial guesses: attack=0.1, defense=-0.1, intercept=0.2
        init_params = np.concatenate([np.repeat(0.1, num_teams), np.repeat(-0.1, num_teams), [0.2]])
        
        # Constraint: Sum of attack strengths must equal 0 to ensure mathematical stability
        constraints = [{'type': 'eq', 'fun': lambda x: sum(x[:num_teams])}]
        
        # Optimize parameters using Sequential Least Squares Programming (SLSQP)
        res = minimize(self._poisson_loss, init_params, args=(match_history_df, self.teams), 
                       method='SLSQP', constraints=constraints)
        
        # Store trained parameters map
        for i, team in enumerate(self.teams):
            self.team_parameters[team] = {
                'attack': res.x[i],
                'defense': res.x[num_teams + i]
            }
        self.intercept = res.x[-1]

    def predict_match(self, home_team, away_team):
        """Predicts exact score probabilities and outputs overall match probabilities."""
        if home_team not in self.team_parameters or away_team not in self.team_parameters:
            return 0.33, 0.34, 0.33 # Default safe baseline fallback split
            
        home_lambda = np.exp(self.team_parameters[home_team]['attack'] + self.team_parameters[away_team]['defense'] + self.intercept)
        away_lambda = np.exp(self.team_parameters[away_team]['attack'] + self.team_parameters[home_team]['defense'])
        
        # Create score matrix grid up to 10 goals to catch extreme outlier metrics
        max_goals = 10
        team1_probs = poisson.pmf(range(max_goals), home_lambda)
        team2_probs = poisson.pmf(range(max_goals), away_lambda)
        
        score_matrix = np.outer(team1_probs, team2_probs)
        
        # Sum matrix segments for standard match outcomes
        home_win_prob = np.sum(np.tril(score_matrix, -1))
        draw_prob = np.sum(np.diag(score_matrix))
        away_win_prob = np.sum(np.triu(score_matrix, 1))
        
        return home_win_prob, draw_prob, away_win_prob


# ==============================================================================
# 2. STREAMLIT USER INTERFACE LAYOUT
# ==============================================================================
st.set_page_config(page_title="AI World Cup Predictor", page_icon="⚽", layout="wide")

st.title("⚽ Advanced AI World Cup Predictor Engine")
st.write("Leveraging Poisson Regression and historical performance data to simulate match outcomes.")

# Dataset Ingestion Loader
@st.cache_data
def load_historical_data():
    # Synthetic dataset for instantiation. Replace with your custom web-scraper / CSV data pipeline.
    data = {
        'home_team': ['Argentina', 'France', 'Brazil', 'England', 'Croatia', 'Morocco', 'Argentina', 'Spain', 'USA', 'Mexico', 'Germany', 'Italy'],
        'away_team': ['France', 'Brazil', 'England', 'Croatia', 'Morocco', 'Spain', 'Croatia', 'France', 'Mexico', 'Germany', 'Italy', 'USA'],
        'home_score':,
        'away_score': [3, 0, 2, 1, 0, 2, 0, 2, 1, 2, 2, 1]
    }
    return pd.DataFrame(data)

# Process and fit data
df = load_historical_data()
predictor = WorldCupPredictor()
predictor.fit(df)

# Responsive Column Architecture
col1, col2 = st.columns(2)

with col1:
    st.subheader("Select Matchup")
    all_teams = predictor.teams
    
    team_a = st.selectbox("Home / Team A", all_teams, index=0)
    team_b = st.selectbox("Away / Team B", all_teams, index=1)

with col2:
    st.subheader("Prediction Probabilities")
    if team_a == team_b:
        st.error("Please select two different teams to simulate a match prediction.")
    else:
        # Fire prediction engine calculations
        win, draw, loss = predictor.predict_match(team_a, team_b)
        
        # Data Metrics Display Cards
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{team_a} Win", f"{win*100:.1f}%")
        c2.metric("Draw Probability", f"{draw*100:.1f}%")
        c3.metric(f"{team_b} Win", f"{loss*100:.1f}%")
        
        # Dynamic Interactive Chart Component
        chart_data = pd.DataFrame({
            'Outcome': [team_a, 'Draw', team_b],
            'Probability (%)': [win*100, draw*100, loss*100]
        })
        st.bar_chart(chart_data, x='Outcome', y='Probability (%)')

# Sidebar Contextual Module
st.sidebar.markdown("""
### 🧠 Model Methodology
This AI engine runs using a **Maximum Likelihood Estimation (MLE)** framework to isolate individual team offensive and defensive coefficients. It natively accounts for:
* Structural team goal scoring frequencies
* Cross-over defensive resistance pairings

### 🚀 Deployment Notice
Save this entire block as `app.py`. Push it to GitHub along with a `requirements.txt` containing `streamlit`, `pandas`, `numpy`, and `scipy` to go live immediately on Streamlit Cloud!
""")
