# app.py

import streamlit as st
import pandas as pd

from config import APP_NAME
from api import FootballAPI
from ratings import RatingEngine
from simulator import TournamentSimulator
from visuals import (
    probability_chart,
    bracket_chart,
    team_strength_chart,
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏆",
    layout="wide"
)


# --------------------------------------------------
# Load services
# --------------------------------------------------

@st.cache_resource
def load_model():

    api = FootballAPI(
        api_key=st.secrets["FOOTBALL_API_KEY"]
    )

    matches = api.get_historical_matches(
        seasons=5
    )

    ratings = RatingEngine(
        matches
    )

    ratings.fit()

    return api, ratings


api, ratings = load_model()


# --------------------------------------------------
# Simulator
# --------------------------------------------------

@st.cache_resource
def load_simulator():

    fixtures = api.get_world_cup_bracket()

    return TournamentSimulator(
        ratings=ratings,
        fixtures=fixtures
    )


simulator = load_simulator()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🏆 2026 World Cup Simulator")

st.caption(
    """
    Dynamic football forecasting model:
    API data · Elo updates · Dixon-Coles scoring ·
    venue adjustment · penalty model · Monte Carlo simulation
    """
)


# --------------------------------------------------
# Tabs
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Tournament",
        "⚔️ Match Predictor",
        "🗓 Bracket",
        "🔬 Model"
    ]
)


# --------------------------------------------------
# Tournament simulation
# --------------------------------------------------

with tab1:

    st.subheader(
        "Championship probabilities"
    )


    simulations = st.slider(
        "Number of simulations",
        1000,
        100000,
        25000,
        step=5000
    )


    if st.button(
        "Run World Cup Simulation",
        type="primary"
    ):

        with st.spinner(
            "Running tournaments..."
        ):

            results = simulator.run(
                simulations
            )


        df = (
            pd.DataFrame(results)
            .sort_values(
                "championship_probability",
                ascending=False
            )
        )


        winner = df.iloc[0]


        st.success(
            f"""
            🏆 Predicted champion:

            **{winner.team}**

            Probability:
            {winner.championship_probability:.2%}
            """
        )


        st.dataframe(
            df,
            use_container_width=True
        )


        probability_chart(df)



# --------------------------------------------------
# Match predictor
# --------------------------------------------------

with tab2:

    st.subheader(
        "Single match forecast"
    )


    teams = ratings.team_list()


    team_a = st.selectbox(
        "Team A",
        teams
    )


    team_b = st.selectbox(
        "Team B",
        teams,
        index=1
    )


    venue = st.text_input(
        "Venue",
        "Neutral"
    )


    if st.button(
        "Predict Match"
    ):


        prediction = simulator.predict_match(
            team_a,
            team_b,
            venue
        )


        c1,c2,c3 = st.columns(3)


        c1.metric(
            team_a,
            f"{prediction.team_a_win:.1%}"
        )


        c2.metric(
            "Draw after 90",
            f"{prediction.draw:.1%}"
        )


        c3.metric(
            team_b,
            f"{prediction.team_b_win:.1%}"
        )


        st.write(
            "Expected goals"
        )


        st.json(
            {
                team_a:
                    prediction.xg_a,

                team_b:
                    prediction.xg_b,

                "Most likely score":
                    prediction.score
            }
        )



# --------------------------------------------------
# Bracket
# --------------------------------------------------

with tab3:

    st.subheader(
        "Live World Cup bracket"
    )


    bracket = simulator.current_bracket()


    st.dataframe(
        bracket,
        use_container_width=True
    )


    bracket_chart(
        bracket
    )



# --------------------------------------------------
# Model explanation
# --------------------------------------------------

with tab4:

    st.markdown(
        """
## Statistical model

### Team strength

Ratings are estimated from historical matches:

- attack strength
- defensive strength
- Elo rating
- recent form


### Goal model

Uses:

- Dixon-Coles correction
- Poisson goal distribution
- time-decayed matches


### Match adjustments

Includes:

- venue
- travel
- rest days
- injuries
- expected lineup


### Knockout simulation

Includes:

- extra time
- team-specific penalty probability
- Monte Carlo tournament sampling
        """
    )
