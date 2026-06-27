# 2026 World Cup Simulator

Poisson goal model · Elo-calibrated team ratings · Real R32 bracket · Monte Carlo

## What it does

- **Tournament simulation** — runs up to 50,000 Monte Carlo simulations of the full 2026 WC bracket (R32 → R16 → QF → SF → Final) using Poisson-distributed goal draws
- **Head-to-head** — calculates 90-minute win/draw/loss probabilities and expected goals for any two teams
- **R32 bracket** — shows all confirmed Round of 32 fixtures with model win percentages
- **Model explainer** — transparent breakdown of how ratings and the Poisson engine work

## How the model works

Each match draws goals from `Poisson(xG)`:

```
xG_A = attack_A × defense_B × (1 + home_boost)
xG_B = attack_B × defense_A
```

Draws go to extra time (33% xG), then penalties (50/50). The bracket follows the real 2026 WC structure. Team ratings are calibrated from World Football Elo ratings and group-stage results.

## Deploy on Streamlit Community Cloud

1. **Fork or push this repo to GitHub**
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/wc2026-simulator.git
   git push -u origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub

3. **Click "New app"**, choose your repo, branch `main`, and set the main file path to `app.py`

4. **Click Deploy** — Streamlit will install requirements and launch the app

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## File structure

```
wc2026-simulator/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Updating ratings mid-tournament

Find the `TEAMS` dict in `app.py`. Each team has:
- `atk` — expected goals scored vs an average opponent (higher = more attacking)
- `def` — defense multiplier applied to opponent attack (lower = better defense)
- `elo` — approximate World Football Elo rating (used for display only)

After each round, you can increase `atk` for teams in form or lower `def` for teams conceding heavily.

## Limitations

This model uses the right architecture (Poisson goals, separate attack/defense, proper bracket). What professional models add on top:
- Maximum-likelihood estimation of ratings from thousands of historical matches
- Live injury and squad data
- Continuous retraining after every result
- Team-specific penalty records
- Shot-level xG from StatsBomb/Opta rather than Elo-derived estimates
