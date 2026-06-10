# ⚽ GoalAnalytics — World Cup 2026 Prediction Engine

> Module 1 of 3 · Elo + Bivariate Poisson + Monte Carlo simulation  
> Built live during the tournament · tracking real-time accuracy

**[→ Live Dashboard](https://goal-analytics.streamlit.app)**  
**[→ Substack](https://substack.com/@nithinnarla)**

---

## What this is

GoalAnalytics runs 10,000 Monte Carlo simulations of the 2026 FIFA World Cup using:

- **Elo ratings** calibrated from FIFA/Coca-Cola World Rankings (June 2026)
- **Bivariate Poisson model** for scoreline probabilities (Dixon-Coles methodology)
- **Home advantage** modelling for Mexico, USA, and Canada (+100 Elo)
- **Live accuracy tracking** — enter actual results and watch the model's Brier score update in real time

This is Module 1. Modules 2 (Player2Vec squad embeddings) and 3 (ILP fantasy optimizer) ship post-tournament.

---

## Project structure

```
goal-analytics/
├── data/
│   ├── teams.py        # 48 teams, Elo ratings, FIFA rankings
│   └── fixtures.py     # All 72 group-stage fixtures with dates/venues
├── models/
│   ├── elo.py          # Elo win probability + expected goals
│   ├── poisson.py      # Bivariate Poisson scoreline distribution
│   └── monte_carlo.py  # Full tournament simulation (10,000 runs)
├── dashboard/
│   └── app.py          # Streamlit dashboard (4 tabs)
├── run.py              # CLI — print predictions to stdout
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/nithinnarla/goal-analytics
cd goal-analytics
pip install -r requirements.txt

# Print predictions to terminal
python run.py

# Launch interactive dashboard
streamlit run dashboard/app.py
```

---

## Dashboard tabs

| Tab | What it shows |
|-----|---------------|
| 🏆 Win Probabilities | Tournament win % for all 48 teams, bar chart + full table |
| 📊 Group Predictions | Predicted standings per group, P(1st/2nd/3rd/4th) |
| 📍 Match Predictor | Any two teams → win/draw/loss %, xG, scoreline heatmap |
| 📈 Live Tracker | Enter actual results → see model accuracy update in real time |

---

## Model methodology

### Elo ratings
Pre-tournament Elo values are hand-calibrated from the June 2026 FIFA/Coca-Cola World Rankings and recent competitive results. Argentina (#1, 2100), Spain (#2, 2060), France (#3, 2050), England (#4, 2010). Host nations receive +100 Elo when playing at home venues.

### Poisson model
Expected goals for each team derived from Elo difference:
```
λ_home = avg_goals × 10^(Δelo / 800)
λ_away = avg_goals / 10^(Δelo / 800)
```
Scoreline probabilities computed analytically from the joint Poisson distribution up to 8 goals per team.

### Monte Carlo simulation
10,000 independent full-tournament simulations. Each group match sampled from the Poisson scoreline distribution. Group standings determined by FIFA tiebreaker rules (pts → GD → GF → H2H → random). Best 8 third-placed teams selected by points. Knockout draws resolved by penalty shootout (50/50 with marginal Elo adjustment).

---

## Accuracy tracking

The Live Tracker tab lets you enter actual match results and computes:
- **Result accuracy** — did the model pick the right W/D/L?
- **Exact score accuracy** — did the model's most likely score hit?
- Running totals across all entered matches

The model updates all win probabilities based on known results, so knockout-stage predictions improve as the group stage plays out.

---

## Roadmap

- [x] Module 1: Elo + Poisson + Monte Carlo match predictor
- [ ] Module 2: Player2Vec squad embeddings (post-tournament)
- [ ] Module 3: ILP fantasy squad optimizer (pre-Euro 2027)

---

## License

MIT · Built by [@nithinnarla](https://substack.com/@nithinnarla)
