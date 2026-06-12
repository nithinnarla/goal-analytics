# ⚽ Goal Analytics: A Multi-Model Forecasting System for the 2026 FIFA World Cup

**Module 1 of 3 · Elo + Poisson + Logistic Regression + Random Forest + XGBoost + Monte Carlo simulation**
**Built and validated live during the tournament**

**[→ Live Dashboard](https://goal-analytics.streamlit.app)** · **[→ Substack](https://substack.com/@nithinnarla)**

---

## Abstract

GoalAnalytics forecasts match outcomes and tournament progression for the 2026 FIFA World Cup (48 teams, 12 groups, 104 matches under the expanded format). It combines a classical Elo rating system, an independent-margins Poisson scoreline model, and three supervised classifiers — logistic regression, random forest, and gradient-boosted trees (XGBoost) — trained on a shared six-feature representation of each fixture. Tournament-level quantities (group standings, knockout progression, title probabilities) are estimated by Monte Carlo simulation over the real WC2026 bracket: 10,000 runs for headline win probabilities, 5,000 for group-position breakdowns. All six approaches are evaluated against the *actual* results of the 2018 and 2022 World Cups using **point-in-time training** — each model is refit using only data available before that tournament's opening match, so no model is graded on data it has already seen — and scored on accuracy and multi-class Brier score. Everything is exposed through an interactive Streamlit dashboard, including a live FIFA World Ranking comparison panel and an in-app backtest runner.

---

## 1. Introduction

International football outcomes are notoriously hard to predict: a 48-team, single-elimination-heavy tournament compresses a season's worth of variance into a month. GoalAnalytics treats this as a forecasting problem with two layers:

1. **Match-level models** — given two teams (and venue context), estimate P(home win), P(draw), P(away win), and the most likely scoreline.
2. **Tournament-level simulation** — repeatedly sample match-level outcomes across the full 104-match bracket to estimate each team's probability of reaching the Round of 32, Round of 16, quarter-finals, semi-finals, the final, and the title.

Rather than committing to a single "best" model, the project deliberately runs six approaches side by side — one classical rating system, one analytical scoreline model, and three machine-learned classifiers — and reports how each one would have performed on the last two World Cups. The goal is calibration and transparency, not a single leaderboard winner.

This is **Module 1** of a three-part project. Module 2 (player-level squad embeddings) and Module 3 (an integer linear programming fantasy-squad optimizer) are planned for after the tournament.

---

## 2. Data

| Source | Used for | Notes |
|---|---|---|
| [`martj42/international_results`](https://github.com/martj42/international_results) (`results.csv`, `shootouts.csv`) | Elo rating history, recent-form features, training data for Logistic Regression / Random Forest / XGBoost, and the 2018/2022 backtest ground truth | Full international match history, fetched live via `data/historical.py`; ~25k matches in the "recent era" (2010–present) slice |
| `data/teams.py` | Pre-tournament Elo ratings, FIFA ranks, confederations, and group assignments for all 48 WC2026 teams | Hand-calibrated from the June 2026 FIFA/Coca-Cola World Ranking and recent results — **separate from** the Elo ratings `compute_elo_ratings()` derives from the historical match log |
| `data/fixtures.py` | All 72 group-stage fixtures (dates, venues, cities) | |
| `data/knockout_fixtures.py` | The real WC2026 knockout bracket — Matches 73–104 (Round of 32 → Round of 16 → quarter-finals → semi-finals → final + 3rd-place play-off), including FIFA's literal Annex C "best third-placed team" assignment table (495/495 verified) | |
| `data/fifa_rankings.py` | Live FIFA World Ranking, scraped from a public mirror | **Informational only** — does not feed Elo, Poisson, or Monte Carlo |

Team-name aliasing (`data/historical.py`) reconciles naming differences between the `martj42` dataset and `data/teams.py` (e.g. "USA" ↔ "United States", "South Korea" ↔ "Korea Republic"), with accent-stripping for names like "Curaçao" / "Türkiye" / "Côte d'Ivoire".

---

## 3. Feature Engineering

Every match-level model (Logistic Regression, Random Forest, XGBoost) is trained on the same six-feature vector, built by `models/features.py`:

| Feature | Definition |
|---|---|
| `elo_diff` | `home_elo − away_elo` |
| `home_advantage` | `1` if the match is at a home venue (not neutral), else `0` |
| `form_diff` | home recent form (avg. points/game) − away recent form |
| `scored_diff` | home avg. goals scored − away avg. goals scored |
| `conceded_diff` | away avg. goals conceded − home avg. goals conceded (positive = home team faces a leakier defence) |
| `elo_sq_diff` | `elo_diff² / 10,000` (captures non-linearity in Elo gap) |

Labels are encoded as `2 = home win`, `1 = draw`, `0 = away win`.

---

## 4. Models

### 4.1 Elo Ratings (`models/elo.py`)

Standard Elo expected-score formula:

```
E_home = 1 / (1 + 10^((R_away − R_home) / 400))
```

Win/draw/loss probabilities are derived by scaling the two-outcome Elo expectation by `(1 − draw_rate)` and allocating the remainder to a draw, where the draw rate is closeness-adjusted (closer matches draw more often, capped at 38%) around an empirical World Cup baseline of **22%** (1994–2022).

Two distinct home-advantage mechanisms are used in the codebase:
- **WC2026 host nations**: Mexico, USA, and Canada receive **+100 Elo** when playing in one of their own host cities (`data/teams.py:get_elo`).
- **General historical home advantage**: a **+100 Elo** adjustment is applied to the home team for any non-neutral-venue match when computing ratings, training features, and backtest predictions (`HOME_ADVANTAGE_ELO` in `models/backtest.py`, mirrored in `data/historical.py`).

*References: Elo (1978); Hvattum & Arntzen (2010); eloratings.net.*

### 4.2 Poisson Scoreline Model (`models/poisson.py`)

Expected goals for each side are derived from the Elo difference:

```
λ_home = avg_goals × 10^(Δelo / 800)
λ_away = avg_goals / 10^(Δelo / 800)
```

with `avg_goals = 1.25` (the approximate World Cup average goals-per-team-per-game, 1994–2022). Goals for each team are then modelled as **independent** Poisson distributions:

```
P(score = x–y) = Poisson(x; λ_home) × Poisson(y; λ_away)
```

computed analytically over all scorelines up to 8 goals per side, then renormalised. Match result probabilities (home/draw/away) and the most likely scoreline fall directly out of this joint distribution.

> **Honest framing:** the module's docstring cites Dixon & Coles (1997) and Maher (1982) as the literature this approach draws from, but the *implementation* uses independent Poisson margins, not the full Dixon-Coles bivariate model with its low-score dependence (`τ`) correction. Adding that correction is listed under [Limitations & Future Work](#9-limitations--future-work).

### 4.3 Logistic Regression (`models/logistic.py`)

A calibrated linear baseline: `StandardScaler` + multinomial `LogisticRegression(C=1.0, max_iter=500)` over the six-feature vector, trained on **World Cup matches only** (~900 matches, 1930–2022). Reports 5-fold stratified cross-validation accuracy. Falls back to a closed-form Elo-based estimate if scikit-learn is unavailable or the model hasn't been trained.

### 4.4 Random Forest (`models/random_forest.py`)

`RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)` over the same six features, trained on the **"recent era"** (2010-onward) slice of the full international match history — roughly 25,000 matches, far more signal than the World-Cup-only set. Evaluated with 5-fold stratified cross-validation. Falls back to Elo if scikit-learn is unavailable or there isn't enough recent-era data (`< 200` matches).

### 4.5 XGBoost (`models/xgboost_model.py`)

`XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, objective="multi:softprob", eval_metric="mlogloss", random_state=42)`, trained on the same recent-era slice and feature set as the Random Forest, with the same 5-fold CV reporting and Elo fallback.

### 4.6 Monte Carlo Tournament Simulation (`models/monte_carlo.py`)

Each simulation:

1. Plays all 72 group-stage matches, sampling scorelines from the Elo→Poisson distribution.
2. Ranks each group by **points → goal difference → goals for → random tiebreak** (FIFA's head-to-head and disciplinary tiebreakers are *not* implemented — see [Limitations](#9-limitations--future-work)).
3. Selects the top 2 from each of the 12 groups plus the **best 8 third-placed teams** (ranked by the same points → GD → GF → random criteria) — 32 qualifiers.
4. Resolves the Round-of-32 pairing against the **real WC2026 bracket** (`data/knockout_fixtures.py`, Matches 73–104), using FIFA's literal Annex C table to assign third-placed teams (avoiding a team facing its own group's anchor).
5. Walks the bracket through Round of 16 → quarter-finals → semi-finals → final (+ 3rd-place play-off). Drawn knockout matches go to a penalty shootout modelled as 50% ± an Elo-proportional edge, capped to [30%, 70%].

Two simulation counts are used for two different purposes:
- **10,000 runs**, seeded (`seed=42`), for the headline tournament win probabilities (`win_probabilities()`, used by the dashboard's Win Probabilities tab and `run.py`).
- **5,000 runs** for group-stage finishing-position probabilities (P(1st/2nd/3rd/4th), dashboard's Group Predictions tab) and as the empirical Monte Carlo cross-check inside the backtest (below).

---

## 5. Validation: Point-in-Time Backtesting (`models/backtest.py`)

To check whether any of these six approaches is actually useful, each one is re-evaluated against the **real results of the 2018 and 2022 World Cups**.

**Why point-in-time training matters.** The dashboard's live models are trained on the *entire* historical dataset — which already contains the 2018 and 2022 results. Scoring those models on 2018/2022 directly would be in-sample evaluation. Instead, `_train_point_in_time()` builds a **separate copy** of every model using only matches strictly before that tournament's opening kickoff:

| Tournament | Cutoff date | Cutoff match |
|---|---|---|
| 2018 | 2018-06-14 | Russia 5–0 Saudi Arabia |
| 2022 | 2022-11-20 | Qatar 0–2 Ecuador |

- Elo ratings and recent-form lookups are replayed only through the cutoff.
- Logistic Regression is retrained on prior-World-Cup-only matches available as of the cutoff.
- Random Forest and XGBoost are retrained on the recent-era (2010+) slice available as of the cutoff.

**Models compared**: Elo (W/D/L), Elo + Poisson, Monte Carlo (5,000 sims/match, `mc_seed=42`), Logistic Regression, Random Forest, XGBoost.

**Metrics**:
- **Accuracy** — fraction of matches where the model's most-likely outcome (argmax of P(home/draw/away)) matches the actual result.
- **Multi-class Brier score** — `Σ (predicted_prob − actual_indicator)²` over {home win, draw, away win}, range **[0, 2]**, lower is better. Rewards calibrated confidence, not just correct picks.

The backtest is run on demand from the dashboard's **🧪 Model Backtest** tab (cached for 24 hours), or programmatically:

```python
from models.backtest import run_full_backtest
results = run_full_backtest(years=(2018, 2022), n_mc_sims=5000)
```

This README intentionally does not hard-code accuracy/Brier numbers — they're computed live against the current historical dataset and are most meaningful viewed match-by-match in the dashboard, where each model's pick is shown alongside the actual result.

---

## 6. Interactive Dashboard (`dashboard/app.py`)

| Tab | What it shows |
|---|---|
| 🏆 Win Probabilities | Tournament win % for all 48 teams (10,000-sim Monte Carlo using Elo + Poisson), bar chart + full progression table (R32/R16/QF/SF/Final/Win), plus a collapsible **live FIFA World Ranking vs. model rank** comparison panel |
| 📊 Group Predictions | Predicted standings per group, P(1st/2nd/3rd/4th) from 5,000-sim Monte Carlo |
| 📍 Match Predictor | Any two teams → win/draw/loss %, expected goals, top scorelines, and a full scoreline probability heatmap |
| 📈 Live Tracker | Enter actual results as the tournament unfolds → see model accuracy update in real time |
| 🗺️ Bracket | Group-stage qualifiers and the projected knockout bracket, with a model cross-check panel comparing the Monte Carlo bracket against the ML models |
| 🧪 Model Backtest | On-demand point-in-time backtest of all six models against the actual 2018 & 2022 World Cups (Section 5) |

---

## 7. Reproducibility / Quickstart

```bash
git clone https://github.com/nithinnarla/goal-analytics
cd goal-analytics
pip install -r requirements.txt

# Print predictions to terminal
python run.py

# Launch interactive dashboard
streamlit run dashboard/app.py

# Run the standalone analysis script (mirrors GoalAnalytics_Analysis.ipynb,
# writes figures to ./figures/)
python GoalAnalytics_Analysis.py

# Run the 2018/2022 backtest from the command line
python -c "from models.backtest import run_full_backtest; print(run_full_backtest())"
```

---

## 8. Project Structure

```
goal-analytics/
├── data/
│   ├── teams.py              # 48 WC2026 teams: groups, Elo, FIFA rank, confederation
│   ├── fixtures.py           # All 72 group-stage fixtures
│   ├── knockout_fixtures.py  # Real WC2026 knockout bracket (Matches 73-104)
│   ├── historical.py         # Fetches/cleans international results, computes Elo + form
│   └── fifa_rankings.py       # Live FIFA World Ranking scraper (informational only)
├── models/
│   ├── elo.py                # Elo win probability + expected goals
│   ├── poisson.py             # Independent-margins Poisson scoreline distribution
│   ├── features.py            # Shared 6-feature vector + label encoding
│   ├── logistic.py            # Logistic Regression (WC-only training set)
│   ├── random_forest.py       # Random Forest (recent-era training set)
│   ├── xgboost_model.py       # XGBoost (recent-era training set)
│   ├── monte_carlo.py         # Full tournament simulation (10,000 / 5,000 runs)
│   └── backtest.py            # Point-in-time backtest vs. 2018 & 2022 World Cups
├── analysis/
│   └── visualizations.py      # Matplotlib figures (winners history, win probs, etc.)
├── dashboard/
│   └── app.py                 # Streamlit dashboard (6 tabs)
├── GoalAnalytics_Analysis.py  # Standalone analysis script
├── run.py                     # CLI — print predictions to stdout
└── requirements.txt
```

---

## 9. Limitations & Future Work

- **Poisson independence assumption.** Goals are modelled as independent Poisson draws. A full Dixon-Coles bivariate adjustment (low-score correlation `τ` term) would better capture the empirical excess of 0-0/1-0/0-1/1-1 results and is the most direct methodological upgrade to `models/poisson.py`.
- **Group tiebreakers are simplified.** Both group standings and "best third-placed team" selection use points → goal difference → goals for → random, omitting FIFA's head-to-head and disciplinary-points tiebreakers. This mainly affects edge cases where multiple teams finish level on points, GD, and GF.
- **Two Elo systems coexist.** The 48 hand-calibrated pre-tournament Elo ratings in `data/teams.py` (used for the live WC2026 simulation) are separate from the Elo ratings `compute_elo_ratings()` derives from the full historical match log (used for ML training and the backtest). They are not reconciled into a single rating system.
- **Live FIFA ranking is decorative.** `data/fifa_rankings.py` surfaces a live/static rank comparison but does not feed back into Elo, Poisson, or any classifier — a natural extension would be to use ranking *deltas* as an additional feature.
- **Backtest sample size.** 2018 and 2022 each contribute a relatively small number of matches; accuracy and Brier-score differences between models should be read as directional, not statistically definitive.
- **No player-level information.** Squad changes, injuries, and suspensions are not modelled. This is the explicit motivation for Module 2 (player embeddings), planned post-tournament.

---

## 10. References

- Elo, A. (1978). *The Rating of Chess Players, Past and Present.*
- Hvattum, L.M. & Arntzen, H. (2010). *Using ELO ratings for match result prediction in association football.* International Journal of Forecasting.
- [eloratings.net](https://www.eloratings.net/) — football Elo methodology.
- Maher, M.J. (1982). *Modelling Association Football Scores.* Statistica Neerlandica.
- Dixon, M.J. & Coles, S.G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical Society, Series C.
- [martj42/international_results](https://github.com/martj42/international_results) — historical international match results dataset.
- FIFA/Coca-Cola World Ranking; live comparison data via [whereig.com](https://www.whereig.com/football/fifa-world-rankings.html).

---

## Roadmap

- [x] Module 1: Elo + Poisson + Logistic Regression + Random Forest + XGBoost + Monte Carlo, with point-in-time backtesting against 2018 & 2022
- [ ] Module 2: Player2Vec squad embeddings (post-tournament)
- [ ] Module 3: ILP fantasy squad optimizer (pre-Euro 2028)

---

## License

MIT · Built by [@nithinnarla](https://substack.com/@nithinnarla)
