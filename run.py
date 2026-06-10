"""
GoalAnalytics — CLI runner
Prints pre-tournament win probabilities and group predictions to stdout.
Usage:  python run.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.teams import TEAMS, GROUPS, TEAM_GROUP
from models.monte_carlo import win_probabilities, simulate_group
from models.poisson import expected_goals, match_result_probs_poisson
from data.teams import get_elo

def main():
    print("\n" + "="*60)
    print("  ⚽  GoalAnalytics — World Cup 2026 Predictions")
    print("  Elo + Bivariate Poisson + Monte Carlo (10,000 sims)")
    print("="*60)

    print("\n📊 Running simulations... (this takes ~30s)")
    probs = win_probabilities(n=10_000, seed=42)

    print("\n🏆 Tournament Win Probabilities (Top 20)")
    print(f"{'Rank':<5} {'Team':<28} {'Group':<7} {'Elo':<6} {'Win %'}")
    print("-"*58)
    for rank, (team, prob) in enumerate(probs[:20], 1):
        flag = TEAMS[team]['flag']
        grp  = TEAM_GROUP[team]
        elo  = TEAMS[team]['elo']
        print(f"{rank:<5} {flag} {team:<25} {grp:<7} {elo:<6} {prob*100:.2f}%")

    print("\n\n📋 Group Stage Predicted Standings")
    for group in "ABCDEFGHIJKL":
        print(f"\n  Group {group}")
        print(f"  {'Team':<28} {'Elo'}")
        print("  " + "-"*38)
        from models.monte_carlo import simulate_group
        # Quick 1-sim deterministic preview using Elo order
        teams = sorted(GROUPS[group], key=lambda t: TEAMS[t]['elo'], reverse=True)
        for i, t in enumerate(teams, 1):
            flag = TEAMS[t]['flag']
            elo  = TEAMS[t]['elo']
            print(f"  {i}. {flag} {t:<25} {elo}")

    print("\n\n🎯 Sample Match Predictions")
    sample_matches = [
        ("Argentina", "France"),
        ("Spain", "Brazil"),
        ("England", "Germany"),
        ("Mexico", "United States"),
    ]
    for home, away in sample_matches:
        eh = get_elo(home)
        ea = get_elo(away)
        ph, pd_, pa = match_result_probs_poisson(eh, ea)
        lh, la = expected_goals(eh, ea)
        print(f"\n  {TEAMS[home]['flag']} {home} vs {TEAMS[away]['flag']} {away}")
        print(f"    xG: {lh:.2f} - {la:.2f}")
        print(f"    H {ph*100:.0f}%  |  D {pd_*100:.0f}%  |  A {pa*100:.0f}%")

    print("\n" + "="*60)
    print("  Launch dashboard:  streamlit run dashboard/app.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
