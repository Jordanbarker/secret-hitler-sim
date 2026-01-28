"""
Run game simulations and collect comprehensive statistics.

Usage: uv run python scripts/game_stats.py [--games N]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from policy_game.generator import generate_game


def collect_stats(num_games: int = 1000) -> dict:
    """Run multiple game simulations and collect statistics."""
    stats = {
        "total_games": num_games,
        "wins": {"LIBERAL": 0, "FASCIST": 0},
        "win_conditions": defaultdict(int),
        "round_counts": [],
        "good_policies": [],
        "bad_policies": [],
        "total_rounds": 0,
        "chaos_rounds": 0,
        "votes_passed": 0,
        "votes_failed": 0,
        "chancellor_liberal": 0,
        "chancellor_fascist": 0,
        "chancellor_hitler": 0,
        "total_elections": 0,
    }

    for _ in range(num_games):
        game_data = generate_game()
        process_game(game_data, stats)

    return stats


def process_game(game_data: dict, stats: dict) -> None:
    """Process a single game and update statistics."""
    final_state = game_data["final_state"]
    config = game_data["config"]
    rounds = game_data["rounds"]

    # Winner
    winner = final_state.get("winner")
    if winner:
        stats["wins"][winner] += 1

    # Win condition
    win_condition = final_state.get("win_condition", "Unknown")
    stats["win_conditions"][win_condition] += 1

    # Round count
    stats["round_counts"].append(len(rounds))

    # Final policies
    policies = final_state.get("policies_enacted", {})
    stats["good_policies"].append(policies.get("good", 0))
    stats["bad_policies"].append(policies.get("bad", 0))

    # Per-round stats
    bad_players = set(config["bad_players"])
    hitler_id = config["hitler_id"]

    for round_data in rounds:
        stats["total_rounds"] += 1

        round_type = round_data.get("round_type", "LEGISLATIVE")

        # Chaos rounds
        if round_type == "CHAOS":
            stats["chaos_rounds"] += 1

        # Vote tracking
        vote_passed = round_data.get("vote_passed", False)
        if vote_passed:
            stats["votes_passed"] += 1
        else:
            stats["votes_failed"] += 1

        # Chancellor role tracking (only for passed elections)
        if vote_passed:
            chancellor_id = round_data.get("chancellor_id")
            if chancellor_id is not None:
                stats["total_elections"] += 1
                if chancellor_id == hitler_id:
                    stats["chancellor_hitler"] += 1
                elif chancellor_id in bad_players:
                    stats["chancellor_fascist"] += 1
                else:
                    stats["chancellor_liberal"] += 1


def print_report(stats: dict) -> None:
    """Print formatted statistics report."""
    total = stats["total_games"]

    print(f"\n=== Game Statistics ({total} games) ===\n")

    # Win rates
    liberal_wins = stats["wins"]["LIBERAL"]
    fascist_wins = stats["wins"]["FASCIST"]
    print("WIN RATES:")
    print(f"  Liberal:  {100 * liberal_wins / total:5.1f}% ({liberal_wins} wins)")
    print(f"  Fascist:  {100 * fascist_wins / total:5.1f}% ({fascist_wins} wins)")

    # Win conditions
    print("\nWIN CONDITIONS:")
    for condition, count in sorted(stats["win_conditions"].items(), key=lambda x: -x[1]):
        print(f"  {condition}: {100 * count / total:5.1f}%")

    # Game length
    round_counts = stats["round_counts"]
    avg_rounds = sum(round_counts) / len(round_counts)
    print("\nGAME LENGTH:")
    print(f"  Average rounds: {avg_rounds:.1f}")
    print(f"  Min/Max: {min(round_counts)} / {max(round_counts)}")

    # Policy stats
    avg_good = sum(stats["good_policies"]) / len(stats["good_policies"])
    avg_bad = sum(stats["bad_policies"]) / len(stats["bad_policies"])
    print("\nPOLICY STATS:")
    print(f"  Avg good policies: {avg_good:.1f}")
    print(f"  Avg bad policies:  {avg_bad:.1f}")

    # Election stats
    total_rounds = stats["total_rounds"]
    print("\nELECTION STATS:")
    total_votes = stats["votes_passed"] + stats["votes_failed"]
    if total_votes > 0:
        print(f"  Vote pass rate:      {100 * stats['votes_passed'] / total_votes:5.1f}%")
    chaos_pct = 100 * stats["chaos_rounds"] / total_rounds
    print(f"  Chaos rounds:        {chaos_pct:5.1f}% of total rounds")

    # Chancellor breakdown
    total_elections = stats["total_elections"]
    if total_elections > 0:
        print("\nCHANCELLOR ELECTIONS:")
        lib_pct = 100 * stats["chancellor_liberal"] / total_elections
        fas_pct = 100 * stats["chancellor_fascist"] / total_elections
        hit_pct = 100 * stats["chancellor_hitler"] / total_elections
        print(f"  Liberal chancellor:  {lib_pct:5.1f}%")
        print(f"  Fascist chancellor:  {fas_pct:5.1f}%")
        print(f"  Hitler chancellor:   {hit_pct:5.1f}% (of all elections)")


def main():
    parser = argparse.ArgumentParser(description="Run game simulations and collect statistics")
    parser.add_argument("--games", "-n", type=int, default=1000, help="Number of games to simulate")
    args = parser.parse_args()

    print(f"Running {args.games} game simulations...")
    stats = collect_stats(args.games)
    print_report(stats)


if __name__ == "__main__":
    main()
