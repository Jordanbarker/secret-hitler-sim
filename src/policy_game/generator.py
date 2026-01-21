"""
Generate a complete game sequence for HTML visualization.

Creates a JSON file with all round data for a multi-player game
where some players are randomly selected to be bad.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .core import Draw, Policy, chancellor_enacts, president_passes
from .simulation import GameSimulation


def simulate_actual_draw(deck_bad: int, deck_good: int, draw_count: int = 3) -> Draw:
    """
    Simulate an actual draw from the deck.
    Returns what cards were actually drawn.
    """
    deck = ["BAD"] * deck_bad + ["GOOD"] * deck_good
    drawn = random.sample(deck, min(draw_count, len(deck)))
    return Draw(drawn.count("BAD"), drawn.count("GOOD"))


def determine_enacted_policy(
    draw: Draw, president_is_bad: bool, chancellor_is_bad: bool, pass_count: int = 2
) -> Policy:
    """
    Given actual draw and true player types, determine what gets enacted.
    """
    passed = president_passes(draw, president_is_bad, pass_count)
    return chancellor_enacts(passed, chancellor_is_bad)


def generate_game(
    num_players: int = 6,
    bad_player_ids: list[int] | None = None,
    bad_policies: int = 11,
    good_policies: int = 6,
    num_rounds: int = 10,
    seed: int | None = None,
) -> dict:
    """
    Generate a complete game with actual draws and policy outcomes.

    Args:
        num_players: Number of players at the table
        bad_player_ids: Which players are secretly bad
        bad_policies: Initial bad policies in deck
        good_policies: Initial good policies in deck
        num_rounds: Maximum rounds to play
        seed: Random seed for reproducibility

    Returns:
        Dictionary with complete game data for visualization
    """
    if seed is not None:
        random.seed(seed)

    if bad_player_ids is None:
        # Randomly select 2 bad players from all players
        bad_player_ids = random.sample(range(num_players), 2)

    bad_player_set = set(bad_player_ids)

    # Prior probability based on known bad count
    prior_bad_prob = len(bad_player_ids) / num_players

    # Initialize simulation for Bayesian tracking
    sim = GameSimulation(
        bad_policies=bad_policies,
        good_policies=good_policies,
        num_players=num_players,
        prior_bad_prob=prior_bad_prob,
    )

    # Track actual deck state
    actual_deck_bad = bad_policies
    actual_deck_good = good_policies

    game_data = {
        "config": {
            "num_players": num_players,
            "bad_players": bad_player_ids,
            "initial_deck": {"bad": bad_policies, "good": good_policies},
            "prior_bad_prob": prior_bad_prob,
        },
        "initial_state": {
            "player_beliefs": {str(i): prior_bad_prob for i in range(num_players)},
            "deck_expected": {"bad": bad_policies, "good": good_policies},
        },
        "rounds": [],
    }

    president_id = random.randint(0, num_players - 1)
    chancellor_id: int | None = None

    for round_idx in range(num_rounds):
        president_id = (president_id + 1) % num_players

        # Term limit: Chancellor must be new
        possible_chancellors = [
            i
            for i in range(num_players)
            if i != president_id and (chancellor_id is None or i != chancellor_id)
        ]
        chancellor_id = random.choice(possible_chancellors)

        # Reshuffle with fresh deck if fewer than 3 cards remain
        reshuffled = False
        if actual_deck_bad + actual_deck_good < 3:
            actual_deck_bad = bad_policies
            actual_deck_good = good_policies
            sim.reset_deck(bad_policies, good_policies)
            reshuffled = True

        # Simulate actual draw
        draw = simulate_actual_draw(actual_deck_bad, actual_deck_good)

        # Determine what actually happens based on true player types
        president_is_bad = president_id in bad_player_set
        chancellor_is_bad = chancellor_id in bad_player_set
        enacted = determine_enacted_policy(draw, president_is_bad, chancellor_is_bad)

        # Update actual deck (all 3 drawn cards are discarded)
        actual_deck_bad -= draw.bad
        actual_deck_good -= draw.good

        # Run Bayesian update
        result = sim.play_round(president_id, chancellor_id, enacted)

        # Use result.to_dict() and add game-specific fields
        round_data = result.to_dict()
        round_data["actual_draw"] = {"bad": draw.bad, "good": draw.good}
        round_data["actual_deck"] = {"bad": actual_deck_bad, "good": actual_deck_good}
        round_data["reshuffled"] = reshuffled

        game_data["rounds"].append(round_data)

    return game_data


def main():
    """Generate and save a game for visualization."""
    game_data = generate_game()

    # Save to data directory relative to package
    output_path = Path(__file__).parent.parent.parent / "data" / "game_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(game_data, f, indent=2)

    print(f"Game generated with {len(game_data['rounds'])} rounds")
    print(f"Saved to: {output_path}")
    print()
    print("Summary:")
    print(f"  Bad players: {game_data['config']['bad_players']}")
    print(f"  Prior P(bad): {game_data['config']['prior_bad_prob']:.1%}")
    print()

    for r in game_data["rounds"]:
        print(
            f"  Round {r['round_num']}: P{r['president_id']}->P{r['chancellor_id']} = {r['enacted']}"
        )

    print()
    print("Final beliefs:")
    final_beliefs = game_data["rounds"][-1]["player_beliefs"]
    for player_id, prob in sorted(final_beliefs.items(), key=lambda x: int(x[0])):
        is_bad = int(player_id) in game_data["config"]["bad_players"]
        marker = " (BAD)" if is_bad else ""
        print(f"  Player {player_id}: {float(prob):.1%}{marker}")


if __name__ == "__main__":
    main()
