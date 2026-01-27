"""
Generate a complete game sequence for HTML visualization.

Creates a JSON file with all round data for a multi-player game
where some players are randomly selected to be bad.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .constants import (
    BAD_POLICIES,
    GOOD_POLICIES,
    INITIAL_PRIOR_BAD_PROB,
    NUM_FASCISTS,
    NUM_PLAYERS,
)
from .core import (
    Draw,
    Policy,
    Role,
    TermLimits,
    VoteRecord,
    chancellor_enacts,
    fascist_voting_strategy,
    hitler_voting_strategy,
    liberal_voting_strategy,
    president_passes,
)
from .simulation import GameSimulation, GameState, PlayerRoles, RoundType


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


def conduct_vote(
    num_players: int,
    president_id: int,
    chancellor_id: int,
    player_roles: PlayerRoles,
    suspicions: dict[int, float],
    election_tracker: int,
    bad_policies_enacted: int,
) -> VoteRecord:
    """
    Conduct a vote on a chancellor nomination.

    All players vote using their role-appropriate strategy.
    """
    votes = VoteRecord()

    # Build is_fascist dict for fascist strategy
    is_fascist = {pid: player_roles.is_bad(pid) for pid in range(num_players)}

    for voter_id in range(num_players):
        role = player_roles.get_role(voter_id)

        if role == Role.LIBERAL:
            vote = liberal_voting_strategy(
                voter_id=voter_id,
                president_id=president_id,
                chancellor_id=chancellor_id,
                suspicions=suspicions,
                election_tracker=election_tracker,
                bad_policies_enacted=bad_policies_enacted,
            )
        elif role == Role.FASCIST:
            vote = fascist_voting_strategy(
                voter_id=voter_id,
                president_id=president_id,
                chancellor_id=chancellor_id,
                is_fascist=is_fascist,
                hitler_id=player_roles.hitler_id,
                bad_policies_enacted=bad_policies_enacted,
                election_tracker=election_tracker,
            )
        else:  # HITLER
            vote = hitler_voting_strategy(
                voter_id=voter_id,
                president_id=president_id,
                chancellor_id=chancellor_id,
                suspicions=suspicions,
                is_fascist=is_fascist,
                election_tracker=election_tracker,
                bad_policies_enacted=bad_policies_enacted,
            )

        votes.votes[voter_id] = vote

    return votes


def get_eligible_chancellors(
    num_players: int,
    president_id: int,
    term_limits: TermLimits,
) -> list[int]:
    """Get list of players eligible to be nominated as chancellor."""
    eligible = []
    for player_id in range(num_players):
        if player_id == president_id:
            continue
        if term_limits.is_eligible(player_id, num_players):
            eligible.append(player_id)
    return eligible


def choose_chancellor_nomination(
    president_id: int,
    eligible_candidates: list[int],
    player_roles: PlayerRoles,
    suspicions: dict[int, float],
    bad_policies_enacted: int,
) -> int:
    """
    President chooses a chancellor from eligible candidates.

    Strategy varies by role:
    - Liberal: Choose least suspicious eligible player
    - Fascist: Choose fellow fascist if possible, or Hitler after 3 BAD policies
    - Hitler: Choose least suspicious to appear liberal
    """
    if not eligible_candidates:
        raise ValueError("No eligible chancellor candidates")

    role = player_roles.get_role(president_id)

    if role == Role.FASCIST:
        # Try to nominate Hitler after 3 BAD policies (win condition)
        if bad_policies_enacted >= 3 and player_roles.hitler_id in eligible_candidates:
            return player_roles.hitler_id
        # Otherwise prefer fellow fascists
        for candidate in eligible_candidates:
            if player_roles.is_bad(candidate):
                return candidate
        # Fall back to random
        return random.choice(eligible_candidates)

    # Liberal or Hitler: choose least suspicious
    sorted_candidates = sorted(eligible_candidates, key=lambda p: suspicions.get(p, 0.5))
    return sorted_candidates[0]


def flip_top_policy(deck_bad: int, deck_good: int) -> tuple[Policy, int, int]:
    """
    Flip the top policy from the deck (chaos).

    Returns (policy, new_deck_bad, new_deck_good).
    """
    total = deck_bad + deck_good
    if total == 0:
        raise ValueError("Cannot flip from empty deck")

    # Probability of drawing BAD
    if random.random() < deck_bad / total:
        return Policy.BAD, deck_bad - 1, deck_good
    else:
        return Policy.GOOD, deck_bad, deck_good - 1


def generate_game(
    num_rounds: int = 20,
    seed: int | None = None,
) -> dict:
    """
    Generate a complete game with voting, election tracker, and policy outcomes.

    Args:
        num_rounds: Maximum rounds to play
        seed: Random seed for reproducibility

    Returns:
        Dictionary with complete game data for visualization
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)

    bad_player_ids = random.sample(range(NUM_PLAYERS), NUM_FASCISTS)
    player_roles = PlayerRoles.create(NUM_PLAYERS, bad_player_ids, None)
    sim = GameSimulation()
    game_state = GameState()

    # Track actual deck state
    actual_deck_bad = BAD_POLICIES
    actual_deck_good = GOOD_POLICIES

    game_data = {
        "config": {
            "seed": seed,
            "num_players": NUM_PLAYERS,
            "bad_players": bad_player_ids,
            "hitler_id": player_roles.hitler_id,
            "initial_deck": {"bad": BAD_POLICIES, "good": GOOD_POLICIES},
            "prior_bad_prob": INITIAL_PRIOR_BAD_PROB,
        },
        "initial_state": {
            "player_beliefs": {str(i): INITIAL_PRIOR_BAD_PROB for i in range(NUM_PLAYERS)},
            "deck_expected": {"bad": BAD_POLICIES, "good": GOOD_POLICIES},
        },
        "rounds": [],
    }

    president_id = random.randint(0, NUM_PLAYERS - 1)
    round_num = 0

    while round_num < num_rounds and not game_state.game_over:
        round_num += 1

        # 1. Rotate president
        president_id = (president_id + 1) % NUM_PLAYERS

        # 2. Get eligible chancellor candidates (apply term limits)
        eligible_chancellors = get_eligible_chancellors(
            NUM_PLAYERS, president_id, game_state.term_limits
        )

        if not eligible_chancellors:
            # Should not happen in normal play, but handle gracefully
            continue

        # Get current suspicions from Bayesian beliefs
        suspicions = dict(sim.player_beliefs.priors)

        # 3. President nominates chancellor
        chancellor_id = choose_chancellor_nomination(
            president_id=president_id,
            eligible_candidates=eligible_chancellors,
            player_roles=player_roles,
            suspicions=suspicions,
            bad_policies_enacted=game_state.bad_policies_enacted,
        )

        # 4. All players vote
        votes = conduct_vote(
            num_players=NUM_PLAYERS,
            president_id=president_id,
            chancellor_id=chancellor_id,
            player_roles=player_roles,
            suspicions=suspicions,
            election_tracker=game_state.election_tracker.count,
            bad_policies_enacted=game_state.bad_policies_enacted,
        )

        vote_passed = votes.passed

        # 5. Handle vote result
        if not vote_passed:
            # Vote failed - increment election tracker
            chaos_triggered = game_state.handle_failed_election()

            if chaos_triggered:
                # 5a. Chaos: flip top policy
                # Reshuffle if needed
                reshuffled = False
                if actual_deck_bad + actual_deck_good < 1:
                    actual_deck_bad = BAD_POLICIES
                    actual_deck_good = GOOD_POLICIES
                    sim.reset_deck(BAD_POLICIES, GOOD_POLICIES)
                    reshuffled = True

                enacted, actual_deck_bad, actual_deck_good = flip_top_policy(
                    actual_deck_bad, actual_deck_good
                )

                # Enact the policy
                game_state.enact_policy(enacted)
                game_state.handle_chaos()

                # Record chaos round
                round_data = {
                    "round_num": round_num,
                    "president_id": president_id,
                    "chancellor_id": chancellor_id,
                    "round_type": RoundType.CHAOS.value,
                    "vote_passed": False,
                    "votes": votes.to_dict(),
                    "election_tracker": 0,  # Reset after chaos
                    "enacted": enacted.value,
                    "chaos": True,
                    "actual_deck": {"bad": actual_deck_bad, "good": actual_deck_good},
                    "reshuffled": reshuffled,
                    "player_beliefs": {str(k): round(v, 4) for k, v in suspicions.items()},
                    "deck_expected": sim.deck_state.get_expected_composition(),
                    "policies_enacted": {
                        "bad": game_state.bad_policies_enacted,
                        "good": game_state.good_policies_enacted,
                    },
                }
            else:
                # 5b. Just a failed election
                round_data = {
                    "round_num": round_num,
                    "president_id": president_id,
                    "chancellor_id": chancellor_id,
                    "round_type": RoundType.FAILED_ELECTION.value,
                    "vote_passed": False,
                    "votes": votes.to_dict(),
                    "election_tracker": game_state.election_tracker.count,
                    "player_beliefs": {str(k): round(v, 4) for k, v in suspicions.items()},
                    "deck_expected": {
                        "bad": round(sim.deck_state.get_expected_composition()[0], 1),
                        "good": round(sim.deck_state.get_expected_composition()[1], 1),
                    },
                    "policies_enacted": {
                        "bad": game_state.bad_policies_enacted,
                        "good": game_state.good_policies_enacted,
                    },
                }
        else:
            # 6. Vote passed
            # Check Hitler win condition first
            if game_state.check_hitler_chancellor_win(chancellor_id, player_roles):
                round_data = {
                    "round_num": round_num,
                    "president_id": president_id,
                    "chancellor_id": chancellor_id,
                    "round_type": RoundType.LEGISLATIVE.value,
                    "vote_passed": True,
                    "votes": votes.to_dict(),
                    "election_tracker": 0,
                    "game_over": True,
                    "winner": "FASCIST",
                    "win_condition": game_state.win_condition,
                    "player_beliefs": {str(k): round(v, 4) for k, v in suspicions.items()},
                    "deck_expected": {
                        "bad": round(sim.deck_state.get_expected_composition()[0], 1),
                        "good": round(sim.deck_state.get_expected_composition()[1], 1),
                    },
                    "policies_enacted": {
                        "bad": game_state.bad_policies_enacted,
                        "good": game_state.good_policies_enacted,
                    },
                }
                game_data["rounds"].append(round_data)
                break

            # Update game state for successful election
            game_state.handle_successful_election(president_id, chancellor_id)

            # Reshuffle with fresh deck if fewer than 3 cards remain
            reshuffled = False
            if actual_deck_bad + actual_deck_good < 3:
                actual_deck_bad = BAD_POLICIES
                actual_deck_good = GOOD_POLICIES
                sim.reset_deck(BAD_POLICIES, GOOD_POLICIES)
                reshuffled = True

            # Simulate actual draw
            draw = simulate_actual_draw(actual_deck_bad, actual_deck_good)

            # Determine what actually happens based on true player types
            president_is_bad = player_roles.is_bad(president_id)
            chancellor_is_bad = player_roles.is_bad(chancellor_id)
            enacted = determine_enacted_policy(draw, president_is_bad, chancellor_is_bad)

            # Update actual deck (all 3 drawn cards are discarded)
            actual_deck_bad -= draw.bad
            actual_deck_good -= draw.good

            # Run Bayesian update
            result = sim.play_round(president_id, chancellor_id, enacted)

            # Enact the policy and check win conditions
            game_state.enact_policy(enacted)

            # Build round data
            round_data = result.to_dict()
            round_data["round_type"] = RoundType.LEGISLATIVE.value
            round_data["votes"] = votes.to_dict()
            round_data["vote_passed"] = True
            round_data["election_tracker"] = game_state.election_tracker.count
            round_data["actual_draw"] = {"bad": draw.bad, "good": draw.good}
            round_data["actual_deck"] = {"bad": actual_deck_bad, "good": actual_deck_good}
            round_data["reshuffled"] = reshuffled
            round_data["policies_enacted"] = {
                "bad": game_state.bad_policies_enacted,
                "good": game_state.good_policies_enacted,
            }

            if game_state.game_over:
                round_data["game_over"] = True
                round_data["winner"] = game_state.winner
                round_data["win_condition"] = game_state.win_condition

        game_data["rounds"].append(round_data)

    # Add final game state
    game_data["final_state"] = {
        "game_over": game_state.game_over,
        "winner": game_state.winner,
        "win_condition": game_state.win_condition,
        "policies_enacted": {
            "bad": game_state.bad_policies_enacted,
            "good": game_state.good_policies_enacted,
        },
    }

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
    print(f"  Hitler: Player {game_data['config']['hitler_id']}")
    print(f"  Prior P(bad): {game_data['config']['prior_bad_prob']:.1%}")
    print()

    for r in game_data["rounds"]:
        round_type = r.get("round_type", "LEGISLATIVE")
        if round_type == "FAILED_ELECTION":
            print(
                f"  Round {r['round_num']}: P{r['president_id']}->P{r['chancellor_id']} "
                f"FAILED (tracker: {r['election_tracker']})"
            )
        elif round_type == "CHAOS":
            print(f"  Round {r['round_num']}: CHAOS - {r.get('enacted', 'N/A')} enacted")
        else:
            enacted = r.get("enacted", "N/A")
            vote_info = "PASSED" if r.get("vote_passed", True) else "FAILED"
            print(
                f"  Round {r['round_num']}: P{r['president_id']}->P{r['chancellor_id']} "
                f"= {enacted} ({vote_info})"
            )

    print()

    # Print final game state
    final_state = game_data.get("final_state", {})
    if final_state.get("game_over"):
        print(f"Game Over: {final_state['winner']} wins!")
        print(f"  Reason: {final_state['win_condition']}")
    else:
        print("Game not finished (max rounds reached)")

    print()
    print(f"Policies enacted: {final_state.get('policies_enacted', {})}")

    print()
    print("Final beliefs:")
    final_beliefs = game_data["rounds"][-1]["player_beliefs"]
    for player_id, prob in sorted(final_beliefs.items(), key=lambda x: int(x[0])):
        is_bad = int(player_id) in game_data["config"]["bad_players"]
        is_hitler = int(player_id) == game_data["config"]["hitler_id"]
        marker = " (HITLER)" if is_hitler else (" (FASCIST)" if is_bad else "")
        print(f"  Player {player_id}: {float(prob):.1%}{marker}")


if __name__ == "__main__":
    main()
