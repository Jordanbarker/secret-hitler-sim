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
    FASCIST_POLICIES,
    INITIAL_PRIOR_FASCIST_PROB,
    LIBERAL_POLICIES,
    NUM_FASCISTS,
    NUM_PLAYERS,
)
from .core import (
    Draw,
    ExecutivePower,
    Policy,
    Role,
    TermLimits,
    VoteRecord,
    chancellor_enacts,
    choose_execution_target,
    fascist_voting_strategy,
    hitler_voting_strategy,
    liberal_voting_strategy,
    president_passes,
    should_accept_veto,
    should_propose_veto,
)
from .simulation import GameSimulation, GameState, PlayerRoles, RoundType


def simulate_actual_draw(deck_fascist: int, deck_liberal: int, draw_count: int = 3) -> Draw:
    """
    Simulate a draw from the deck.
    Returns what cards were drawn.
    """
    deck = ["BAD"] * deck_fascist + ["GOOD"] * deck_liberal
    drawn = random.sample(deck, min(draw_count, len(deck)))
    return Draw(drawn.count("BAD"), drawn.count("GOOD"))


def determine_enacted_policy(
    draw: Draw, president_is_fascist: bool, chancellor_is_fascist: bool, pass_count: int = 2
) -> Policy:
    """
    Given draw and player types (Facist/Liberal), determine what gets enacted.
    """
    passed = president_passes(draw, president_is_fascist, pass_count)
    return chancellor_enacts(passed, chancellor_is_fascist)


def conduct_vote(
    num_players: int,
    president_id: int,
    chancellor_id: int,
    player_roles: PlayerRoles,
    suspicions: dict[int, float],
    election_tracker: int,
    fascist_policies_enacted: int,
    executed_players: set[int] | None = None,
) -> VoteRecord:
    """
    Conduct a vote on a chancellor nomination.

    All alive players vote using their role-appropriate strategy.
    """
    votes = VoteRecord()
    executed = executed_players or set()

    # Build is_fascist dict for fascist strategy
    is_fascist = {pid: player_roles.is_facist_or_hitler(pid) for pid in range(num_players)}

    for voter_id in range(num_players):
        if voter_id in executed:
            continue

        role = player_roles.get_role(voter_id)

        if role == Role.LIBERAL:
            vote = liberal_voting_strategy(
                president_id=president_id,
                chancellor_id=chancellor_id,
                suspicions=suspicions,
                election_tracker=election_tracker,
                bad_policies_enacted=fascist_policies_enacted,
            )
        elif role == Role.FASCIST:
            vote = fascist_voting_strategy(
                president_id=president_id,
                chancellor_id=chancellor_id,
                is_fascist=is_fascist,
                hitler_id=player_roles.hitler_id,
                bad_policies_enacted=fascist_policies_enacted,
                election_tracker=election_tracker,
            )
        else:  # HITLER
            vote = hitler_voting_strategy(
                president_id=president_id,
                chancellor_id=chancellor_id,
                suspicions=suspicions,
                election_tracker=election_tracker,
                bad_policies_enacted=fascist_policies_enacted,
            )

        votes.votes[voter_id] = vote

    return votes


def get_eligible_chancellors(
    num_players: int,
    president_id: int,
    term_limits: TermLimits,
    executed_players: set[int] | None = None,
) -> list[int]:
    """Get list of players eligible to be nominated as chancellor."""
    executed = executed_players or set()
    eligible = []
    for player_id in range(num_players):
        if player_id == president_id:
            continue
        if player_id in executed:
            continue
        if term_limits.is_eligible(player_id, num_players):
            eligible.append(player_id)
    assert eligible, "No eligible chancellor candidates"
    return eligible


def choose_chancellor_nomination(
    president_id: int,
    eligible_candidates: list[int],
    player_roles: PlayerRoles,
    suspicions: dict[int, float],
    fascist_policies_enacted: int,
) -> int:
    """
    President chooses a chancellor from eligible candidates.

    Strategy varies by role:
    - Liberal: Choose least suspicious eligible player
    - Fascist: Choose fellow fascist if possible, or Hitler after 3 Fascist policies
    - Hitler: Choose least suspicious to appear liberal
    """
    if not eligible_candidates:
        raise ValueError("No eligible chancellor candidates")

    role = player_roles.get_role(president_id)

    if role == Role.FASCIST:
        # Try to nominate Hitler after 3 Fascist policies (win condition)
        if fascist_policies_enacted >= 3 and player_roles.hitler_id in eligible_candidates:
            return player_roles.hitler_id
        # Otherwise prefer fellow fascists
        for candidate in eligible_candidates:
            if player_roles.is_facist_or_hitler(candidate):
                return candidate
        # Fall back to random
        return random.choice(eligible_candidates)

    # Liberal or Hitler: choose least suspicious
    sorted_candidates = sorted(eligible_candidates, key=lambda p: suspicions.get(p, 0.5))
    return sorted_candidates[0]


def flip_top_policy(deck_fascist: int, deck_liberal: int) -> tuple[Policy, int, int]:
    """
    Flip the top policy from the deck (chaos).

    Returns (policy, new_deck_fascist, new_deck_liberal).
    """
    total = deck_fascist + deck_liberal
    if total == 0:
        raise ValueError("Cannot flip from empty deck")

    # Probability of drawing Fascist
    if random.random() < deck_fascist / total:
        return Policy.BAD, deck_fascist - 1, deck_liberal
    else:
        return Policy.GOOD, deck_fascist, deck_liberal - 1


def generate_game(
    num_rounds: int = 30,
    seed: int | None = None,
) -> dict:
    """
    Generate a complete game with voting, election tracker, policy outcomes,
    executive actions, and veto power.

    Args:
        num_rounds: Maximum rounds to play
        seed: Random seed for reproducibility

    Returns:
        Dictionary with complete game data for visualization
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)

    fascist_player_ids = random.sample(range(NUM_PLAYERS), NUM_FASCISTS)
    player_roles = PlayerRoles.create(NUM_PLAYERS, fascist_player_ids, None)
    sim = GameSimulation()
    game_state = GameState()
    actual_deck_fascist = FASCIST_POLICIES
    actual_deck_liberal = LIBERAL_POLICIES

    game_data = {
        "config": {
            "seed": seed,
            "num_players": NUM_PLAYERS,
            "fascist_players": fascist_player_ids,
            "hitler_id": player_roles.hitler_id,
            "initial_deck": {"fascist": FASCIST_POLICIES, "liberal": LIBERAL_POLICIES},
            "prior_fascist_prob": INITIAL_PRIOR_FASCIST_PROB,
        },
        "initial_state": {
            "player_beliefs": {str(i): INITIAL_PRIOR_FASCIST_PROB for i in range(NUM_PLAYERS)},
            "deck_expected": {"fascist": FASCIST_POLICIES, "liberal": LIBERAL_POLICIES},
        },
        "rounds": [],
    }

    president_id = random.randint(0, NUM_PLAYERS - 1)
    round_num = 0

    while round_num < num_rounds and not game_state.game_over:
        round_num += 1

        # 1. Rotate president (skip executed players)
        president_id = (president_id + 1) % NUM_PLAYERS
        while not game_state.is_alive(president_id):
            president_id = (president_id + 1) % NUM_PLAYERS

        # 2. Get eligible chancellor candidates (apply term limits, exclude executed)
        eligible_chancellors = get_eligible_chancellors(
            NUM_PLAYERS,
            president_id,
            game_state.term_limits,
            executed_players=game_state.executed_players,
        )

        # Get current suspicions from Bayesian beliefs
        suspicions = dict(sim.player_beliefs.priors)

        # 3. President nominates chancellor
        chancellor_id = choose_chancellor_nomination(
            president_id=president_id,
            eligible_candidates=eligible_chancellors,
            player_roles=player_roles,
            suspicions=suspicions,
            fascist_policies_enacted=game_state.fascist_policies_enacted,
        )

        # 4. All alive players vote
        votes = conduct_vote(
            num_players=NUM_PLAYERS,
            president_id=president_id,
            chancellor_id=chancellor_id,
            player_roles=player_roles,
            suspicions=suspicions,
            election_tracker=game_state.election_tracker.count,
            fascist_policies_enacted=game_state.fascist_policies_enacted,
            executed_players=game_state.executed_players,
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
                if actual_deck_fascist + actual_deck_liberal < 1:
                    actual_deck_fascist = FASCIST_POLICIES
                    actual_deck_liberal = LIBERAL_POLICIES
                    sim.reset_deck(FASCIST_POLICIES, LIBERAL_POLICIES)
                    reshuffled = True

                enacted, actual_deck_fascist, actual_deck_liberal = flip_top_policy(
                    actual_deck_fascist, actual_deck_liberal
                )

                # Enact the policy and reset election tracker
                game_state.handle_chaos(enacted)

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
                    "actual_deck": {
                        "fascist": actual_deck_fascist,
                        "liberal": actual_deck_liberal,
                    },
                    "reshuffled": reshuffled,
                    "player_beliefs": {str(k): round(v, 4) for k, v in suspicions.items()},
                    "deck_expected": sim.deck_state.get_expected_composition(),
                    "policies_enacted": {
                        "fascist": game_state.fascist_policies_enacted,
                        "liberal": game_state.liberal_policies_enacted,
                    },
                    "executed_players": sorted(game_state.executed_players),
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
                        "fascist": round(sim.deck_state.get_expected_composition()[0], 1),
                        "liberal": round(sim.deck_state.get_expected_composition()[1], 1),
                    },
                    "policies_enacted": {
                        "fascist": game_state.fascist_policies_enacted,
                        "liberal": game_state.liberal_policies_enacted,
                    },
                    "executed_players": sorted(game_state.executed_players),
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
                        "fascist": round(sim.deck_state.get_expected_composition()[0], 1),
                        "liberal": round(sim.deck_state.get_expected_composition()[1], 1),
                    },
                    "policies_enacted": {
                        "fascist": game_state.fascist_policies_enacted,
                        "liberal": game_state.liberal_policies_enacted,
                    },
                    "executed_players": sorted(game_state.executed_players),
                }
                game_data["rounds"].append(round_data)
                break

            # Update game state for successful election
            game_state.handle_successful_election(president_id, chancellor_id)

            # Reshuffle with fresh deck if fewer than 3 cards remain
            reshuffled = False
            if actual_deck_fascist + actual_deck_liberal < 3:
                actual_deck_fascist = FASCIST_POLICIES
                actual_deck_liberal = LIBERAL_POLICIES
                sim.reset_deck(FASCIST_POLICIES, LIBERAL_POLICIES)
                reshuffled = True

            # Simulate actual draw
            draw = simulate_actual_draw(actual_deck_fascist, actual_deck_liberal)

            # Determine what actually happens based on true player types
            president_is_fascist = player_roles.is_facist_or_hitler(president_id)
            chancellor_is_fascist = player_roles.is_facist_or_hitler(chancellor_id)

            # Determine what the president passes to chancellor
            passed = president_passes(draw, president_is_fascist)

            # Check for veto power
            veto_proposed = False
            veto_accepted = False
            if game_state.veto_available:
                chancellor_role = player_roles.get_role(chancellor_id)
                if should_propose_veto(chancellor_role, passed):
                    veto_proposed = True
                    president_role = player_roles.get_role(president_id)
                    chanc_suspicion = suspicions.get(chancellor_id, 0.5)
                    if should_accept_veto(president_role, chanc_suspicion):
                        veto_accepted = True

            if veto_proposed and veto_accepted:
                # Veto succeeds: discard both cards, election tracker +1
                actual_deck_fascist -= draw.fascist
                actual_deck_liberal -= draw.liberal

                chaos_triggered = game_state.election_tracker.increment()

                round_data = {
                    "round_num": round_num,
                    "president_id": president_id,
                    "chancellor_id": chancellor_id,
                    "round_type": RoundType.LEGISLATIVE.value,
                    "vote_passed": True,
                    "votes": votes.to_dict(),
                    "election_tracker": game_state.election_tracker.count,
                    "actual_draw": {
                        "fascist": draw.fascist,
                        "liberal": draw.liberal,
                    },
                    "actual_deck": {
                        "fascist": actual_deck_fascist,
                        "liberal": actual_deck_liberal,
                    },
                    "reshuffled": reshuffled,
                    "player_beliefs": {str(k): round(v, 4) for k, v in suspicions.items()},
                    "deck_expected": {
                        "fascist": round(sim.deck_state.get_expected_composition()[0], 1),
                        "liberal": round(sim.deck_state.get_expected_composition()[1], 1),
                    },
                    "policies_enacted": {
                        "fascist": game_state.fascist_policies_enacted,
                        "liberal": game_state.liberal_policies_enacted,
                    },
                    "veto_proposed": True,
                    "veto_accepted": True,
                    "executed_players": sorted(game_state.executed_players),
                }

                # If chaos triggered by veto, enact top policy
                if chaos_triggered:
                    if actual_deck_fascist + actual_deck_liberal < 1:
                        actual_deck_fascist = FASCIST_POLICIES
                        actual_deck_liberal = LIBERAL_POLICIES
                        sim.reset_deck(FASCIST_POLICIES, LIBERAL_POLICIES)

                    (
                        chaos_policy,
                        actual_deck_fascist,
                        actual_deck_liberal,
                    ) = flip_top_policy(actual_deck_fascist, actual_deck_liberal)
                    game_state.handle_chaos(chaos_policy)
                    round_data["chaos"] = True
                    round_data["enacted"] = chaos_policy.value
                    round_data["policies_enacted"] = {
                        "fascist": game_state.fascist_policies_enacted,
                        "liberal": game_state.liberal_policies_enacted,
                    }

                    if game_state.game_over:
                        round_data["game_over"] = True
                        round_data["winner"] = game_state.winner
                        round_data["win_condition"] = game_state.win_condition
            else:
                # Normal legislative session (or veto rejected)
                enacted = chancellor_enacts(passed, chancellor_is_fascist)

                # Update actual deck (all 3 drawn cards are discarded)
                actual_deck_fascist -= draw.fascist
                actual_deck_liberal -= draw.liberal

                # Run Bayesian update
                result = sim.play_round(president_id, chancellor_id, enacted)

                # Enact the policy and check win conditions
                executive_power = game_state.enact_policy(enacted)

                # Build round data
                round_data = result.to_dict()
                round_data["round_type"] = RoundType.LEGISLATIVE.value
                round_data["votes"] = votes.to_dict()
                round_data["vote_passed"] = True
                round_data["election_tracker"] = game_state.election_tracker.count
                round_data["actual_draw"] = {
                    "fascist": draw.fascist,
                    "liberal": draw.liberal,
                }
                round_data["actual_deck"] = {
                    "fascist": actual_deck_fascist,
                    "liberal": actual_deck_liberal,
                }
                round_data["reshuffled"] = reshuffled
                round_data["policies_enacted"] = {
                    "fascist": game_state.fascist_policies_enacted,
                    "liberal": game_state.liberal_policies_enacted,
                }
                round_data["executed_players"] = sorted(game_state.executed_players)

                if veto_proposed and not veto_accepted:
                    round_data["veto_proposed"] = True
                    round_data["veto_accepted"] = False

                if game_state.game_over:
                    round_data["game_over"] = True
                    round_data["winner"] = game_state.winner
                    round_data["win_condition"] = game_state.win_condition

                # Handle executive power (only if game isn't over)
                elif executive_power != ExecutivePower.NONE:
                    round_data["executive_action"] = executive_power.value

                    if executive_power == ExecutivePower.POLICY_PEEK:
                        # President sees top 3 cards
                        deck = ["BAD"] * actual_deck_fascist + ["GOOD"] * actual_deck_liberal
                        peek = deck[:3] if len(deck) >= 3 else deck
                        round_data["policy_peek"] = peek

                    elif executive_power == ExecutivePower.EXECUTION:
                        # President chooses a player to execute
                        alive_players = [p for p in range(NUM_PLAYERS) if game_state.is_alive(p)]
                        is_fascist_map = {
                            p: player_roles.is_facist_or_hitler(p) for p in range(NUM_PLAYERS)
                        }
                        president_role = player_roles.get_role(president_id)
                        target = choose_execution_target(
                            president_id=president_id,
                            role=president_role,
                            alive_players=alive_players,
                            is_fascist_map=is_fascist_map,
                            hitler_id=player_roles.hitler_id,
                            suspicions=suspicions,
                        )
                        game_state.execute_player(target)
                        round_data["executed_player"] = target
                        round_data["executed_players"] = sorted(game_state.executed_players)

                        # Check if Hitler was assassinated
                        if game_state.check_hitler_assassination(target, player_roles):
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
            "fascist": game_state.fascist_policies_enacted,
            "liberal": game_state.liberal_policies_enacted,
        },
        "executed_players": sorted(game_state.executed_players),
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
    print(f"  Fascist players: {game_data['config']['fascist_players']}")
    print(f"  Hitler: Player {game_data['config']['hitler_id']}")
    print(f"  Prior P(Fascist): {game_data['config']['prior_fascist_prob']:.1%}")
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
        is_fascist = int(player_id) in game_data["config"]["fascist_players"]
        is_hitler = int(player_id) == game_data["config"]["hitler_id"]
        marker = " (HITLER)" if is_hitler else (" (FASCIST)" if is_fascist else "")
        print(f"  Player {player_id}: {float(prob):.1%}{marker}")


if __name__ == "__main__":
    main()
