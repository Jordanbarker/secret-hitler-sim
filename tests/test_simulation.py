"""
Tests for simulation and Bayesian inference classes.
"""

from math import isclose

import pytest

from policy_game import (
    DeckComposition,
    DeckState,
    ExecutivePower,
    GameSimulation,
    GameState,
    PlayerBeliefs,
    PlayerRoles,
    Policy,
)
from policy_game.generator import generate_game

# =============================================================================
# TESTS: DeckState
# =============================================================================


class TestDeckState:
    """Tests for DeckState belief tracking."""

    def test_initial_state(self):
        """Initial state should be certain."""
        deck = DeckState(11, 6)
        assert len(deck.distribution) == 1
        assert DeckComposition(11, 6) in deck.distribution
        assert deck.distribution[DeckComposition(11, 6)] == 1.0

    def test_expected_composition_initial(self):
        """Expected composition should match initial state."""
        deck = DeckState(11, 6)
        exp_fascist, exp_liberal = deck.get_expected_composition()
        assert exp_fascist == 11
        assert exp_liberal == 6

    def test_distribution_sums_to_one(self):
        """Distribution should always sum to 1 after updates."""
        deck = DeckState(11, 6)
        beliefs = {
            (False, False): 0.25,
            (False, True): 0.25,
            (True, False): 0.25,
            (True, True): 0.25,
        }

        deck.update(Policy.BAD, beliefs)
        total = sum(deck.distribution.values())
        assert isclose(total, 1.0, abs_tol=1e-9)


# =============================================================================
# TESTS: PlayerBeliefs
# =============================================================================


class TestPlayerBeliefs:
    """Tests for PlayerBeliefs tracking."""

    def test_initial_beliefs(self):
        """Initial beliefs should match prior."""
        beliefs = PlayerBeliefs(num_players=2, prior_fascist_prob=0.4)
        assert beliefs.priors[0] == 0.4
        assert beliefs.priors[1] == 0.4

    def test_joint_probability(self):
        """Test joint probability calculation."""
        beliefs = PlayerBeliefs(num_players=2, prior_fascist_prob=0.5)
        joint = beliefs.get_joint_probability(0, 1)

        assert isclose(joint[(False, False)], 0.25)
        assert isclose(joint[(False, True)], 0.25)
        assert isclose(joint[(True, False)], 0.25)
        assert isclose(joint[(True, True)], 0.25)

    def test_beliefs_increase_on_fascist_policy(self):
        """Beliefs should increase when fascist policy enacted."""
        deck = DeckState(11, 6)
        beliefs = PlayerBeliefs(num_players=2, prior_fascist_prob=0.5)

        pres_before = beliefs.priors[0]
        chanc_before = beliefs.priors[1]

        beliefs.update(0, 1, Policy.BAD, deck)

        # Fascist policy should increase suspicion
        assert beliefs.priors[0] > pres_before
        assert beliefs.priors[1] > chanc_before

    def test_beliefs_decrease_on_liberal_policy(self):
        """Beliefs should decrease when liberal policy enacted."""
        deck = DeckState(11, 6)
        beliefs = PlayerBeliefs(num_players=2, prior_fascist_prob=0.5)

        pres_before = beliefs.priors[0]
        chanc_before = beliefs.priors[1]

        beliefs.update(0, 1, Policy.GOOD, deck)

        # Liberal policy should decrease suspicion
        assert beliefs.priors[0] < pres_before
        assert beliefs.priors[1] < chanc_before


# =============================================================================
# TESTS: GameSimulation
# =============================================================================


class TestGameSimulation:
    """Integration tests for full simulation."""

    def test_simulation_runs(self):
        """Basic test that simulation runs without errors."""
        sim = GameSimulation()
        result = sim.play_round(0, 1, Policy.BAD)

        assert result.round_num == 1
        assert result.enacted == Policy.BAD

    def test_multiple_rounds(self):
        """Test running multiple rounds."""
        sim = GameSimulation()

        sim.play_round(0, 1, Policy.BAD)
        sim.play_round(1, 0, Policy.BAD)
        result = sim.play_round(0, 1, Policy.GOOD)

        assert result.round_num == 3
        assert len(sim.history) == 3

    def test_probabilities_valid(self):
        """All probabilities should be between 0 and 1."""
        sim = GameSimulation()

        for _ in range(3):
            result = sim.play_round(0, 1, Policy.BAD)

            assert 0 <= result.president_prob_fascist <= 1
            assert 0 <= result.chancellor_prob_fascist <= 1

            for _, prob in result.top_deck_states:
                assert 0 <= prob <= 1

    def test_deck_shrinks(self):
        """Deck should shrink after each round."""
        sim = GameSimulation(fascist_policies=11, liberal_policies=6)

        initial_exp = sim.deck_state.get_expected_composition()
        assert initial_exp[0] + initial_exp[1] == 17

        sim.play_round(0, 1, Policy.BAD)
        after_one = sim.deck_state.get_expected_composition()
        # Expected total is 14 (17 - 3 drawn), using isclose for float comparison
        assert isclose(after_one[0] + after_one[1], 14, abs_tol=1e-9)  # 17 - 3

    def test_custom_parameters(self):
        """Test with custom game parameters."""
        sim = GameSimulation(
            fascist_policies=8, liberal_policies=4, draw_count=4, pass_count=3, num_players=3
        )

        assert len(sim.player_beliefs.priors) == 3
        result = sim.play_round(0, 2, Policy.GOOD)
        assert result is not None


# =============================================================================
# TESTS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_deck_all_fascist(self):
        """Test with deck that's all fascist policies."""
        sim = GameSimulation(fascist_policies=10, liberal_policies=0, prior_fascist_prob=0.5)
        result = sim.play_round(0, 1, Policy.BAD)

        # With all fascist policies, BAD is certain regardless of player types
        # So probabilities shouldn't change much from prior
        assert isclose(result.president_prob_fascist, result.president_prob_before, abs_tol=0.01)
        assert isclose(result.chancellor_prob_fascist, result.chancellor_prob_before, abs_tol=0.01)

    def test_extreme_deck_all_liberal(self):
        """Test with deck that's all liberal policies."""
        sim = GameSimulation(fascist_policies=0, liberal_policies=10, prior_fascist_prob=0.5)
        result = sim.play_round(0, 1, Policy.GOOD)

        # With all liberal policies, GOOD is certain regardless of player types
        assert isclose(result.president_prob_fascist, result.president_prob_before, abs_tol=0.01)
        assert isclose(result.chancellor_prob_fascist, result.chancellor_prob_before, abs_tol=0.01)

    def test_asymmetric_prior(self):
        """Test with asymmetric prior probability."""
        sim = GameSimulation(prior_fascist_prob=0.3)

        # Initial beliefs should match prior
        assert sim.player_beliefs.priors[0] == 0.3
        assert sim.player_beliefs.priors[1] == 0.3


class TestExecutivePowers:
    """Tests for executive power triggers."""

    def test_no_power_first_two_fascist(self):
        """First two Fascist policies grant no power."""
        gs = GameState()
        power1 = gs.enact_policy(Policy.BAD)
        assert power1 == ExecutivePower.NONE

        power2 = gs.enact_policy(Policy.BAD)
        assert power2 == ExecutivePower.NONE

    def test_policy_peek_on_third_fascist(self):
        """Third Fascist policy grants Policy Peek."""
        gs = GameState()
        gs.enact_policy(Policy.BAD)
        gs.enact_policy(Policy.BAD)
        power3 = gs.enact_policy(Policy.BAD)
        assert power3 == ExecutivePower.POLICY_PEEK

    def test_execution_on_fourth_fascist(self):
        """Fourth Fascist policy grants Execution."""
        gs = GameState()
        for _ in range(3):
            gs.enact_policy(Policy.BAD)
        power4 = gs.enact_policy(Policy.BAD)
        assert power4 == ExecutivePower.EXECUTION

    def test_execution_on_fifth_fascist(self):
        """Fifth Fascist policy grants Execution."""
        gs = GameState()
        for _ in range(4):
            gs.enact_policy(Policy.BAD)
        power5 = gs.enact_policy(Policy.BAD)
        assert power5 == ExecutivePower.EXECUTION

    def test_liberal_policy_no_power(self):
        """Liberal policies never grant executive power."""
        gs = GameState()
        power = gs.enact_policy(Policy.GOOD)
        assert power == ExecutivePower.NONE

    def test_sixth_fascist_game_over_no_power(self):
        """Sixth Fascist policy ends game, returns NONE."""
        gs = GameState()
        for _ in range(5):
            gs.enact_policy(Policy.BAD)
        power6 = gs.enact_policy(Policy.BAD)
        assert power6 == ExecutivePower.NONE
        assert gs.game_over is True


class TestHitlerAssassination:
    """Tests for Hitler assassination win condition."""

    def test_assassinating_hitler_wins(self):
        """Executing Hitler should end the game with Liberal win."""
        gs = GameState()
        roles = PlayerRoles.create(6, [0, 1], hitler_id=0)

        gs.execute_player(0)
        result = gs.check_hitler_assassination(0, roles)

        assert result is True
        assert gs.game_over is True
        assert gs.winner == "LIBERAL"
        assert gs.win_condition == "Hitler was assassinated"

    def test_assassinating_non_hitler_continues(self):
        """Executing a non-Hitler player should not end the game."""
        gs = GameState()
        roles = PlayerRoles.create(6, [0, 1], hitler_id=0)

        gs.execute_player(2)
        result = gs.check_hitler_assassination(2, roles)

        assert result is False
        assert gs.game_over is False


class TestExecutedPlayers:
    """Tests for executed player tracking."""

    def test_execute_player(self):
        """Executed players should be tracked."""
        gs = GameState()
        assert gs.is_alive(3) is True

        gs.execute_player(3)
        assert gs.is_alive(3) is False
        assert 3 in gs.executed_players

    def test_multiple_executions(self):
        """Multiple players can be executed."""
        gs = GameState()
        gs.execute_player(1)
        gs.execute_player(4)

        assert gs.is_alive(1) is False
        assert gs.is_alive(4) is False
        assert gs.is_alive(0) is True
        assert len(gs.executed_players) == 2


class TestVetoAvailability:
    """Tests for veto power availability."""

    def test_veto_not_available_initially(self):
        """Veto should not be available at the start."""
        gs = GameState()
        assert gs.veto_available is False

    def test_veto_available_after_fifth_fascist(self):
        """Veto should be available after 5th Fascist policy."""
        gs = GameState()
        for _ in range(5):
            gs.enact_policy(Policy.BAD)
        assert gs.veto_available is True

    def test_veto_not_available_after_four_fascist(self):
        """Veto should not be available after only 4 Fascist policies."""
        gs = GameState()
        for _ in range(4):
            gs.enact_policy(Policy.BAD)
        assert gs.veto_available is False


class TestGenerateGameNewMechanics:
    """Integration tests for new game mechanics in generated games."""

    def test_execution_appears_in_game_data(self):
        """Games with enough Fascist policies should have execution actions."""
        # Run many games looking for one with an execution
        found_execution = False
        for seed in range(100):
            game = generate_game(seed=seed)
            for rnd in game["rounds"]:
                if rnd.get("executive_action") == "EXECUTION":
                    found_execution = True
                    assert "executed_player" in rnd
                    break
            if found_execution:
                break
        assert found_execution, "No execution found in 100 games"

    def test_policy_peek_appears_in_game_data(self):
        """Games with 3 Fascist policies should have policy peek."""
        found_peek = False
        for seed in range(100):
            game = generate_game(seed=seed)
            for rnd in game["rounds"]:
                if rnd.get("executive_action") == "POLICY_PEEK":
                    found_peek = True
                    assert "policy_peek" in rnd
                    assert len(rnd["policy_peek"]) <= 3
                    break
            if found_peek:
                break
        assert found_peek, "No policy peek found in 100 games"

    def test_executed_players_in_final_state(self):
        """Final state should include executed_players list."""
        game = generate_game(seed=42)
        assert "executed_players" in game["final_state"]
        assert isinstance(game["final_state"]["executed_players"], list)

    def test_hitler_assassination_win(self):
        """Should be possible to win by assassinating Hitler."""
        found_assassination = False
        for seed in range(500):
            game = generate_game(seed=seed)
            if game["final_state"].get("win_condition") == "Hitler was assassinated":
                found_assassination = True
                assert game["final_state"]["winner"] == "LIBERAL"
                break
        assert found_assassination, "No Hitler assassination win found in 500 games"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
