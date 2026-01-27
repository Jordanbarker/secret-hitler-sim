"""
Tests for simulation and Bayesian inference classes.
"""

from math import isclose

import pytest

from policy_game import (
    DeckComposition,
    DeckState,
    GameSimulation,
    PlayerBeliefs,
    Policy,
)

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
        exp_bad, exp_good = deck.get_expected_composition()
        assert exp_bad == 11
        assert exp_good == 6

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
        beliefs = PlayerBeliefs(num_players=2, prior_bad_prob=0.4)
        assert beliefs.priors[0] == 0.4
        assert beliefs.priors[1] == 0.4

    def test_joint_probability(self):
        """Test joint probability calculation."""
        beliefs = PlayerBeliefs(num_players=2, prior_bad_prob=0.5)
        joint = beliefs.get_joint_probability(0, 1)

        assert isclose(joint[(False, False)], 0.25)
        assert isclose(joint[(False, True)], 0.25)
        assert isclose(joint[(True, False)], 0.25)
        assert isclose(joint[(True, True)], 0.25)

    def test_beliefs_increase_on_bad_policy(self):
        """Beliefs should increase when bad policy enacted."""
        deck = DeckState(11, 6)
        beliefs = PlayerBeliefs(num_players=2, prior_bad_prob=0.5)

        pres_before = beliefs.priors[0]
        chanc_before = beliefs.priors[1]

        beliefs.update(0, 1, Policy.BAD, deck)

        # Bad policy should increase suspicion
        assert beliefs.priors[0] > pres_before
        assert beliefs.priors[1] > chanc_before

    def test_beliefs_decrease_on_good_policy(self):
        """Beliefs should decrease when good policy enacted."""
        deck = DeckState(11, 6)
        beliefs = PlayerBeliefs(num_players=2, prior_bad_prob=0.5)

        pres_before = beliefs.priors[0]
        chanc_before = beliefs.priors[1]

        beliefs.update(0, 1, Policy.GOOD, deck)

        # Good policy should decrease suspicion
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

            assert 0 <= result.president_prob_bad <= 1
            assert 0 <= result.chancellor_prob_bad <= 1

            for _, prob in result.top_deck_states:
                assert 0 <= prob <= 1

    def test_deck_shrinks(self):
        """Deck should shrink after each round."""
        sim = GameSimulation(bad_policies=11, good_policies=6)

        initial_exp = sim.deck_state.get_expected_composition()
        assert initial_exp[0] + initial_exp[1] == 17

        sim.play_round(0, 1, Policy.BAD)
        after_one = sim.deck_state.get_expected_composition()
        # Expected total is 14 (17 - 3 drawn), using isclose for float comparison
        assert isclose(after_one[0] + after_one[1], 14, abs_tol=1e-9)  # 17 - 3

    def test_custom_parameters(self):
        """Test with custom game parameters."""
        sim = GameSimulation(
            bad_policies=8, good_policies=4, draw_count=4, pass_count=3, num_players=3
        )

        assert len(sim.player_beliefs.priors) == 3
        result = sim.play_round(0, 2, Policy.GOOD)
        assert result is not None


# =============================================================================
# TESTS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_deck_all_bad(self):
        """Test with deck that's all bad policies."""
        sim = GameSimulation(bad_policies=10, good_policies=0, prior_bad_prob=0.5)
        result = sim.play_round(0, 1, Policy.BAD)

        # With all bad policies, BAD is certain regardless of player types
        # So probabilities shouldn't change much from prior
        assert isclose(result.president_prob_bad, result.president_prob_before, abs_tol=0.01)
        assert isclose(result.chancellor_prob_bad, result.chancellor_prob_before, abs_tol=0.01)

    def test_extreme_deck_all_good(self):
        """Test with deck that's all good policies."""
        sim = GameSimulation(bad_policies=0, good_policies=10, prior_bad_prob=0.5)
        result = sim.play_round(0, 1, Policy.GOOD)

        # With all good policies, GOOD is certain regardless of player types
        assert isclose(result.president_prob_bad, result.president_prob_before, abs_tol=0.01)
        assert isclose(result.chancellor_prob_bad, result.chancellor_prob_before, abs_tol=0.01)

    def test_asymmetric_prior(self):
        """Test with asymmetric prior probability."""
        sim = GameSimulation(prior_bad_prob=0.3)

        # Initial beliefs should match prior
        assert sim.player_beliefs.priors[0] == 0.3
        assert sim.player_beliefs.priors[1] == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
