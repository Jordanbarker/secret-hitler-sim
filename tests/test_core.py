"""
Tests for core data models and strategy functions.
"""

from math import isclose

import pytest

from policy_game import (
    DeckComposition,
    Draw,
    Policy,
    Role,
    chancellor_enacts,
    choose_execution_target,
    enacted_policy_for_types,
    president_passes,
    should_accept_veto,
    should_propose_veto,
)

# =============================================================================
# TESTS: Draw and DeckComposition
# =============================================================================


class TestDeckComposition:
    """Tests for DeckComposition class."""

    def test_total(self):
        deck = DeckComposition(11, 6)
        assert deck.total == 17

    def test_can_draw(self):
        deck = DeckComposition(5, 3)
        assert deck.can_draw(3) is True
        assert deck.can_draw(8) is True
        assert deck.can_draw(9) is False

    def test_possible_draws_normal(self):
        """Test possible draws from a normal deck."""
        deck = DeckComposition(11, 6)
        draws = deck.possible_draws(3)

        # Should have 4 possibilities: (3,0), (2,1), (1,2), (0,3)
        assert len(draws) == 4
        assert Draw(3, 0) in draws
        assert Draw(2, 1) in draws
        assert Draw(1, 2) in draws
        assert Draw(0, 3) in draws

    def test_possible_draws_limited_liberal(self):
        """Test draws when liberal policies are limited."""
        deck = DeckComposition(10, 1)
        draws = deck.possible_draws(3)

        # Can only draw 0 or 1 liberal
        assert len(draws) == 2
        assert Draw(3, 0) in draws
        assert Draw(2, 1) in draws

    def test_possible_draws_limited_fascist(self):
        """Test draws when fascist policies are limited."""
        deck = DeckComposition(1, 5)
        draws = deck.possible_draws(3)

        # Can only draw 0 or 1 fascist
        assert len(draws) == 2
        assert Draw(0, 3) in draws
        assert Draw(1, 2) in draws

    def test_draw_probability_sums_to_one(self):
        """Probabilities of all possible draws should sum to 1."""
        deck = DeckComposition(11, 6)
        draws = deck.possible_draws(3)
        total_prob = sum(deck.draw_probability(d) for d in draws)
        assert isclose(total_prob, 1.0, abs_tol=1e-9)

    def test_draw_probability_known_values(self):
        """Test specific probability calculations."""
        deck = DeckComposition(11, 6)

        # P(3 fascist, 0 liberal) = C(11,3)*C(6,0) / C(17,3)
        # = 165 * 1 / 680 = 0.2426...
        prob_3fascist = deck.draw_probability(Draw(3, 0))
        assert isclose(prob_3fascist, 165 / 680, abs_tol=1e-9)

        # P(0 fascist, 3 liberal) = C(11,0)*C(6,3) / C(17,3)
        # = 1 * 20 / 680 = 0.0294...
        prob_3liberal = deck.draw_probability(Draw(0, 3))
        assert isclose(prob_3liberal, 20 / 680, abs_tol=1e-9)

    def test_after_draw(self):
        """Test deck state after removing drawn cards."""
        deck = DeckComposition(11, 6)
        new_deck = deck.after_draw(Draw(2, 1))
        assert new_deck.fascist == 9
        assert new_deck.liberal == 5


# =============================================================================
# TESTS: Strategy Functions
# =============================================================================


class TestStrategy:
    """Tests for optimal play strategy functions."""

    def test_liberal_president_discards_fascist(self):
        """Liberal president should discard fascist policies when possible."""
        # Draw (2 fascist, 1 liberal) -> should pass (1 fascist, 1 liberal)
        passed = president_passes(Draw(2, 1), is_fascist=False)
        assert passed.fascist == 1
        assert passed.liberal == 1

        # Draw (3 fascist, 0 liberal) -> must pass (2 fascist, 0 liberal)
        passed = president_passes(Draw(3, 0), is_fascist=False)
        assert passed.fascist == 2
        assert passed.liberal == 0

        # Draw (1 fascist, 2 liberal) -> should pass (0 fascist, 2 liberal)
        passed = president_passes(Draw(1, 2), is_fascist=False)
        assert passed.fascist == 0
        assert passed.liberal == 2

    def test_fascist_president_discards_liberal(self):
        """Fascist president should discard liberal policies when possible."""
        # Draw (2 fascist, 1 liberal) -> should pass (2 fascist, 0 liberal)
        passed = president_passes(Draw(2, 1), is_fascist=True)
        assert passed.fascist == 2
        assert passed.liberal == 0

        # Draw (1 fascist, 2 liberal) -> should pass (1 fascist, 1 liberal)
        passed = president_passes(Draw(1, 2), is_fascist=True)
        assert passed.fascist == 1
        assert passed.liberal == 1

        # Draw (0 fascist, 3 liberal) -> must pass (0 fascist, 2 liberal)
        passed = president_passes(Draw(0, 3), is_fascist=True)
        assert passed.fascist == 0
        assert passed.liberal == 2

    def test_liberal_chancellor_enacts_liberal(self):
        """Liberal chancellor should enact liberal policy when possible."""
        assert chancellor_enacts(Draw(1, 1), is_fascist=False) == Policy.GOOD
        assert chancellor_enacts(Draw(0, 2), is_fascist=False) == Policy.GOOD
        assert chancellor_enacts(Draw(2, 0), is_fascist=False) == Policy.BAD  # forced

    def test_fascist_chancellor_enacts_fascist(self):
        """Fascist chancellor should enact fascist policy when possible."""
        assert chancellor_enacts(Draw(1, 1), is_fascist=True) == Policy.BAD
        assert chancellor_enacts(Draw(2, 0), is_fascist=True) == Policy.BAD
        assert chancellor_enacts(Draw(0, 2), is_fascist=True) == Policy.GOOD  # forced

    def test_enacted_policy_combinations(self):
        """Test full enacted policy for various type combinations."""
        # Both liberal, draw (2,1): liberal pres passes (1,1), liberal chanc enacts GOOD
        assert enacted_policy_for_types(Draw(2, 1), False, False) == Policy.GOOD

        # Both liberal, draw (3,0): liberal pres passes (2,0), liberal chanc forced BAD
        assert enacted_policy_for_types(Draw(3, 0), False, False) == Policy.BAD

        # Fascist pres, liberal chanc, draw (2,1): fascist pres passes (2,0), forced BAD
        assert enacted_policy_for_types(Draw(2, 1), True, False) == Policy.BAD

        # Liberal pres, fascist chanc, draw (2,1): liberal pres passes (1,1), enacts BAD
        assert enacted_policy_for_types(Draw(2, 1), False, True) == Policy.BAD

        # Both fascist, draw (0,3): fascist pres passes (0,2), fascist chanc forced GOOD
        assert enacted_policy_for_types(Draw(0, 3), True, True) == Policy.GOOD


class TestExecutionStrategy:
    """Tests for choose_execution_target function."""

    def test_liberal_targets_most_suspicious(self):
        """Liberal president should execute the most suspicious player."""
        target = choose_execution_target(
            president_id=0,
            role=Role.LIBERAL,
            alive_players=[0, 1, 2, 3, 4, 5],
            is_fascist_map={0: False, 1: False, 2: True, 3: False, 4: True, 5: False},
            hitler_id=4,
            suspicions={0: 0.1, 1: 0.3, 2: 0.9, 3: 0.2, 4: 0.7, 5: 0.1},
        )
        assert target == 2  # Most suspicious

    def test_fascist_avoids_hitler(self):
        """Fascist president should never execute Hitler."""
        for _ in range(20):
            target = choose_execution_target(
                president_id=0,
                role=Role.FASCIST,
                alive_players=[0, 1, 2, 3, 4, 5],
                is_fascist_map={0: True, 1: False, 2: False, 3: False, 4: True, 5: False},
                hitler_id=4,
                suspicions={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5},
            )
            assert target != 4, "Fascist should never execute Hitler"
            assert target != 0, "Should not execute self"
            assert target in [1, 2, 3, 5], "Should target a Liberal"

    def test_hitler_acts_like_liberal(self):
        """Hitler should execute the most suspicious player (like a Liberal)."""
        target = choose_execution_target(
            president_id=0,
            role=Role.HITLER,
            alive_players=[0, 1, 2, 3, 4, 5],
            is_fascist_map={0: True, 1: False, 2: True, 3: False, 4: False, 5: False},
            hitler_id=0,
            suspicions={0: 0.1, 1: 0.3, 2: 0.2, 3: 0.8, 4: 0.5, 5: 0.1},
        )
        assert target == 3  # Most suspicious


class TestVetoStrategy:
    """Tests for veto strategy functions."""

    def test_liberal_chancellor_vetoes_all_fascist(self):
        """Liberal chancellor should propose veto when only Fascist cards available."""
        assert should_propose_veto(Role.LIBERAL, Draw(2, 0)) is True

    def test_liberal_chancellor_no_veto_with_liberal(self):
        """Liberal chancellor should not veto when Liberal cards available."""
        assert should_propose_veto(Role.LIBERAL, Draw(1, 1)) is False
        assert should_propose_veto(Role.LIBERAL, Draw(0, 2)) is False

    def test_fascist_chancellor_vetoes_all_liberal(self):
        """Fascist chancellor should propose veto when only Liberal cards available."""
        assert should_propose_veto(Role.FASCIST, Draw(0, 2)) is True

    def test_fascist_chancellor_no_veto_with_fascist(self):
        """Fascist chancellor should not veto when Fascist cards available."""
        assert should_propose_veto(Role.FASCIST, Draw(1, 1)) is False
        assert should_propose_veto(Role.FASCIST, Draw(2, 0)) is False

    def test_liberal_president_accepts_veto_from_trusted(self):
        """Liberal president should accept veto from low-suspicion chancellor."""
        assert should_accept_veto(Role.LIBERAL, chancellor_suspicion=0.3) is True

    def test_liberal_president_rejects_veto_from_suspicious(self):
        """Liberal president should reject veto from high-suspicion chancellor."""
        assert should_accept_veto(Role.LIBERAL, chancellor_suspicion=0.8) is False

    def test_fascist_president_accepts_veto(self):
        """Fascist president should accept veto."""
        assert should_accept_veto(Role.FASCIST, chancellor_suspicion=0.8) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
