"""
Tests for core data models and strategy functions.
"""

from math import isclose

import pytest

from policy_game import (
    DeckComposition,
    Draw,
    Policy,
    chancellor_enacts,
    enacted_policy_for_types,
    president_passes,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
