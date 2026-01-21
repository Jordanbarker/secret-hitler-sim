"""
Tests for core data models and strategy functions.
"""

import pytest
from math import isclose

from policy_game import (
    Policy,
    Draw,
    DeckComposition,
    president_passes,
    chancellor_enacts,
    enacted_policy_for_types,
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

    def test_possible_draws_limited_good(self):
        """Test draws when good policies are limited."""
        deck = DeckComposition(10, 1)
        draws = deck.possible_draws(3)

        # Can only draw 0 or 1 good
        assert len(draws) == 2
        assert Draw(3, 0) in draws
        assert Draw(2, 1) in draws

    def test_possible_draws_limited_bad(self):
        """Test draws when bad policies are limited."""
        deck = DeckComposition(1, 5)
        draws = deck.possible_draws(3)

        # Can only draw 0 or 1 bad
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

        # P(3 bad, 0 good) = C(11,3)*C(6,0) / C(17,3)
        # = 165 * 1 / 680 = 0.2426...
        prob_3bad = deck.draw_probability(Draw(3, 0))
        assert isclose(prob_3bad, 165 / 680, abs_tol=1e-9)

        # P(0 bad, 3 good) = C(11,0)*C(6,3) / C(17,3)
        # = 1 * 20 / 680 = 0.0294...
        prob_3good = deck.draw_probability(Draw(0, 3))
        assert isclose(prob_3good, 20 / 680, abs_tol=1e-9)

    def test_after_draw(self):
        """Test deck state after removing drawn cards."""
        deck = DeckComposition(11, 6)
        new_deck = deck.after_draw(Draw(2, 1))
        assert new_deck.bad == 9
        assert new_deck.good == 5


# =============================================================================
# TESTS: Strategy Functions
# =============================================================================


class TestStrategy:
    """Tests for optimal play strategy functions."""

    def test_good_president_discards_bad(self):
        """Good president should discard bad policies when possible."""
        # Draw (2 bad, 1 good) -> should pass (1 bad, 1 good)
        passed = president_passes(Draw(2, 1), is_bad=False)
        assert passed.bad == 1
        assert passed.good == 1

        # Draw (3 bad, 0 good) -> must pass (2 bad, 0 good)
        passed = president_passes(Draw(3, 0), is_bad=False)
        assert passed.bad == 2
        assert passed.good == 0

        # Draw (1 bad, 2 good) -> should pass (0 bad, 2 good)
        passed = president_passes(Draw(1, 2), is_bad=False)
        assert passed.bad == 0
        assert passed.good == 2

    def test_bad_president_discards_good(self):
        """Bad president should discard good policies when possible."""
        # Draw (2 bad, 1 good) -> should pass (2 bad, 0 good)
        passed = president_passes(Draw(2, 1), is_bad=True)
        assert passed.bad == 2
        assert passed.good == 0

        # Draw (1 bad, 2 good) -> should pass (1 bad, 1 good)
        passed = president_passes(Draw(1, 2), is_bad=True)
        assert passed.bad == 1
        assert passed.good == 1

        # Draw (0 bad, 3 good) -> must pass (0 bad, 2 good)
        passed = president_passes(Draw(0, 3), is_bad=True)
        assert passed.bad == 0
        assert passed.good == 2

    def test_good_chancellor_enacts_good(self):
        """Good chancellor should enact good policy when possible."""
        assert chancellor_enacts(Draw(1, 1), is_bad=False) == Policy.GOOD
        assert chancellor_enacts(Draw(0, 2), is_bad=False) == Policy.GOOD
        assert chancellor_enacts(Draw(2, 0), is_bad=False) == Policy.BAD  # forced

    def test_bad_chancellor_enacts_bad(self):
        """Bad chancellor should enact bad policy when possible."""
        assert chancellor_enacts(Draw(1, 1), is_bad=True) == Policy.BAD
        assert chancellor_enacts(Draw(2, 0), is_bad=True) == Policy.BAD
        assert chancellor_enacts(Draw(0, 2), is_bad=True) == Policy.GOOD  # forced

    def test_enacted_policy_combinations(self):
        """Test full enacted policy for various type combinations."""
        # Both good, draw (2,1): good pres passes (1,1), good chanc enacts GOOD
        assert enacted_policy_for_types(Draw(2, 1), False, False) == Policy.GOOD

        # Both good, draw (3,0): good pres passes (2,0), good chanc forced BAD
        assert enacted_policy_for_types(Draw(3, 0), False, False) == Policy.BAD

        # Bad pres, good chanc, draw (2,1): bad pres passes (2,0), good chanc forced BAD
        assert enacted_policy_for_types(Draw(2, 1), True, False) == Policy.BAD

        # Good pres, bad chanc, draw (2,1): good pres passes (1,1), bad chanc enacts BAD
        assert enacted_policy_for_types(Draw(2, 1), False, True) == Policy.BAD

        # Both bad, draw (0,3): bad pres passes (0,2), bad chanc forced GOOD
        assert enacted_policy_for_types(Draw(0, 3), True, True) == Policy.GOOD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
