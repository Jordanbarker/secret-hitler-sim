"""
Core data models and strategy functions for the policy game.

This module contains:
- Policy enum and Draw/DeckComposition dataclasses
- Optimal play strategy functions for presidents and chancellors
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import comb
from typing import Iterator


class Policy(Enum):
    """Policy types in the game."""

    BAD = "BAD"
    GOOD = "GOOD"


@dataclass(frozen=True)
class Draw:
    """Represents a draw of policies from the deck."""

    bad: int
    good: int

    @property
    def total(self) -> int:
        return self.bad + self.good

    def __iter__(self) -> Iterator[int]:
        yield self.bad
        yield self.good


@dataclass(frozen=True)
class DeckComposition:
    """Represents a possible deck state."""

    bad: int
    good: int

    @property
    def total(self) -> int:
        return self.bad + self.good

    def can_draw(self, draw_count: int) -> bool:
        """Check if this deck can support drawing draw_count cards."""
        return self.total >= draw_count

    def possible_draws(self, draw_count: int) -> list[Draw]:
        """Generate all possible draws of draw_count cards from this deck."""
        draws = []
        for bad_drawn in range(
            max(0, draw_count - self.good), min(draw_count, self.bad) + 1
        ):
            good_drawn = draw_count - bad_drawn
            if good_drawn <= self.good:
                draws.append(Draw(bad_drawn, good_drawn))
        return draws

    def draw_probability(self, draw: Draw) -> float:
        """
        Calculate probability of drawing exactly this combination.
        Uses hypergeometric distribution.
        """
        if draw.bad > self.bad or draw.good > self.good:
            return 0.0
        if draw.total > self.total:
            return 0.0

        numerator = comb(self.bad, draw.bad) * comb(self.good, draw.good)
        denominator = comb(self.total, draw.total)
        return numerator / denominator if denominator > 0 else 0.0

    def after_draw(self, draw: Draw) -> DeckComposition:
        """Return deck state after removing drawn cards."""
        return DeckComposition(self.bad - draw.bad, self.good - draw.good)


# =============================================================================
# OPTIMAL PLAY STRATEGY
# =============================================================================


def president_passes(draw: Draw, is_bad: bool, pass_count: int = 2) -> Draw:
    """
    Determine what the president passes to the chancellor.

    Good president: Maximizes good policies passed (discards bad if possible)
    Bad president: Maximizes bad policies passed (discards good if possible)

    Args:
        draw: The cards drawn by the president
        is_bad: Whether the president is a bad player
        pass_count: Number of cards to pass (default 2)

    Returns:
        Draw representing what is passed to chancellor
    """
    discard_count = draw.total - pass_count

    if is_bad:
        # Bad president discards good policies first
        good_discarded = min(draw.good, discard_count)
        bad_discarded = discard_count - good_discarded
    else:
        # Good president discards bad policies first
        bad_discarded = min(draw.bad, discard_count)
        good_discarded = discard_count - bad_discarded

    return Draw(draw.bad - bad_discarded, draw.good - good_discarded)


def chancellor_enacts(received: Draw, is_bad: bool) -> Policy:
    """
    Determine what policy the chancellor enacts.

    Good chancellor: Enacts good policy if available
    Bad chancellor: Enacts bad policy if available

    Args:
        received: The cards received from president
        is_bad: Whether the chancellor is a bad player

    Returns:
        The policy that gets enacted
    """
    if is_bad:
        # Bad chancellor enacts bad if possible
        return Policy.BAD if received.bad > 0 else Policy.GOOD
    else:
        # Good chancellor enacts good if possible
        return Policy.GOOD if received.good > 0 else Policy.BAD


def enacted_policy_for_types(
    draw: Draw, president_bad: bool, chancellor_bad: bool, pass_count: int = 2
) -> Policy:
    """
    Determine what policy gets enacted given a draw and player types.
    """
    passed = president_passes(draw, president_bad, pass_count)
    return chancellor_enacts(passed, chancellor_bad)
