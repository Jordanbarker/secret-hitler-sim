"""
Core data models and strategy functions for the policy game.

This module contains:
- Policy enum and Draw/DeckComposition dataclasses
- Vote/Role enums and VoteRecord/TermLimits/ElectionTracker dataclasses
- Optimal play strategy functions for presidents and chancellors
- Voting strategy functions for different player roles
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import comb
from typing import Iterator


class Policy(Enum):
    """Policy types in the game."""

    BAD = "BAD"
    GOOD = "GOOD"


class Vote(Enum):
    """Vote types for chancellor nominations."""

    JA = "JA"
    NEIN = "NEIN"


class Role(Enum):
    """Player roles in the game."""

    LIBERAL = "LIBERAL"
    FASCIST = "FASCIST"
    HITLER = "HITLER"


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
        for bad_drawn in range(max(0, draw_count - self.good), min(draw_count, self.bad) + 1):
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
# VOTING AND ELECTION DATA MODELS
# =============================================================================


@dataclass
class VoteRecord:
    """Tracks all player votes for a nomination."""

    votes: dict[int, Vote] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if the vote passed (majority JA)."""
        if not self.votes:
            return False
        ja_count = sum(1 for v in self.votes.values() if v == Vote.JA)
        return ja_count > len(self.votes) / 2

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {str(k): v.value for k, v in self.votes.items()}


@dataclass
class TermLimits:
    """Tracks term limits for president and chancellor positions."""

    last_president: int | None = None
    last_chancellor: int | None = None

    def is_eligible(self, player_id: int, num_players: int) -> bool:
        """Check if a player is eligible to be nominated as chancellor."""
        # In games with 5 or fewer players, only the last chancellor is ineligible
        # In games with more than 5 players, both last president and chancellor are ineligible
        if player_id == self.last_chancellor:
            return False
        if num_players > 5 and player_id == self.last_president:
            return False
        return True

    def clear(self) -> None:
        """Clear term limits (called after chaos)."""
        self.last_president = None
        self.last_chancellor = None

    def update(self, president_id: int, chancellor_id: int) -> None:
        """Update term limits after successful election."""
        self.last_president = president_id
        self.last_chancellor = chancellor_id


@dataclass
class ElectionTracker:
    """Tracks failed elections and triggers chaos at 3."""

    count: int = 0

    def increment(self) -> bool:
        """Increment the tracker. Returns True if chaos triggered (count reaches 3)."""
        self.count += 1
        if self.count >= 3:
            self.count = 0
            return True
        return False

    def reset(self) -> None:
        """Reset the tracker to 0."""
        self.count = 0


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


# =============================================================================
# VOTING STRATEGIES
# =============================================================================


def liberal_voting_strategy(
    voter_id: int,
    president_id: int,
    chancellor_id: int,
    suspicions: dict[int, float],
    election_tracker: int,
    bad_policies_enacted: int,
) -> Vote:
    """
    Liberal voting strategy based on suspicion levels and game state.

    Liberals vote based on:
    - Suspicion of both candidates (vote NEIN if both seem suspicious)
    - Election tracker pressure (vote JA if tracker is at 2 to avoid chaos)
    - General tendency to approve unless suspicious
    """
    pres_suspicion = suspicions.get(president_id, 0.5)
    chanc_suspicion = suspicions.get(chancellor_id, 0.5)

    # If election tracker is at 2, tend to vote JA to avoid chaos
    if election_tracker >= 2:
        # Still vote NEIN if both are highly suspicious
        if pres_suspicion > 0.7 and chanc_suspicion > 0.7:
            return Vote.NEIN
        return Vote.JA

    # Vote NEIN if either candidate seems suspicious
    if pres_suspicion > 0.6 or chanc_suspicion > 0.6:
        return Vote.NEIN

    # Default to JA
    return Vote.JA


def fascist_voting_strategy(
    voter_id: int,
    president_id: int,
    chancellor_id: int,
    is_fascist: dict[int, bool],
    hitler_id: int,
    bad_policies_enacted: int,
    election_tracker: int,
) -> Vote:
    """
    Fascist voting strategy to support fascist goals.

    Fascists vote based on:
    - Supporting fellow fascists getting into power
    - Especially supporting Hitler as chancellor after 3 BAD policies (win condition)
    - Appearing liberal by occasionally voting NEIN on obvious fascist pairings
    """
    president_is_fascist = is_fascist.get(president_id, False)
    chancellor_is_fascist = is_fascist.get(chancellor_id, False)
    chancellor_is_hitler = chancellor_id == hitler_id

    # Win condition: Hitler as chancellor after 3+ BAD policies
    if bad_policies_enacted >= 3 and chancellor_is_hitler:
        return Vote.JA

    # Support fascist governments
    if president_is_fascist or chancellor_is_fascist:
        return Vote.JA

    # Occasionally vote JA even for liberal governments to blend in
    # But more likely to vote NEIN for liberal governments
    return Vote.NEIN


def hitler_voting_strategy(
    voter_id: int,
    president_id: int,
    chancellor_id: int,
    suspicions: dict[int, float],
    is_fascist: dict[int, bool],
    election_tracker: int,
    bad_policies_enacted: int,
) -> Vote:
    """
    Hitler's voting strategy to appear liberal while subtly supporting fascists.

    Hitler:
    - Doesn't know who the other fascists are (in 6+ player games)
    - Tries to appear liberal to avoid suspicion
    - Uses similar logic to liberals but slightly biased toward supporting fascists
    """
    pres_suspicion = suspicions.get(president_id, 0.5)
    chanc_suspicion = suspicions.get(chancellor_id, 0.5)

    # If election tracker is high, vote JA to appear cooperative
    if election_tracker >= 2:
        return Vote.JA

    # Vote similarly to a liberal but with lower suspicion threshold
    # This makes Hitler appear slightly more cooperative/trusting
    if pres_suspicion > 0.7 and chanc_suspicion > 0.7:
        return Vote.NEIN

    # Default to JA more often than a typical liberal would
    return Vote.JA
