"""
Core data models and strategy functions for the policy game.

This module contains:
- Policy enum and Draw/DeckComposition dataclasses
- Vote/Role enums and VoteRecord/TermLimits/ElectionTracker dataclasses
- Optimal play strategy functions for presidents and chancellors
- Voting strategy functions for different player roles
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from math import comb
from typing import Iterator


class Policy(Enum):
    """Policy types in the game."""

    BAD = "BAD"
    GOOD = "GOOD"


class ExecutivePower(Enum):
    """Executive powers granted after enacting Fascist policies."""

    NONE = "NONE"
    POLICY_PEEK = "POLICY_PEEK"
    EXECUTION = "EXECUTION"


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

    fascist: int
    liberal: int

    @property
    def total(self) -> int:
        return self.fascist + self.liberal

    def __iter__(self) -> Iterator[int]:
        yield self.fascist
        yield self.liberal


@dataclass(frozen=True)
class DeckComposition:
    """Represents a possible deck state."""

    fascist: int
    liberal: int

    @property
    def total(self) -> int:
        return self.fascist + self.liberal

    def can_draw(self, draw_count: int) -> bool:
        """Check if this deck can support drawing draw_count cards."""
        return self.total >= draw_count

    def possible_draws(self, draw_count: int) -> list[Draw]:
        """Generate all possible draws of draw_count cards from this deck."""
        draws = []
        for fascist_drawn in range(
            max(0, draw_count - self.liberal), min(draw_count, self.fascist) + 1
        ):
            liberal_drawn = draw_count - fascist_drawn
            if liberal_drawn <= self.liberal:
                draws.append(Draw(fascist_drawn, liberal_drawn))
        return draws

    def draw_probability(self, draw: Draw) -> float:
        """
        Calculate probability of drawing exactly this combination.
        Uses hypergeometric distribution.
        """
        if draw.fascist > self.fascist or draw.liberal > self.liberal:
            return 0.0
        if draw.total > self.total:
            return 0.0

        numerator = comb(self.fascist, draw.fascist) * comb(self.liberal, draw.liberal)
        denominator = comb(self.total, draw.total)
        return numerator / denominator if denominator > 0 else 0.0

    def after_draw(self, draw: Draw) -> DeckComposition:
        """Return deck state after removing drawn cards."""
        return DeckComposition(self.fascist - draw.fascist, self.liberal - draw.liberal)


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


def president_passes(draw: Draw, is_fascist: bool, pass_count: int = 2) -> Draw:
    """
    Determine what the president passes to the chancellor.

    Liberal president: Maximizes liberal policies passed (discards fascist if possible)
    Fascist president: Maximizes fascist policies passed (discards liberal if possible)

    Args:
        draw: The cards drawn by the president
        is_fascist: Whether the president is a fascist player
        pass_count: Number of cards to pass (default 2)

    Returns:
        Draw representing what is passed to chancellor
    """
    discard_count = draw.total - pass_count

    if is_fascist:
        # Fascist president discards liberal policies first
        liberal_discarded = min(draw.liberal, discard_count)
        fascist_discarded = discard_count - liberal_discarded
    else:
        # Liberal president discards fascist policies first
        fascist_discarded = min(draw.fascist, discard_count)
        liberal_discarded = discard_count - fascist_discarded

    return Draw(draw.fascist - fascist_discarded, draw.liberal - liberal_discarded)


def chancellor_enacts(received: Draw, is_fascist: bool) -> Policy:
    """
    Determine what policy the chancellor enacts.

    Liberal chancellor: Enacts liberal policy if available
    Fascist chancellor: Enacts fascist policy if available

    Args:
        received: The cards received from president
        is_fascist: Whether the chancellor is a fascist player

    Returns:
        The policy that gets enacted
    """
    if is_fascist:
        # Fascist chancellor enacts fascist if possible
        return Policy.BAD if received.fascist > 0 else Policy.GOOD
    else:
        # Liberal chancellor enacts liberal if possible
        return Policy.GOOD if received.liberal > 0 else Policy.BAD


def enacted_policy_for_types(
    draw: Draw, president_fascist: bool, chancellor_fascist: bool, pass_count: int = 2
) -> Policy:
    """
    Determine what policy gets enacted given a draw and player types.
    """
    passed = president_passes(draw, president_fascist, pass_count)
    return chancellor_enacts(passed, chancellor_fascist)


# =============================================================================
# VOTING STRATEGIES
# =============================================================================


def liberal_voting_strategy(
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
    pres_suspicion = suspicions.get(president_id)
    chanc_suspicion = suspicions.get(chancellor_id)
    assert pres_suspicion is not None, "Suspicion values must be provided for president candidate"
    assert chanc_suspicion is not None, "Suspicion values must be provided for chancellor candidate"

    # If election tracker is at 2, tend to vote JA to avoid chaos
    if election_tracker >= 2:
        # Still vote NEIN if both are highly suspicious
        if pres_suspicion > 0.7 and chanc_suspicion > 0.7:
            return Vote.NEIN
        return Vote.JA

    # Vote NEIN if either candidate seems suspicious
    if pres_suspicion > 0.6 or chanc_suspicion > 0.6:
        return Vote.NEIN
    return Vote.JA


def fascist_voting_strategy(
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
    president_id: int,
    chancellor_id: int,
    suspicions: dict[int, float],
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


# =============================================================================
# EXECUTION STRATEGY
# =============================================================================


def choose_execution_target(
    president_id: int,
    role: Role,
    alive_players: list[int],
    is_fascist_map: dict[int, bool],
    hitler_id: int,
    suspicions: dict[int, float],
) -> int:
    """
    President chooses a player to execute.

    Strategy:
    - Liberal president: Execute the most suspicious player
    - Fascist president: Execute a random Liberal (avoid Hitler and fellow Fascists)
    - Hitler president: Act like a Liberal (execute most suspicious) to blend in
    """
    candidates = [p for p in alive_players if p != president_id]

    if role == Role.FASCIST:
        # Avoid executing Hitler or fellow Fascists
        liberal_candidates = [p for p in candidates if not is_fascist_map.get(p, False)]
        if liberal_candidates:
            return random.choice(liberal_candidates)
        # Fallback: pick random non-self (shouldn't happen normally)
        return random.choice(candidates)

    # Liberal or Hitler: execute most suspicious player
    sorted_candidates = sorted(candidates, key=lambda p: suspicions.get(p, 0.5), reverse=True)
    return sorted_candidates[0]


# =============================================================================
# VETO STRATEGY
# =============================================================================


def should_propose_veto(
    chancellor_role: Role,
    received: Draw,
) -> bool:
    """
    Chancellor decides whether to propose a veto.

    Strategy:
    - Liberal chancellor: Propose veto when forced to enact Fascist (received 2 Fascist)
    - Fascist chancellor: Propose veto when forced to enact Liberal (received 2 Liberal)
    - Hitler: Act like Liberal
    """
    if chancellor_role in (Role.LIBERAL, Role.HITLER):
        # Veto when only Fascist policies available
        return received.liberal == 0
    else:
        # Fascist: veto when only Liberal policies available
        return received.fascist == 0


def should_accept_veto(
    president_role: Role,
    chancellor_suspicion: float,
) -> bool:
    """
    President decides whether to accept a veto proposal.

    Strategy:
    - Liberal president: Accept if chancellor seems Liberal (low suspicion)
    - Fascist president: Accept (vetoing delays Liberal policies)
    - Hitler: Act like Liberal
    """
    if president_role in (Role.LIBERAL, Role.HITLER):
        # Accept veto if chancellor doesn't seem suspicious
        return chancellor_suspicion < 0.6
    else:
        # Fascist president: usually accept veto
        return True
