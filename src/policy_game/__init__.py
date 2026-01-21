"""
Policy Game Bayesian Simulation

A simulation for social deduction games where players draw policies and we use
Bayesian inference to estimate the probability that each player is "bad" or "good"
based on observed policy outcomes.

Game Mechanics:
- Policy Deck: Configurable bad/good policies (default: 11 bad, 6 good)
- President draws N policies (default: 3), discards 1, passes N-1 to Chancellor
- Chancellor discards 1, enacts the remaining policy
- All drawn policies are removed from the deck
"""

from .core import (
    DeckComposition,
    Draw,
    Policy,
    chancellor_enacts,
    enacted_policy_for_types,
    president_passes,
)
from .generator import generate_game
from .simulation import (
    DeckState,
    GameSimulation,
    PlayerBeliefs,
    RoundResult,
)

__all__ = [
    # Core models
    "Policy",
    "Draw",
    "DeckComposition",
    # Strategy functions
    "president_passes",
    "chancellor_enacts",
    "enacted_policy_for_types",
    # Simulation classes
    "DeckState",
    "PlayerBeliefs",
    "RoundResult",
    "GameSimulation",
    # Generator
    "generate_game",
]
