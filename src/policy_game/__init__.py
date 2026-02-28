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
- Players vote on chancellor nominations
- Election tracker triggers chaos at 3 failed elections
- Game ends when 5 GOOD or 6 BAD policies enacted, or Hitler elected after 3 BAD
"""

from .core import (
    DeckComposition,
    Draw,
    ElectionTracker,
    ExecutivePower,
    Policy,
    Role,
    TermLimits,
    Vote,
    VoteRecord,
    chancellor_enacts,
    choose_execution_target,
    enacted_policy_for_types,
    fascist_voting_strategy,
    hitler_voting_strategy,
    liberal_voting_strategy,
    president_passes,
    should_accept_veto,
    should_propose_veto,
)
from .generator import generate_game
from .simulation import (
    DeckState,
    GameSimulation,
    GameState,
    PlayerBeliefs,
    PlayerRoles,
    RoundResult,
    RoundType,
    VotingBeliefs,
)

__all__ = [
    # Core models
    "Policy",
    "ExecutivePower",
    "Draw",
    "DeckComposition",
    # Voting and election models
    "Vote",
    "Role",
    "VoteRecord",
    "TermLimits",
    "ElectionTracker",
    # Strategy functions
    "president_passes",
    "chancellor_enacts",
    "enacted_policy_for_types",
    "choose_execution_target",
    "should_propose_veto",
    "should_accept_veto",
    # Voting strategy functions
    "liberal_voting_strategy",
    "fascist_voting_strategy",
    "hitler_voting_strategy",
    # Simulation classes
    "DeckState",
    "PlayerBeliefs",
    "PlayerRoles",
    "GameState",
    "VotingBeliefs",
    "RoundResult",
    "RoundType",
    "GameSimulation",
    # Generator
    "generate_game",
]
