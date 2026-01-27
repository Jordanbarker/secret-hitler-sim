"""
Game simulation and Bayesian inference classes.

This module contains:
- DeckState: Tracks probability distribution over deck compositions
- PlayerBeliefs: Tracks probability estimates for each player being bad
- PlayerRoles: Tracks which player has which role (Liberal/Fascist/Hitler)
- GameState: Tracks policies enacted, election tracker, term limits, win conditions
- VotingBeliefs: Bayesian updates from voting patterns
- RoundResult: Results from a single round
- GameSimulation: Orchestrates the game and inference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .constants import (
    BAD_POLICIES,
    GOOD_POLICIES,
    INITIAL_PRIOR_BAD_PROB,
    NUM_PLAYERS,
)
from .core import (
    DeckComposition,
    ElectionTracker,
    Policy,
    Role,
    TermLimits,
    VoteRecord,
    enacted_policy_for_types,
)


class RoundType(Enum):
    """Types of rounds in the game."""

    LEGISLATIVE = "LEGISLATIVE"
    FAILED_ELECTION = "FAILED_ELECTION"
    CHAOS = "CHAOS"


class DeckState:
    """
    Tracks the probability distribution over possible deck compositions.

    Since we only observe the enacted policy (not the full draw or discards),
    the deck composition becomes uncertain after each round.
    """

    def __init__(self, bad_policies: int = BAD_POLICIES, good_policies: int = GOOD_POLICIES):
        """Initialize with known deck composition."""
        initial = DeckComposition(bad_policies, good_policies)
        self.distribution: dict[DeckComposition, float] = {initial: 1.0}

    def get_expected_composition(self) -> tuple[float, float]:
        """Return expected (bad, good) counts."""
        exp_bad = sum(deck.bad * prob for deck, prob in self.distribution.items())
        exp_good = sum(deck.good * prob for deck, prob in self.distribution.items())
        return exp_bad, exp_good

    def update(
        self,
        enacted: Policy,
        player_beliefs: dict[tuple[bool, bool], float],
        draw_count: int = 3,
        pass_count: int = 2,
    ) -> None:
        """
        Update deck belief distribution after observing an enacted policy.

        Args:
            enacted: The policy that was enacted
            player_beliefs: P(pres_bad, chanc_bad) for each type combination
            draw_count: Number of cards drawn by president
            pass_count: Number of cards passed to chancellor
        """
        new_distribution: dict[DeckComposition, float] = {}

        for deck, deck_prob in self.distribution.items():
            if not deck.can_draw(draw_count):
                continue

            for draw in deck.possible_draws(draw_count):
                draw_prob = deck.draw_probability(draw)
                if draw_prob == 0:
                    continue

                # Sum over player type combinations that produce this outcome
                outcome_prob = 0.0
                for (pres_bad, chanc_bad), type_prob in player_beliefs.items():
                    if enacted_policy_for_types(draw, pres_bad, chanc_bad, pass_count) == enacted:
                        outcome_prob += type_prob

                if outcome_prob > 0:
                    new_deck = deck.after_draw(draw)
                    weight = deck_prob * draw_prob * outcome_prob

                    if new_deck in new_distribution:
                        new_distribution[new_deck] += weight
                    else:
                        new_distribution[new_deck] = weight

        # Normalize
        total = sum(new_distribution.values())
        if total > 0:
            self.distribution = {k: v / total for k, v in new_distribution.items()}
        else:
            # This shouldn't happen with valid inputs
            raise ValueError("No valid deck states after update - check inputs")

    def top_states(self, n: int = 5) -> list[tuple[DeckComposition, float]]:
        """Return the n most likely deck states."""
        sorted_states = sorted(self.distribution.items(), key=lambda x: x[1], reverse=True)
        return sorted_states[:n]

    def reset(self, bad_policies: int, good_policies: int) -> None:
        """Reset deck to a fresh state with known composition."""
        initial = DeckComposition(bad_policies, good_policies)
        self.distribution = {initial: 1.0}


class PlayerBeliefs:
    """Tracks probability estimates for each player being bad."""

    def __init__(
        self, num_players: int = NUM_PLAYERS, prior_bad_prob: float = INITIAL_PRIOR_BAD_PROB
    ):
        """
        Initialize player beliefs with uniform priors.

        Args:
            num_players: Number of players in the game
            prior_bad_prob: Prior probability that each player is bad
        """
        self.priors: dict[int, float] = {i: prior_bad_prob for i in range(num_players)}

    def get_joint_probability(
        self, president_id: int, chancellor_id: int
    ) -> dict[tuple[bool, bool], float]:
        """
        Get joint probability distribution over (president_bad, chancellor_bad).

        Assumes independence between players.
        """
        p_pres = self.priors[president_id]
        p_chanc = self.priors[chancellor_id]

        return {
            (False, False): (1 - p_pres) * (1 - p_chanc),
            (False, True): (1 - p_pres) * p_chanc,
            (True, False): p_pres * (1 - p_chanc),
            (True, True): p_pres * p_chanc,
        }

    def update(
        self,
        president_id: int,
        chancellor_id: int,
        enacted: Policy,
        deck_state: DeckState,
        draw_count: int = 3,
        pass_count: int = 2,
    ) -> tuple[float, float]:
        """
        Update beliefs about president and chancellor based on enacted policy.

        Returns:
            Tuple of (new_president_prob, new_chancellor_prob)
        """
        # Calculate likelihood of enacted policy for each type combination
        likelihoods: dict[tuple[bool, bool], float] = {
            (False, False): 0.0,
            (False, True): 0.0,
            (True, False): 0.0,
            (True, True): 0.0,
        }

        for deck, deck_prob in deck_state.distribution.items():
            if not deck.can_draw(draw_count):
                continue

            for draw in deck.possible_draws(draw_count):
                draw_prob = deck.draw_probability(draw)
                if draw_prob == 0:
                    continue

                for pres_bad in [False, True]:
                    for chanc_bad in [False, True]:
                        if (
                            enacted_policy_for_types(draw, pres_bad, chanc_bad, pass_count)
                            == enacted
                        ):
                            likelihoods[(pres_bad, chanc_bad)] += deck_prob * draw_prob

        # Get current joint priors
        joint_prior = self.get_joint_probability(president_id, chancellor_id)

        # Calculate posterior using Bayes' theorem
        joint_posterior: dict[tuple[bool, bool], float] = {}
        total = 0.0

        for types, prior in joint_prior.items():
            posterior = prior * likelihoods[types]
            joint_posterior[types] = posterior
            total += posterior

        # Normalize
        if total > 0:
            joint_posterior = {k: v / total for k, v in joint_posterior.items()}

        # Marginalize to get individual posteriors
        p_pres_bad = joint_posterior[(True, False)] + joint_posterior[(True, True)]
        p_chanc_bad = joint_posterior[(False, True)] + joint_posterior[(True, True)]

        # Update priors for next round
        self.priors[president_id] = p_pres_bad
        self.priors[chancellor_id] = p_chanc_bad

        return p_pres_bad, p_chanc_bad


@dataclass
class PlayerRoles:
    """Tracks which player has which role."""

    roles: dict[int, Role] = field(default_factory=dict)
    hitler_id: int | None = None

    @classmethod
    def create(
        cls, num_players: int, bad_player_ids: list[int], hitler_id: int | None = None
    ) -> "PlayerRoles":
        """
        Create player roles for a game.

        Args:
            num_players: Total number of players
            bad_player_ids: IDs of bad players (fascists + Hitler)
            hitler_id: Which bad player is Hitler (defaults to first bad player)
        """
        if hitler_id is None and bad_player_ids:
            hitler_id = bad_player_ids[0]

        roles = {}
        for player_id in range(num_players):
            if player_id == hitler_id:
                roles[player_id] = Role.HITLER
            elif player_id in bad_player_ids:
                roles[player_id] = Role.FASCIST
            else:
                roles[player_id] = Role.LIBERAL

        return cls(roles=roles, hitler_id=hitler_id)

    def is_bad(self, player_id: int) -> bool:
        """Check if a player is on the fascist team (fascist or Hitler)."""
        role = self.roles.get(player_id)
        return role in (Role.FASCIST, Role.HITLER)

    def is_hitler(self, player_id: int) -> bool:
        """Check if a player is Hitler."""
        return player_id == self.hitler_id

    def get_role(self, player_id: int) -> Role:
        """Get a player's role."""
        return self.roles.get(player_id, Role.LIBERAL)


@dataclass
class GameState:
    """Tracks game state including policies, elections, and win conditions."""

    bad_policies_enacted: int = 0
    good_policies_enacted: int = 0
    election_tracker: ElectionTracker = field(default_factory=ElectionTracker)
    term_limits: TermLimits = field(default_factory=TermLimits)
    game_over: bool = False
    winner: str | None = None  # "LIBERAL" or "FASCIST"
    win_condition: str | None = None

    def enact_policy(
        self,
        policy: Policy,
        player_roles: PlayerRoles | None = None,
        chancellor_id: int | None = None,
    ) -> bool:
        """
        Enact a policy and check win conditions.

        Returns True if game is over.
        """
        if policy == Policy.BAD:
            self.bad_policies_enacted += 1
        else:
            self.good_policies_enacted += 1

        # Check policy win conditions
        if self.good_policies_enacted >= 5:
            self.game_over = True
            self.winner = "LIBERAL"
            self.win_condition = "5 GOOD policies enacted"
            return True
        if self.bad_policies_enacted >= 6:
            self.game_over = True
            self.winner = "FASCIST"
            self.win_condition = "6 BAD policies enacted"
            return True

        return False

    def check_hitler_chancellor_win(self, chancellor_id: int, player_roles: PlayerRoles) -> bool:
        """Check if Hitler being elected chancellor triggers fascist win."""
        if self.bad_policies_enacted >= 3 and player_roles.is_hitler(chancellor_id):
            self.game_over = True
            self.winner = "FASCIST"
            self.win_condition = "Hitler elected Chancellor after 3+ BAD policies"
            return True
        return False

    def handle_successful_election(self, president_id: int, chancellor_id: int) -> None:
        """Update state after a successful election."""
        self.election_tracker.reset()
        self.term_limits.update(president_id, chancellor_id)

    def handle_failed_election(self) -> bool:
        """
        Handle a failed election. Returns True if chaos triggered.
        """
        return self.election_tracker.increment()

    def handle_chaos(self) -> None:
        """Handle chaos - clear term limits."""
        self.term_limits.clear()


class VotingBeliefs:
    """Tracks and updates beliefs based on voting patterns."""

    def __init__(self, num_players: int):
        """Initialize voting belief tracking."""
        self.num_players = num_players
        # Track voting history for pattern analysis
        self.vote_history: list[tuple[int, int, VoteRecord, bool]] = []
        # Track how often players vote together
        self.vote_agreement: dict[tuple[int, int], int] = {}
        self.vote_total: dict[tuple[int, int], int] = {}

    def record_vote(
        self,
        president_id: int,
        chancellor_id: int,
        votes: VoteRecord,
        vote_passed: bool,
    ) -> None:
        """Record a vote for later analysis."""
        self.vote_history.append((president_id, chancellor_id, votes, vote_passed))

        # Update agreement tracking
        player_ids = list(votes.votes.keys())
        for i, p1 in enumerate(player_ids):
            for p2 in player_ids[i + 1 :]:
                pair = (min(p1, p2), max(p1, p2))
                if pair not in self.vote_total:
                    self.vote_total[pair] = 0
                    self.vote_agreement[pair] = 0
                self.vote_total[pair] += 1
                if votes.votes[p1] == votes.votes[p2]:
                    self.vote_agreement[pair] += 1

    def get_agreement_rate(self, player1: int, player2: int) -> float | None:
        """Get the rate at which two players vote the same way."""
        pair = (min(player1, player2), max(player1, player2))
        if pair not in self.vote_total or self.vote_total[pair] == 0:
            return None
        return self.vote_agreement[pair] / self.vote_total[pair]


@dataclass
class RoundResult:
    """Results from a single round of the game."""

    round_num: int
    president_id: int
    chancellor_id: int
    enacted: Policy | None  # None for failed elections
    president_prob_bad: float
    chancellor_prob_bad: float
    president_prob_before: float
    chancellor_prob_before: float
    top_deck_states: list[tuple[DeckComposition, float]]
    all_player_beliefs: dict[int, float] = field(default_factory=dict)
    deck_expected: tuple[float, float] = (0.0, 0.0)
    # New voting fields
    round_type: RoundType = RoundType.LEGISLATIVE
    votes: VoteRecord | None = None
    vote_passed: bool = True
    election_tracker: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary for visualization."""
        result = {
            "round_num": self.round_num,
            "president_id": self.president_id,
            "chancellor_id": self.chancellor_id,
            "round_type": self.round_type.value,
            "vote_passed": self.vote_passed,
            "election_tracker": self.election_tracker,
            "player_beliefs": {str(k): round(v, 4) for k, v in self.all_player_beliefs.items()},
            "deck_expected": {
                "bad": round(self.deck_expected[0], 1),
                "good": round(self.deck_expected[1], 1),
            },
        }
        if self.enacted is not None:
            result["enacted"] = self.enacted.value
        if self.votes is not None:
            result["votes"] = self.votes.to_dict()
        return result


class GameSimulation:
    """
    Orchestrates the game simulation and Bayesian inference.
    """

    def __init__(
        self,
        bad_policies: int = BAD_POLICIES,
        good_policies: int = GOOD_POLICIES,
        draw_count: int = 3,
        pass_count: int = 2,
        num_players: int = NUM_PLAYERS,
        prior_bad_prob: float = INITIAL_PRIOR_BAD_PROB,
    ):
        """
        Initialize the game simulation.

        Args:
            bad_policies: Number of bad policies in initial deck
            good_policies: Number of good policies in initial deck
            draw_count: Number of policies president draws
            pass_count: Number of policies passed to chancellor
            num_players: Number of players in the game
            prior_bad_prob: Prior probability each player is bad
        """
        self.deck_state = DeckState(bad_policies, good_policies)
        self.player_beliefs = PlayerBeliefs(num_players, prior_bad_prob)
        self.draw_count = draw_count
        self.pass_count = pass_count
        self.round_num = 0
        self.history: list[RoundResult] = []

    def play_round(self, president_id: int, chancellor_id: int, enacted: Policy) -> RoundResult:
        """
        Process a round and update all beliefs.

        Args:
            president_id: ID of the president player
            chancellor_id: ID of the chancellor player
            enacted: The policy that was enacted

        Returns:
            RoundResult with updated probabilities
        """
        self.round_num += 1

        # Store beliefs before update
        pres_before = self.player_beliefs.priors[president_id]
        chanc_before = self.player_beliefs.priors[chancellor_id]

        # Update player beliefs
        pres_prob, chanc_prob = self.player_beliefs.update(
            president_id,
            chancellor_id,
            enacted,
            self.deck_state,
            self.draw_count,
            self.pass_count,
        )

        # Update deck state
        joint_probs = self.player_beliefs.get_joint_probability(president_id, chancellor_id)
        self.deck_state.update(enacted, joint_probs, self.draw_count, self.pass_count)

        # Get all player beliefs and deck expected for result
        all_beliefs = dict(self.player_beliefs.priors)
        deck_expected = self.deck_state.get_expected_composition()

        # Create result
        result = RoundResult(
            round_num=self.round_num,
            president_id=president_id,
            chancellor_id=chancellor_id,
            enacted=enacted,
            president_prob_bad=pres_prob,
            chancellor_prob_bad=chanc_prob,
            president_prob_before=pres_before,
            chancellor_prob_before=chanc_before,
            top_deck_states=self.deck_state.top_states(5),
            all_player_beliefs=all_beliefs,
            deck_expected=deck_expected,
        )

        self.history.append(result)
        return result

    def reset_deck(self, bad_policies: int, good_policies: int) -> None:
        """Reset the deck state to a fresh deck (e.g., after reshuffle)."""
        self.deck_state.reset(bad_policies, good_policies)

    def print_round(self, result: RoundResult) -> None:
        """Print a formatted summary of a round."""
        print(f"\n{'=' * 50}")
        print(f"ROUND {result.round_num}")
        print(f"{'=' * 50}")
        print(f"President: Player {result.president_id}")
        print(f"Chancellor: Player {result.chancellor_id}")
        print(f"Enacted Policy: {result.enacted.value}")
        print()

        print("Player Beliefs (P(bad)):")
        print(
            f"  Player {result.president_id} (President): "
            f"{result.president_prob_bad:.1%} (was {result.president_prob_before:.1%})"
        )
        print(
            f"  Player {result.chancellor_id} (Chancellor): "
            f"{result.chancellor_prob_bad:.1%} (was {result.chancellor_prob_before:.1%})"
        )
        print()

        print("Top Deck States (bad, good):")
        for deck, prob in result.top_deck_states:
            print(f"  ({deck.bad:2d} bad, {deck.good:2d} good): {prob:.1%}")

    def get_all_player_beliefs(self) -> dict[int, float]:
        """Get current beliefs for all players."""
        return dict(self.player_beliefs.priors)


def main():
    """Run a demonstration of the simulation."""
    print("=" * 60)
    print("POLICY GAME BAYESIAN SIMULATION")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Initial Deck: {BAD_POLICIES} bad, {GOOD_POLICIES} good policies")
    print("  Draw Count: 3")
    print("  Pass Count: 2")
    print(f"  Players: {NUM_PLAYERS} (IDs: 0-{NUM_PLAYERS - 1})")
    print("  Prior P(bad): 50%")
    print()
    print("Behavioral Assumptions (Optimal Play):")
    print("  - Good players always try to enact good policies")
    print("  - Bad players always try to enact bad policies")

    sim = GameSimulation()

    # Simulate a few rounds with different outcomes
    scenarios = [
        (0, 1, Policy.BAD, "A bad policy is enacted"),
        (1, 0, Policy.BAD, "Another bad policy is enacted"),
        (0, 1, Policy.GOOD, "Finally a good policy!"),
    ]

    for pres, chanc, policy, description in scenarios:
        print(f"\n>>> {description}")
        result = sim.play_round(pres, chanc, policy)
        sim.print_round(result)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print("\nFinal Player Beliefs:")
    for player_id, prob in sim.get_all_player_beliefs().items():
        print(f"  Player {player_id}: {prob:.1%} chance of being bad")

    exp_bad, exp_good = sim.deck_state.get_expected_composition()
    print(f"\nExpected Deck: {exp_bad:.1f} bad, {exp_good:.1f} good policies remaining")


if __name__ == "__main__":
    main()
