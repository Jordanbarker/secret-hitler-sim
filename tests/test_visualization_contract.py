"""
Tests for visualization interface contract.

These tests ensure the Python backend generates data in the exact format
that visualization.html expects. Any changes that break these tests will
likely break the frontend visualization.
"""

import pytest

from policy_game.generator import generate_game


class TestSchemaValidation:
    """Verify game output has all required fields for visualization."""

    def test_generate_game_schema(self):
        """Verify game output has all required top-level fields."""
        game = generate_game(seed=42)

        # Top-level structure
        assert "config" in game
        assert "initial_state" in game
        assert "rounds" in game
        assert "final_state" in game

        # Config fields
        assert "num_players" in game["config"]
        assert "bad_players" in game["config"]
        assert "hitler_id" in game["config"]
        assert "initial_deck" in game["config"]
        assert "prior_bad_prob" in game["config"]

        # Initial state fields
        assert "player_beliefs" in game["initial_state"]
        assert "deck_expected" in game["initial_state"]

        # All player belief keys are strings
        for key in game["initial_state"]["player_beliefs"]:
            assert isinstance(key, str), f"Expected string key, got {type(key)}"


class TestEnumValues:
    """Verify enum values are exactly what visualization.html expects."""

    def test_enum_values_match_visualization(self):
        """Verify enum values use exact strings expected by CSS/JS."""
        game = generate_game(seed=42)

        for round_data in game["rounds"]:
            # Round type must be exact string
            assert round_data["round_type"] in [
                "LEGISLATIVE",
                "FAILED_ELECTION",
                "CHAOS",
            ], f"Invalid round_type: {round_data['round_type']}"

            # Enacted policy (if present) must be exact string for CSS class
            if "enacted" in round_data:
                assert round_data["enacted"] in [
                    "BAD",
                    "GOOD",
                ], f"Invalid enacted value: {round_data['enacted']}"

            # Votes must be exact strings
            for player_id, vote in round_data["votes"].items():
                assert vote in [
                    "JA",
                    "NEIN",
                ], f"Invalid vote value: {vote} for player {player_id}"

            # Winner (if present) must be exact string
            if "winner" in round_data:
                assert round_data["winner"] in [
                    "FASCIST",
                    "LIBERAL",
                ], f"Invalid winner value: {round_data['winner']}"


class TestPlayerBeliefsFormat:
    """Ensure player_beliefs uses string keys (JavaScript requirement)."""

    def test_player_beliefs_string_keys(self):
        """Player beliefs must use string keys for JS object access."""
        game = generate_game(seed=42)

        # Check initial state
        beliefs = game["initial_state"]["player_beliefs"]
        expected_keys = {"0", "1", "2", "3", "4", "5"}
        assert set(beliefs.keys()) == expected_keys, (
            f"Initial state beliefs have wrong keys: {set(beliefs.keys())}"
        )

        # Check each round
        for i, round_data in enumerate(game["rounds"]):
            beliefs = round_data["player_beliefs"]
            assert set(beliefs.keys()) == expected_keys, (
                f"Round {i + 1} beliefs have wrong keys: {set(beliefs.keys())}"
            )

    def test_player_beliefs_are_floats(self):
        """Player belief values must be floats between 0 and 1."""
        game = generate_game(seed=42)

        # Check initial state
        for player_id, prob in game["initial_state"]["player_beliefs"].items():
            assert isinstance(prob, float), (
                f"Initial belief for player {player_id} is not float: {type(prob)}"
            )
            assert 0 <= prob <= 1, f"Initial belief for player {player_id} out of range: {prob}"

        # Check each round
        for round_data in game["rounds"]:
            for player_id, prob in round_data["player_beliefs"].items():
                assert isinstance(prob, (int, float)), (
                    f"Belief for player {player_id} is not numeric: {type(prob)}"
                )
                assert 0 <= prob <= 1, f"Belief for player {player_id} out of range: {prob}"


class TestDeckStructure:
    """Verify deck objects have required bad/good fields."""

    def test_deck_structure(self):
        """Deck objects must have 'bad' and 'good' fields."""
        game = generate_game(seed=42)

        # Initial deck in config
        assert "bad" in game["config"]["initial_deck"]
        assert "good" in game["config"]["initial_deck"]

        # Initial state deck_expected
        assert "bad" in game["initial_state"]["deck_expected"]
        assert "good" in game["initial_state"]["deck_expected"]

        # Each round must have deck_expected
        for i, round_data in enumerate(game["rounds"]):
            assert "deck_expected" in round_data, f"Round {i + 1} missing deck_expected"
            assert "bad" in round_data["deck_expected"], (
                f"Round {i + 1} deck_expected missing 'bad'"
            )
            assert "good" in round_data["deck_expected"], (
                f"Round {i + 1} deck_expected missing 'good'"
            )


class TestPoliciesEnactedStructure:
    """Verify policies_enacted has bad/good counts."""

    def test_policies_enacted_structure(self):
        """policies_enacted must have 'bad' and 'good' integer counts."""
        game = generate_game(seed=42)

        for i, round_data in enumerate(game["rounds"]):
            assert "policies_enacted" in round_data, f"Round {i + 1} missing policies_enacted"

            pe = round_data["policies_enacted"]
            assert "bad" in pe, f"Round {i + 1} policies_enacted missing 'bad'"
            assert "good" in pe, f"Round {i + 1} policies_enacted missing 'good'"
            assert isinstance(pe["bad"], int), (
                f"Round {i + 1} policies_enacted['bad'] is not int: {type(pe['bad'])}"
            )
            assert isinstance(pe["good"], int), (
                f"Round {i + 1} policies_enacted['good'] is not int: {type(pe['good'])}"
            )
            assert pe["bad"] >= 0, f"Round {i + 1} policies_enacted['bad'] is negative"
            assert pe["good"] >= 0, f"Round {i + 1} policies_enacted['good'] is negative"

    def test_policies_enacted_monotonic(self):
        """policies_enacted counts should never decrease."""
        game = generate_game(seed=42)

        prev_bad = 0
        prev_good = 0

        for i, round_data in enumerate(game["rounds"]):
            pe = round_data["policies_enacted"]
            assert pe["bad"] >= prev_bad, (
                f"Round {i + 1} bad policies decreased: {pe['bad']} < {prev_bad}"
            )
            assert pe["good"] >= prev_good, (
                f"Round {i + 1} good policies decreased: {pe['good']} < {prev_good}"
            )
            prev_bad = pe["bad"]
            prev_good = pe["good"]


class TestRoundRequiredFields:
    """Verify each round has all required fields."""

    def test_round_required_fields(self):
        """Each round must have core required fields."""
        game = generate_game(seed=42)

        for i, round_data in enumerate(game["rounds"]):
            round_num = i + 1

            assert "round_num" in round_data, f"Round {round_num} missing round_num"
            assert "round_type" in round_data, f"Round {round_num} missing round_type"
            assert "president_id" in round_data, f"Round {round_num} missing president_id"
            assert "chancellor_id" in round_data, f"Round {round_num} missing chancellor_id"
            assert "vote_passed" in round_data, f"Round {round_num} missing vote_passed"
            assert "votes" in round_data, f"Round {round_num} missing votes"
            assert "election_tracker" in round_data, f"Round {round_num} missing election_tracker"
            assert "player_beliefs" in round_data, f"Round {round_num} missing player_beliefs"
            assert "deck_expected" in round_data, f"Round {round_num} missing deck_expected"
            assert "policies_enacted" in round_data, f"Round {round_num} missing policies_enacted"

    def test_votes_structure(self):
        """Votes must have string keys for all players."""
        game = generate_game(seed=42)
        num_players = game["config"]["num_players"]
        expected_keys = {str(i) for i in range(num_players)}

        for i, round_data in enumerate(game["rounds"]):
            votes = round_data["votes"]
            assert set(votes.keys()) == expected_keys, (
                f"Round {i + 1} votes have wrong keys: {set(votes.keys())}"
            )


class TestFinalState:
    """Verify final_state structure."""

    def test_final_state_structure(self):
        """final_state should have game result fields."""
        game = generate_game(seed=42)

        assert "final_state" in game
        final = game["final_state"]

        assert "game_over" in final
        assert "policies_enacted" in final
        assert "bad" in final["policies_enacted"]
        assert "good" in final["policies_enacted"]

        # If game is over, winner and win_condition should be present
        if final["game_over"]:
            assert "winner" in final
            assert final["winner"] in ["FASCIST", "LIBERAL", None]
            assert "win_condition" in final


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
