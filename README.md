# Policy Game Bayesian Simulation

A simulation for social deduction games where players draw policies and we use Bayesian inference to estimate the probability that each player is "bad" or "good" based on observed policy outcomes. 

## Game Mechanics

- **Policy Deck**: Configurable bad/good policies (default: 11 bad, 6 good)
- **President** draws N policies (default: 3), discards 1, passes N-1 to Chancellor
- **Chancellor** discards 1, enacts the remaining policy
- All drawn policies are removed from the deck

## Installation

```bash
uv sync --dev
```

## Usage

### Run the demo simulation

```bash
python -m policy_game.simulation
```

### Generate a game for visualization

```bash
python -m policy_game.generator
```

This creates `data/game_data.json` which can be viewed with `data/visualization.html`.

### Use as a library

```python
from policy_game import GameSimulation, Policy

sim = GameSimulation(
    bad_policies=11,
    good_policies=6,
    num_players=6,
    prior_bad_prob=1/3,  # 2 bad players out of 6
)

# Play a round
result = sim.play_round(president_id=0, chancellor_id=1, enacted=Policy.BAD)

# Check updated beliefs
print(sim.get_all_player_beliefs())
```

## Project Structure

```
game-sim/
├── src/policy_game/
│   ├── __init__.py      # Package exports
│   ├── core.py          # Policy, Draw, DeckComposition, strategy functions
│   ├── simulation.py    # DeckState, PlayerBeliefs, GameSimulation
│   └── generator.py     # Game generation for visualization
├── tests/
│   ├── test_core.py     # Tests for core models and strategies
│   └── test_simulation.py
├── data/
│   ├── game_data.json   # Generated game data
│   └── visualization.html
└── pyproject.toml
```

## Running Tests

```bash
uv run pytest
```