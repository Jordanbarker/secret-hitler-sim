# Policy Game Bayesian Simulation

A simulation for social deduction games where players draw policies and we use Bayesian inference to estimate the probability that each player is facist. 

## Game Mechanics

- **Policy Deck**: Configurable bad/good policies (default: 11 bad, 6 good)
- **President** draws N policies (default: 3), discards 1, passes N-1 to Chancellor
- **Chancellor** discards 1, enacts the remaining policy
- All drawn policies are removed from the deck

## Game Logic Overview

`generate_game()` in `generator.py` is the entry point. It assigns roles (`PlayerRoles`), initializes Bayesian tracking (`GameSimulation`) and game state (`GameState`), then runs a main loop where each round:

1. **President rotates** and nominates a chancellor from eligible players (term-limited)
2. **All players vote** using role-specific strategies (liberal, fascist, hitler)
3. **If vote fails** — election tracker increments; 3 failures triggers chaos (top policy enacted)
4. **If vote passes** — president draws 3, discards 1; chancellor picks 1 of 2 to enact; Bayesian beliefs update

## Project Structure

```
game-sim/
├── src/policy_game/
│   ├── __init__.py      # Package exports
│   ├── constants.py     # Game constants (deck size, player count, etc.)
│   ├── core.py          # Policy, Draw, DeckComposition, strategy functions
│   ├── simulation.py    # DeckState, PlayerBeliefs, GameSimulation
│   └── generator.py     # Game generation for visualization
├── tests/
│   ├── test_core.py
│   ├── test_simulation.py
│   └── test_visualization_contract.py
├── scripts/
│   └── verify_visualization.py
├── data/
│   └── game_data.json   # Generated game data
├── index.html           # Interactive game visualization
└── pyproject.toml
```