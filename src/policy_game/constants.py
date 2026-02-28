# Game configuration constants
NUM_PLAYERS = 6
NUM_FASCISTS = 2
FASCIST_POLICIES = 11
LIBERAL_POLICIES = 6
INITIAL_PRIOR_FASCIST_PROB = NUM_FASCISTS / NUM_PLAYERS

# Executive powers granted after each Fascist policy (6-player board)
# Index 0 = 1st Fascist policy, Index 4 = 5th Fascist policy
FASCIST_BOARD_POWERS_6P = [None, None, "POLICY_PEEK", "EXECUTION", "EXECUTION"]
