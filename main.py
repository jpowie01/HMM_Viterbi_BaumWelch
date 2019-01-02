import numpy as np

import dice
import croupier
from dice_type import DiceType

np.set_printoptions(precision=3)

OBSERVATIONS = 100
DICE_TYPES = 2

# Prepare environment for our simulation
fair_dice = dice.Dice(np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]))
loaded_dice = dice.Dice(np.array([1/10, 1/10, 1/10, 1/10, 1/10, 1/2]))
initial_dice_probability = np.array([0.5, 0.5])  # Croupier can use either of dices initially
transition_matrix = np.array([
    [0.95, 0.05],  # Fair -> Fair = 0.95, Fair -> Loaded = 0.05
    [0.10, 0.90],  # Loaded -> Fair = 0.10, Loaded -> Loaded = 0.90
])
my_croupier = croupier.Croupier(fair_dice, loaded_dice, initial_dice_probability, transition_matrix)

# Collect some observations about the croupier's work for later analysis
used_dices = []
observations = []
for i in range(OBSERVATIONS):
    used_dice, dice_wall = my_croupier.get_next_value()
    used_dices.append(used_dice)
    observations.append(dice_wall)
observations = np.array(observations)
observations_rows = observations.shape[0]

#
# Viterbi Algorithm
#

# Initial probabilities of each dices depends on random choice of croupier and probability of an observation
viterbi_probabilities = np.zeros([observations_rows, DICE_TYPES])
viterbi_probabilities[0] = np.array([
    initial_dice_probability[DiceType.FAIR] * fair_dice.probabilities[observations[0]],
    initial_dice_probability[DiceType.LOADED] * loaded_dice.probabilities[observations[0]],
])

# Iteratively compute dice probabilities based on previous observations and probabilities
for t, observation in enumerate(observations[1:], 1):
    viterbi_probabilities[t] = np.array([
        fair_dice.probabilities[observation] * np.max([
            # Previously we've used fair dice and now we use fair dice
            viterbi_probabilities[t-1][DiceType.FAIR] * transition_matrix[DiceType.FAIR, DiceType.FAIR],
            # Previously we've used loaded dice and now we use fair dice
            viterbi_probabilities[t-1][DiceType.LOADED] * transition_matrix[DiceType.LOADED, DiceType.FAIR],
        ]),
        loaded_dice.probabilities[observation] * np.max([
            # Previously we've used fair dice and now we use loaded dice
            viterbi_probabilities[t-1][DiceType.FAIR] * transition_matrix[DiceType.FAIR, DiceType.LOADED],
            # Previously we've used loaded dice and now we use loaded dice
            viterbi_probabilities[t-1][DiceType.LOADED] * transition_matrix[DiceType.LOADED, DiceType.LOADED],
        ]),
    ])

    # Normalize current probabilities, so that values won't be too low over time
    viterbi_probabilities[t] /= np.sum(viterbi_probabilities[t])

#
# Forward Algorithm
#

# Initial probabilities for alpha equals probability of random choice of a given dice and roll of given observation
alpha_probabilities = np.zeros([observations_rows, DICE_TYPES])
alpha_probabilities[0] = np.array([
    initial_dice_probability[DiceType.FAIR] * fair_dice.probabilities[observations[0]],
    initial_dice_probability[DiceType.LOADED] * loaded_dice.probabilities[observations[0]],
])

# Iteratively compute probabilities based on previous ones
for t, observation in enumerate(observations[1:], 1):
    alpha_probabilities[t] = np.array([
        fair_dice.probabilities[observation] * np.sum([
            # Previously we've used fair dice and now we use fair dice
            alpha_probabilities[t-1][DiceType.FAIR] * transition_matrix[DiceType.FAIR, DiceType.FAIR],
            # Previously we've used loaded dice and now we use fair dice
            alpha_probabilities[t-1][DiceType.LOADED] * transition_matrix[DiceType.LOADED, DiceType.FAIR]
        ]),
        loaded_dice.probabilities[observation] * np.sum([
            # Previously we've used fair dice and now we use loaded dice
            alpha_probabilities[t-1][DiceType.FAIR] * transition_matrix[DiceType.FAIR, DiceType.LOADED],
            # Previously we've used loaded dice and now we use loaded dice
            alpha_probabilities[t-1][DiceType.LOADED] * transition_matrix[DiceType.LOADED, DiceType.LOADED]
        ]),
    ])

    # Normalize current probabilities, so that values won't be too low over time
    alpha_probabilities[t] /= np.sum(alpha_probabilities[t])

#
# Backward Algorithm
#

# Initial probabilities for beta equals to 1 for each dice. Keep in mind that we need to iterate from the back!
beta_probabilities = np.zeros([observations_rows, DICE_TYPES])
beta_probabilities[observations_rows-1] = np.array([1, 1])

for t, observation in reversed(list(enumerate(observations))[1:]):
    beta_probabilities[t-1] = np.array([
        np.sum([
            # In the observation T croupier used fair dice and fair dice in T-1
            transition_matrix[DiceType.FAIR, DiceType.FAIR] * fair_dice.probabilities[observation] * beta_probabilities[t][DiceType.FAIR],
            # In the observation T croupier used loaded dice and fair dice in T-1
            transition_matrix[DiceType.FAIR, DiceType.LOADED] * loaded_dice.probabilities[observation] * beta_probabilities[t][DiceType.LOADED],
        ]),
        np.sum([
            # In the observation T croupier used fair dice and loaded dice in T-1
            transition_matrix[DiceType.LOADED, DiceType.FAIR] * fair_dice.probabilities[observation] * beta_probabilities[t][DiceType.FAIR],
            # In the observation T croupier used loaded dice and loaded dice in T-1
            transition_matrix[DiceType.LOADED, DiceType.LOADED] * loaded_dice.probabilities[observation] * beta_probabilities[t][DiceType.LOADED],
        ]),
    ])

    # Again, as previously, normalize all probabilities, so that values won't be too low over time
    beta_probabilities[t-1] /= np.sum(beta_probabilities[t-1])

# Now, let's calculate aposteriori probabilities based on alphas and betas
aposteriori_probabilities = np.multiply(alpha_probabilities, beta_probabilities)

#
# Summary of above algorithms
#

# Print all the things we've collected so far
for t in range(observations_rows):
    print(f'Observation: {observations[t]} Dice: {used_dices[t]} | Viterbi: {viterbi_probabilities[t]} Guess: {np.argmax(viterbi_probabilities[t])} '
          f'| Alpha: {alpha_probabilities[t]} | Beta: {beta_probabilities[t]} | Aposteriori: {aposteriori_probabilities[t]} Guess: {np.argmax(aposteriori_probabilities[t])}')
