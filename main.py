import numpy as np

import algorithms
import croupier
import dice
from dice_type import DiceType, AVAILABLE_DICES

np.set_printoptions(precision=3)

OBSERVATIONS = 100

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
# All needed calculations
#

# Pass observations through Viterbi algorithm
viterbi_probabilities = algorithms.calculate_viterbi_probabilities(observations, my_croupier, fair_dice, loaded_dice,
                                                                   initial_dice_probability, transition_matrix)

# Calculate forward & backward probabilities to calculate aposteriori probabilities
alpha_probabilities = algorithms.calculate_forward_probabilities(observations, my_croupier, fair_dice, loaded_dice,
                                                                 initial_dice_probability, transition_matrix)
beta_probabilities = algorithms.calculate_backward_probabilities(observations, my_croupier, fair_dice, loaded_dice,
                                                                 initial_dice_probability, transition_matrix)

# Now, let's calculate aposteriori probabilities based on alphas and betas
aposteriori_probabilities = np.multiply(alpha_probabilities, beta_probabilities)

#
# Summary of above algorithms
#

print('+---------------------------+')
print('|   Viterbi & Aposteriori   |')
print('+---------------------------+')
for t in range(observations_rows):
    print(f'Observation: {observations[t]} Dice: {used_dices[t]} '
          f'| Viterbi: {viterbi_probabilities[t]} Guess: {np.argmax(viterbi_probabilities[t])} '
          f'| Alpha: {alpha_probabilities[t]} | Beta: {beta_probabilities[t]} '
          f'| Aposteriori: {aposteriori_probabilities[t]} Guess: {np.argmax(aposteriori_probabilities[t])}')
