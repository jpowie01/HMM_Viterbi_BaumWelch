import numpy as np

import algorithms
import croupier
import dice
from dice_type import DiceType, AVAILABLE_DICES

np.set_printoptions(precision=3)

OBSERVATION_LENGTH = 300
OBSERVATIONS = 300

# Prepare environment for our simulation
# fair_dice = dice.Dice(np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]))
# loaded_dice = dice.Dice(np.array([1/10, 1/10, 1/10, 1/10, 1/10, 1/2]))
fair_dice = dice.Dice(np.array([1/5, 1/5, 1/5, 1/5, 1/5, 0]))
loaded_dice = dice.Dice(np.array([0, 0, 0, 0, 0, 1]))
initial_dice_probability = np.array([0.5, 0.5])  # Croupier can use either of dices initially
transition_matrix = np.array([
    [0.95, 0.05],  # Fair -> Fair = 0.95, Fair -> Loaded = 0.05
    [0.10, 0.90],  # Loaded -> Fair = 0.10, Loaded -> Loaded = 0.90
])
my_croupier = croupier.Croupier(fair_dice, loaded_dice, initial_dice_probability, transition_matrix)

# Collect some observations about the croupier's work for later analysis
observations, used_dices = my_croupier.get_observations(OBSERVATION_LENGTH)

#
# Viterbi & Aposteriori Probabilities
#

# Pass observations through Viterbi algorithm
viterbi_probabilities = algorithms.calculate_viterbi_probabilities(observations, fair_dice, loaded_dice,
                                                                   initial_dice_probability, transition_matrix)

# Calculate forward & backward probabilities to calculate aposteriori probabilities
alpha_probabilities = algorithms.calculate_forward_probabilities(observations, fair_dice, loaded_dice,
                                                                 initial_dice_probability, transition_matrix)
beta_probabilities = algorithms.calculate_backward_probabilities(observations, fair_dice, loaded_dice,
                                                                 initial_dice_probability, transition_matrix)

# Now, let's calculate aposteriori probabilities based on alphas and betas
aposteriori_probabilities = np.multiply(alpha_probabilities, beta_probabilities)

print('+---------------------------+')
print('|   Viterbi & Aposteriori   |')
print('+---------------------------+')
for t in range(OBSERVATIONS):
    print(f'Observation: {observations[t]} Dice: {used_dices[t]} '
          f'| Viterbi: {viterbi_probabilities[t]} Guess: {np.argmax(viterbi_probabilities[t])} '
          f'| Alpha: {alpha_probabilities[t]} | Beta: {beta_probabilities[t]} '
          f'| Aposteriori: {aposteriori_probabilities[t]} Guess: {np.argmax(aposteriori_probabilities[t])}')
print('')

#
# Baum-Welch Training
#

print('+----------------------------+')
print('|    Baum-Welch Algorithm    |')
print('+----------------------------+')

# Pass training observations to Baum-Welch algorightm
multiple_observations = [my_croupier.get_observations(OBSERVATION_LENGTH)[0] for _ in range(OBSERVATIONS)]
first_dice, second_dice, initial_dice_probability, transition_matrix = algorithms.baum_welch(multiple_observations)

print('+--------------------------+')
print('|    Baum-Welch Summary    |')
print('+--------------------------+')
print('Dice #1:', first_dice.probabilities)
print('Dice #2:', second_dice.probabilities)
print('Initial Dice Probabilities:', initial_dice_probability)
print('Transision Matrix:', transition_matrix)
