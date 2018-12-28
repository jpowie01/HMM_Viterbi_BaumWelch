import numpy as np

import dice
import croupier


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
for i in range(100):
    used_dice, dice_wall = my_croupier.get_next_value()
    used_dices.append(used_dice)
    observations.append(dice_wall)
observations = np.array(observations)

#
# Viterbi Algorithm
#

# Initial probabilities of each dices depends on random choise of croupier and probability of an observation
dices_probabilites = np.zeros([observations.shape[0], 2])
dices_probabilites[0] = np.array([
    initial_dice_probability[0] * fair_dice.probabilities[observations[0]],
    initial_dice_probability[1] * loaded_dice.probabilities[observations[0]],
])
print('Probabilities:', dices_probabilites[0], 'Observation:', observations[0], 'Guess:', np.argmax(dices_probabilites[0]), 'Real:', used_dices[0])

# Iterativly compute dice probabilities based on previous observations and probabilities
for t, observation in enumerate(observations[1:], 1):
    dices_probabilites[t] = np.array([
        np.max([
            # Previously we've used fair dice and now we use fair dice
            dices_probabilites[t-1][0] * transition_matrix[0, 0] * fair_dice.probabilities[observation],
            # Previously we've used fair dice and now we use loaded dice
            dices_probabilites[t-1][0] * transition_matrix[0, 1] * loaded_dice.probabilities[observation],
        ]),
        np.max([
            # Previously we've used loaded dice and now we use fair dice
            dices_probabilites[t-1][1] * transition_matrix[1, 0] * fair_dice.probabilities[observation],
            # Previously we've used loaded dice and now we use loaded dice
            dices_probabilites[t-1][1] * transition_matrix[1, 1] * loaded_dice.probabilities[observation],
        ]),
    ])

    # Normalize current probabilities, so that values won't be too low over time
    dices_probabilites[t] /= np.sum(dices_probabilites[t])

    # Print current dice prediction
    print('Probabilities:', dices_probabilites[t], 'Observation:', observation, 'Guess:', np.argmax(dices_probabilites[t]), 'Real:', used_dices[t])

