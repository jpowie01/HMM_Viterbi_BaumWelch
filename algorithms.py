from typing import Tuple

import numpy as np

import croupier
import dice
from dice_type import DiceType, AVAILABLE_DICES


def calculate_viterbi_probabilities(observations: np.ndarray, fair_dice: dice.Dice, loaded_dice: dice.Dice,
                                    initial_dice_probability: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    observations_rows = observations.shape[0]

    # Initial probabilities of each dices depends on random choice of croupier and probability of an observation
    viterbi_probabilities = np.zeros([observations_rows, AVAILABLE_DICES])
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

    return viterbi_probabilities

def calculate_forward_probabilities(observations: np.ndarray, fair_dice: dice.Dice, loaded_dice: dice.Dice,
                                    initial_dice_probability: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    observations_rows = observations.shape[0]

    # Initial probabilities for alpha equals probability of random choice of a given dice and roll of given observation
    alpha_probabilities = np.zeros([observations_rows, AVAILABLE_DICES])
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

    return alpha_probabilities


def calculate_backward_probabilities(observations: np.ndarray, fair_dice: dice.Dice, loaded_dice: dice.Dice,
                                     initial_dice_probability: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    observations_rows = observations.shape[0]

    # Initial probabilities for beta equals to 1 for each dice. Keep in mind that we need to iterate from the back!
    beta_probabilities = np.zeros([observations_rows, AVAILABLE_DICES])
    beta_probabilities[observations_rows-1] = np.array([1, 1])

    for t, observation in reversed(list(enumerate(observations))[1:]):
        beta_probabilities[t-1] = np.array([
            np.sum([
                # In the observation T croupier used fair dice and fair dice in T-1
                transition_matrix[DiceType.FAIR, DiceType.FAIR] * fair_dice.probabilities[observation]
                  * beta_probabilities[t][DiceType.FAIR],
                # In the observation T croupier used loaded dice and fair dice in T-1
                transition_matrix[DiceType.FAIR, DiceType.LOADED] * loaded_dice.probabilities[observation]
                  * beta_probabilities[t][DiceType.LOADED],
            ]),
            np.sum([
                # In the observation T croupier used fair dice and loaded dice in T-1
                transition_matrix[DiceType.LOADED, DiceType.FAIR] * fair_dice.probabilities[observation]
                  * beta_probabilities[t][DiceType.FAIR],
                # In the observation T croupier used loaded dice and loaded dice in T-1
                transition_matrix[DiceType.LOADED, DiceType.LOADED] * loaded_dice.probabilities[observation]
                  * beta_probabilities[t][DiceType.LOADED],
            ]),
        ])

        # Again, as previously, normalize all probabilities, so that values won't be too low over time
        beta_probabilities[t-1] /= np.sum(beta_probabilities[t-1])

    return beta_probabilities


def baum_welch(observations: np.ndarray, epochs:int = 50) -> Tuple[dice.Dice, dice.Dice, np.ndarray, np.ndarray]:
    observations_rows = observations.shape[0]

    first_dice_probabilities = np.random.rand(6)
    first_dice_probabilities /= first_dice_probabilities.sum()  # Make probabilities sum up to 1.0
    first_dice = dice.Dice(first_dice_probabilities)

    second_dice_probabilities = np.random.rand(6)
    second_dice_probabilities /= second_dice_probabilities.sum()  # Make probabilities sum up to 1.0
    second_dice = dice.Dice(second_dice_probabilities)

    initial_dice_probability = np.random.rand(2)
    initial_dice_probability /= initial_dice_probability.sum()  # Make probabilities sum up to 1.0

    transition_matrix = np.random.rand(2, 2)
    transition_matrix /= np.sum(transition_matrix, axis=1)[:, None]  # Make probabilities sum up to 1.0 for each dice

    for epoch in range(epochs):
        alpha_probabilities = calculate_forward_probabilities(observations, first_dice, second_dice,
                                                              initial_dice_probability, transition_matrix)
        beta_probabilities = calculate_backward_probabilities(observations, first_dice, second_dice,
                                                              initial_dice_probability, transition_matrix)
        aposteriori_probabilities = np.multiply(alpha_probabilities, beta_probabilities)

        ksi = np.zeros([observations_rows, 2, 2])
        for t in range(observations_rows-1):
            denominator = (alpha_probabilities[t][0] * beta_probabilities[t][0] + alpha_probabilities[t][1] * beta_probabilities[t][1])
            ksi[t, 0, 0] = (alpha_probabilities[t][0] * transition_matrix[0, 0]
                             * first_dice_probabilities[observations[t+1]] * beta_probabilities[t+1][0]) / denominator
            ksi[t, 0, 1] = (alpha_probabilities[t][0] * transition_matrix[0, 1]
                             * second_dice_probabilities[observations[t+1]] * beta_probabilities[t+1][1]) / denominator
            ksi[t, 1, 0] = (alpha_probabilities[t][1] * transition_matrix[1, 0]
                             * first_dice_probabilities[observations[t+1]] * beta_probabilities[t+1][0]) / denominator
            ksi[t, 1, 1] = (alpha_probabilities[t][1] * transition_matrix[1, 1]
                             * second_dice_probabilities[observations[t+1]] * beta_probabilities[t+1][1]) / denominator

        gamma = np.zeros([observations_rows, 2])
        for t in range(observations_rows-1):
            gamma[t, 0] = ksi[t, 0, 0] * ksi[t, 0, 1]
            gamma[t, 1] = ksi[t, 1, 0] * ksi[t, 1, 1]

        initial_dice_probability = gamma[0] / np.sum(gamma[0])

        transition_matrix[0, 0] = np.sum(ksi[:, 0, 0]) / np.sum(gamma[:, 0])
        transition_matrix[0, 1] = np.sum(ksi[:, 0, 1]) / np.sum(gamma[:, 0])
        transition_matrix[1, 0] = np.sum(ksi[:, 1, 0]) / np.sum(gamma[:, 1])
        transition_matrix[1, 1] = np.sum(ksi[:, 1, 1]) / np.sum(gamma[:, 1])

        for wall in range(6):
            first_dice_probabilities[wall] = np.sum(np.take(aposteriori_probabilities[:, 0], np.where(observations == wall))) / np.sum(aposteriori_probabilities[:, 0])
            second_dice_probabilities[wall] = np.sum(np.take(aposteriori_probabilities[:, 1], np.where(observations == wall))) / np.sum(aposteriori_probabilities[:, 1])

        transition_matrix /= np.sum(transition_matrix, axis=1)[:, None]  # Make probabilities sum up to 1.0 for each dice
        first_dice_probabilities /= first_dice_probabilities.sum()  # Make probabilities sum up to 1.0
        second_dice_probabilities /= second_dice_probabilities.sum()  # Make probabilities sum up to 1.0

        first_dice = dice.Dice(first_dice_probabilities)
        second_dice = dice.Dice(second_dice_probabilities)

        print('= EPOCH #{} ='.format(epoch))
        print('Transition Matrix:', transition_matrix)
        print('Initial Dice Probability:', initial_dice_probability)
        print('First Dice:', first_dice_probabilities)
        print('Second Dice:', second_dice_probabilities)
        print()

    return first_dice, second_dice, initial_dice_probability, transition_matrix
