import random

import numpy as np


def weighted_random(probabilities: np.ndarray) -> int:
    probability_so_far = 0.0
    random_value = random.random()
    for value, probability in enumerate(probabilities):
        probability_so_far += probability
        if probability_so_far > random_value:
            return value

    # By default return N-th possible value
    return probabilities.shape[0] - 1
