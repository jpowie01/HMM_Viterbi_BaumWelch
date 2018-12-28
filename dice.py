import random

import numpy as np


class Dice:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities

    def get_next_value(self) -> int:
        probability_so_far = 0.0
        random_value = random.random()
        for dice_wall, probability in enumerate(self.probabilities):
            probability_so_far += probability
            if probability_so_far > random_value:
                return dice_wall

        # By default return N-th wall
        return self.probabilities.shape[0] - 1

