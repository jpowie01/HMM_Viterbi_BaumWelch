import random

import numpy as np

import utils


class Dice:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities

    def get_next_value(self) -> int:
        return utils.weightned_random(self.probabilities)

