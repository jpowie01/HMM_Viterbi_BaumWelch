import enum
import random
from typing import Tuple

import numpy as np

import dice


FAIR_DICE = 0
LOADED_DICE = 1


class Croupier:
    def __init__(self, fair_dice: dice.Dice, loaded_dice: dice.Dice, transition_matrix: np.ndarray) -> None:
        self.current_dice = FAIR_DICE
        self.fair_dice = fair_dice
        self.loaded_dice = loaded_dice
        self.transition_matrix = transition_matrix

    def get_next_value(self) -> Tuple[int, int]:
        # Try to switch between dices
        probability_so_far = 0.0
        random_value = random.random()
        for next_dice, probability in enumerate(self.transition_matrix[self.current_dice]):
            probability_so_far += probability
            if probability_so_far > random_value:
                self.current_dice = next_dice
                break

        # Let's roll the dice!
        if self.current_dice == FAIR_DICE:
            return self.current_dice, self.fair_dice.get_next_value()
        else:
            return self.current_dice, self.loaded_dice.get_next_value()

