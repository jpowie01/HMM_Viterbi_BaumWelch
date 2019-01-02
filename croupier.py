from typing import Tuple

import numpy as np

import dice
import utils
from dice_type import DiceType


class Croupier:
    def __init__(self, fair_dice: dice.Dice, loaded_dice: dice.Dice,
                 initial_dice_probability: np.ndarray, transition_matrix: np.ndarray) -> None:
        self.fair_dice = fair_dice
        self.loaded_dice = loaded_dice
        self.initial_dice_probability = initial_dice_probability
        self.transition_matrix = transition_matrix
        self.current_dice = utils.weighted_random(self.initial_dice_probability)

    def get_next_value(self) -> Tuple[int, int]:
        # Try to switch between dices
        self.current_dice = utils.weighted_random(self.transition_matrix[self.current_dice])

        # Let's roll the dice!
        if self.current_dice == DiceType.FAIR:
            return self.current_dice, self.fair_dice.get_next_value()
        else:
            return self.current_dice, self.loaded_dice.get_next_value()

