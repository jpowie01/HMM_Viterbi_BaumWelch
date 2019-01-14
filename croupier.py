from typing import Tuple

import numpy as np

import dice
import utils
from dice_type import DiceType


class Croupier:
    def __init__(self, fair_dice: dice.Dice, loaded_dice: dice.Dice,
                 initial_dice_probability: np.ndarray, transition_matrix: np.ndarray) -> None:
        self._fair_dice = fair_dice
        self._loaded_dice = loaded_dice
        self._initial_dice_probability = initial_dice_probability
        self._transition_matrix = transition_matrix
        self._current_dice = utils.weighted_random(self._initial_dice_probability)

    def get_next_value(self) -> Tuple[int, int]:
        # Try to switch between dices
        self._current_dice = utils.weighted_random(self._transition_matrix[self._current_dice])

        # Let's roll the dice!
        if self._current_dice == DiceType.FAIR:
            return self._current_dice, self._fair_dice.get_next_value()
        else:
            return self._current_dice, self._loaded_dice.get_next_value()

