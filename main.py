import numpy as np

import dice
import croupier


fair_dice = dice.Dice(np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]))
loaded_dice = dice.Dice(np.array([1/10, 1/10, 1/10, 1/10, 1/10, 1/2]))
transition_matrix = np.array([
    [0.95, 0.05],  # Fair -> Fair = 0.95, Fair -> Loaded = 0.05
    [0.10, 0.90],  # Loaded -> Fair = 0.10, Loaded -> Loaded = 0.90
])
my_croupier = croupier.Croupier(fair_dice, loaded_dice, transition_matrix)

for i in range(1000):
    used_dice, dice_wall = my_croupier.get_next_value()
    print(f'Roll #{i}: {used_dice} -> {dice_wall}')

