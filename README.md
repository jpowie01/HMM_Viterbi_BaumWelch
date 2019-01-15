Hidden Markov Models - Viterbi & Baum-Welch Algorithms
======================================================

This repository presents example implementation for Viterbi and Baum-Welch
algorithms implementation in Python 3.6+ using Numpy.

## How to run this example?

```bash
$ virtualenv -p python3.6 venv
$ . ./venv/bin/activate
(venv) $ pip install numpy
(venv) $ python main.py
```

### Example output

For such an environment:

```python
fair_dice = dice.Dice(np.array([1/5, 1/5, 1/5, 1/5, 1/5, 0]))
loaded_dice = dice.Dice(np.array([0, 0, 0, 0, 0, 1]))
initial_dice_probability = np.array([0.5, 0.5])  # Croupier can use either of dices initially
transition_matrix = np.array([
    [0.95, 0.05],  # Fair -> Fair = 0.95, Fair -> Loaded = 0.05
    [0.10, 0.90],  # Loaded -> Fair = 0.10, Loaded -> Loaded = 0.90
])
my_croupier = croupier.Croupier(fair_dice, loaded_dice, initial_dice_probability, transition_matrix)
```

We get such analysis:

```
+---------------------------+
|   Viterbi & Aposteriori   |
+---------------------------+
Observation: 3 Dice: 0 | Viterbi: [0.1 0. ] Guess: 0 | Alpha: [0.1 0. ] | Beta: [0.905 0.095] | Aposteriori: [0.09 0.  ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.905 0.095] | Aposteriori: [0.    0.095] Guess: 1
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.905 0.095] | Aposteriori: [0.905 0.   ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [1. 0.] Guess: 0 | Alpha: [1. 0.] | Beta: [0.053 0.947] | Aposteriori: [0.053 0.   ] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [0.053 0.947] | Aposteriori: [0.    0.947] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0. 1.] Guess: 1 | Alpha: [0. 1.] | Beta: [1. 1.] | Aposteriori: [0. 1.] Guess: 1

+----------------------------+
|    Baum-Welch Algorithm    |
+----------------------------+
= EPOCH #0 =
Transition Matrix: [[0.458 0.542]
 [0.358 0.642]]
Initial Dice Probability: [0.915 0.085]
First Dice: [0.107 0.161 0.075 0.2   0.206 0.25 ]
Second Dice: [0.117 0.086 0.151 0.108 0.126 0.412]
Fitness: 149.65528335692835

= EPOCH #1 =
Transition Matrix: [[0.436 0.564]
 [0.312 0.688]]
Initial Dice Probability: [0.902 0.098]
First Dice: [0.109 0.166 0.075 0.206 0.211 0.232]
Second Dice: [0.115 0.084 0.148 0.106 0.125 0.421]
Fitness: 150.00350537621227

= EPOCH #2 =
Transition Matrix: [[0.416 0.584]
 [0.261 0.739]]
Initial Dice Probability: [0.884 0.116]
First Dice: [0.111 0.175 0.075 0.217 0.22  0.202]
Second Dice: [0.113 0.083 0.143 0.106 0.124 0.43 ]
Fitness: 150.10160491473974

= EPOCH #3 =
Transition Matrix: [[0.407 0.593]
 [0.207 0.793]]
Initial Dice Probability: [0.858 0.142]
First Dice: [0.114 0.187 0.075 0.231 0.231 0.162]
Second Dice: [0.11  0.083 0.137 0.105 0.124 0.441]
Fitness: 150.24399765843805

= EPOCH #4 =
Transition Matrix: [[0.42  0.58 ]
 [0.153 0.847]]
Initial Dice Probability: [0.813 0.187]
First Dice: [0.118 0.203 0.075 0.249 0.245 0.11 ]
Second Dice: [0.106 0.081 0.131 0.103 0.122 0.456]
Fitness: 150.57732456508938

= EPOCH #5 =
Transition Matrix: [[0.482 0.518]
 [0.106 0.894]]
Initial Dice Probability: [0.734 0.266]
First Dice: [0.123 0.219 0.076 0.268 0.258 0.056]
Second Dice: [0.101 0.077 0.123 0.099 0.118 0.481]
Fitness: 151.37558606998124

= EPOCH #6 =
Transition Matrix: [[0.619 0.381]
 [0.074 0.926]]
Initial Dice Probability: [0.628 0.372]
First Dice: [0.129 0.23  0.08  0.281 0.264 0.016]
Second Dice: [0.091 0.067 0.111 0.088 0.107 0.535]
Fitness: 153.03322104762069

= EPOCH #7 =
Transition Matrix: [[0.803 0.197]
 [0.054 0.946]]
Initial Dice Probability: [0.584 0.416]
First Dice: [0.143 0.224 0.1   0.273 0.258 0.002]
Second Dice: [0.066 0.041 0.083 0.056 0.073 0.681]
Fitness: 154.94912107504737

= EPOCH #8 =
Transition Matrix: [[0.875 0.125]
 [0.031 0.969]]
Initial Dice Probability: [0.626 0.374]
First Dice: [1.580e-01 2.039e-01 1.515e-01 2.420e-01 2.445e-01 9.440e-05]
Second Dice: [0.024 0.007 0.03  0.013 0.02  0.906]
Fitness: 156.3032585934971

= EPOCH #9 =
Transition Matrix: [[0.819 0.181]
 [0.024 0.976]]
Initial Dice Probability: [0.663 0.337]
First Dice: [1.615e-01 1.958e-01 1.726e-01 2.318e-01 2.383e-01 4.165e-06]
Second Dice: [1.061e-02 6.695e-04 4.451e-03 1.100e-03 3.754e-03 9.794e-01]
Fitness: 162.6774836810894

= EPOCH #10 =
Transition Matrix: [[0.779 0.221]
 [0.022 0.978]]
Initial Dice Probability: [0.672 0.328]
First Dice: [1.611e-01 1.918e-01 1.746e-01 2.299e-01 2.427e-01 1.628e-07]
Second Dice: [8.743e-03 1.040e-04 6.925e-04 1.090e-04 5.144e-04 9.898e-01]
Fitness: 170.1360986590276

= EPOCH #11 =
Transition Matrix: [[0.763 0.237]
 [0.021 0.979]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.598e-01 1.897e-01 1.774e-01 2.280e-01 2.451e-01 5.816e-09]
Second Dice: [8.505e-03 1.977e-05 1.651e-04 1.306e-05 9.309e-05 9.912e-01]
Fitness: 172.64831160352287

= EPOCH #12 =
Transition Matrix: [[0.757 0.243]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.592e-01 1.893e-01 1.784e-01 2.275e-01 2.455e-01 2.003e-10]
Second Dice: [8.594e-03 4.016e-06 4.417e-05 1.677e-06 1.834e-05 9.913e-01]
Fitness: 173.22436706752228

= EPOCH #13 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.590e-01 1.892e-01 1.787e-01 2.274e-01 2.456e-01 6.809e-12]
Second Dice: [8.686e-03 8.331e-07 1.223e-05 2.205e-07 3.706e-06 9.913e-01]
Fitness: 173.39107739598666

= EPOCH #14 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.589e-01 1.892e-01 1.788e-01 2.274e-01 2.457e-01 2.304e-13]
Second Dice: [8.738e-03 1.739e-07 3.423e-06 2.919e-08 7.543e-07 9.913e-01]
Fitness: 173.440734538517

= EPOCH #15 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.788e-01 2.274e-01 2.457e-01 7.788e-15]
Second Dice: [8.762e-03 3.637e-08 9.609e-07 3.874e-09 1.538e-07 9.912e-01]
Fitness: 173.45428771431943

= EPOCH #16 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 2.631e-16]
Second Dice: [8.772e-03 7.612e-09 2.700e-07 5.143e-10 3.139e-08 9.912e-01]
Fitness: 173.45751021788777

= EPOCH #17 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 8.889e-18]
Second Dice: [8.776e-03 1.593e-09 7.588e-08 6.830e-11 6.406e-09 9.912e-01]
Fitness: 173.45810669271992

= EPOCH #18 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 3.003e-19]
Second Dice: [8.777e-03 3.334e-10 2.133e-08 9.071e-12 1.307e-09 9.912e-01]
Fitness: 173.45814655015752

= EPOCH #19 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 1.014e-20]
Second Dice: [8.777e-03 6.979e-11 5.995e-09 1.205e-12 2.668e-10 9.912e-01]
Fitness: 173.4581106982644

= EPOCH #20 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 3.427e-22]
Second Dice: [8.778e-03 1.461e-11 1.685e-09 1.600e-13 5.444e-11 9.912e-01]
Fitness: 173.4580844754659

= EPOCH #21 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 1.158e-23]
Second Dice: [8.778e-03 3.058e-12 4.736e-10 2.125e-14 1.111e-11 9.912e-01]
Fitness: 173.45807183917998

= EPOCH #22 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 3.911e-25]
Second Dice: [8.778e-03 6.400e-13 1.331e-10 2.822e-15 2.268e-12 9.912e-01]
Fitness: 173.45806663951927

= EPOCH #23 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 1.321e-26]
Second Dice: [8.778e-03 1.340e-13 3.742e-11 3.747e-16 4.627e-13 9.912e-01]
Fitness: 173.45806468101952

= EPOCH #24 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 4.463e-28]
Second Dice: [8.778e-03 2.804e-14 1.052e-11 4.976e-17 9.444e-14 9.912e-01]
Fitness: 173.45806398594468

= EPOCH #25 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 1.508e-29]
Second Dice: [8.778e-03 5.868e-15 2.957e-12 6.609e-18 1.927e-14 9.912e-01]
Fitness: 173.4580637500818

= EPOCH #26 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 5.093e-31]
Second Dice: [8.778e-03 1.228e-15 8.310e-13 8.777e-19 3.933e-15 9.912e-01]
Fitness: 173.4580636729291

= EPOCH #27 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 1.720e-32]
Second Dice: [8.778e-03 2.571e-16 2.336e-13 1.166e-19 8.026e-16 9.912e-01]
Fitness: 173.45806364848696

= EPOCH #28 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 5.812e-34]
Second Dice: [8.778e-03 5.381e-17 6.566e-14 1.548e-20 1.638e-16 9.912e-01]
Fitness: 173.45806364096916

= EPOCH #29 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 1.963e-35]
Second Dice: [8.778e-03 1.126e-17 1.845e-14 2.056e-21 3.343e-17 9.912e-01]
Fitness: 173.45806363872288

= EPOCH #30 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 6.633e-37]
Second Dice: [8.778e-03 2.357e-18 5.187e-15 2.730e-22 6.822e-18 9.912e-01]
Fitness: 173.45806363807006

= EPOCH #31 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 2.241e-38]
Second Dice: [8.778e-03 4.934e-19 1.458e-15 3.626e-23 1.392e-18 9.912e-01]
Fitness: 173.45806363788628

= EPOCH #32 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 7.569e-40]
Second Dice: [8.778e-03 1.033e-19 4.098e-16 4.815e-24 2.841e-19 9.912e-01]
Fitness: 173.45806363783697

= EPOCH #33 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 2.557e-41]
Second Dice: [8.778e-03 2.162e-20 1.152e-16 6.395e-25 5.798e-20 9.912e-01]
Fitness: 173.45806363782415

= EPOCH #34 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 8.638e-43]
Second Dice: [8.778e-03 4.525e-21 3.238e-17 8.492e-26 1.183e-20 9.912e-01]
Fitness: 173.45806363782108

= EPOCH #35 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 2.918e-44]
Second Dice: [8.778e-03 9.471e-22 9.101e-18 1.128e-26 2.415e-21 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #36 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 9.858e-46]
Second Dice: [8.778e-03 1.982e-22 2.558e-18 1.498e-27 4.928e-22 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #37 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 3.330e-47]
Second Dice: [8.778e-03 4.149e-23 7.191e-19 1.989e-28 1.006e-22 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #38 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 1.125e-48]
Second Dice: [8.778e-03 8.684e-24 2.021e-19 2.642e-29 2.052e-23 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #39 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 3.800e-50]
Second Dice: [8.778e-03 1.818e-24 5.681e-20 3.508e-30 4.189e-24 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #40 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 1.284e-51]
Second Dice: [8.778e-03 3.805e-25 1.597e-20 4.659e-31 8.548e-25 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #41 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 4.337e-53]
Second Dice: [8.778e-03 7.963e-26 4.488e-21 6.187e-32 1.744e-25 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #42 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 1.465e-54]
Second Dice: [8.778e-03 1.667e-26 1.262e-21 8.217e-33 3.560e-26 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #43 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 4.949e-56]
Second Dice: [8.778e-03 3.489e-27 3.546e-22 1.091e-33 7.265e-27 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #44 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 1.672e-57]
Second Dice: [8.778e-03 7.302e-28 9.968e-23 1.449e-34 1.483e-27 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #45 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 5.648e-59]
Second Dice: [8.778e-03 1.528e-28 2.802e-23 1.925e-35 3.026e-28 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #46 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 1.908e-60]
Second Dice: [8.778e-03 3.199e-29 7.875e-24 2.556e-36 6.175e-29 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #47 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 6.446e-62]
Second Dice: [8.778e-03 6.696e-30 2.214e-24 3.394e-37 1.260e-29 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #48 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 2.177e-63]
Second Dice: [8.778e-03 1.402e-30 6.222e-25 4.508e-38 2.572e-30 9.912e-01]
Fitness: 173.45806363781983

= EPOCH #49 =
Transition Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
Initial Dice Probability: [0.673 0.327]
First Dice: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 7.356e-65]
Second Dice: [8.778e-03 2.933e-31 1.749e-25 5.987e-39 5.248e-31 9.912e-01]
Fitness: 173.45806363781983

+--------------------------+
|    Baum-Welch Summary    |
+--------------------------+
Dice #1: [1.588e-01 1.892e-01 1.789e-01 2.274e-01 2.457e-01 7.356e-65]
Dice #2: [8.778e-03 2.933e-31 1.749e-25 5.987e-39 5.248e-31 9.912e-01]
Initial Dice Probabilities: [0.673 0.327]
Transision Matrix: [[0.755 0.245]
 [0.02  0.98 ]]
```

## Resources

Great explanation of Viterbi Algorithm (starting about 22:40):  
https://www.youtube.com/watch?v=kqSzLo9fenk

Simple implementation for weather example from above video:  
https://github.com/luisguiserrano/hmm/blob/master/Simple%20HMM.ipynb

Great slides with visual interpretation of all equations:  
http://www.cs.tut.fi/~sgn24006/PDF/L08-HMMs.pdf

Good explanation of Forward & Backward algorithms:  
https://www.youtube.com/watch?v=9yl4XGp5OEg

Tutorial on Hidden Markov Models:  
http://www.stat.columbia.edu/~liam/teaching/neurostat-fall17/papers/hmm/rabiner.pdf
