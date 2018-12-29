Hidden Markov Models - Viterbi & Baum-Welch Algorithms
------------------------------------------------------

This repository presents example implementation for Viterbi and Baum-Welch
algorithms implementation in Python 3.6+ using Numpy.

## How to run this example?

```bash
$ virtualenv -p python3.6 venv
$ . ./venv/bin/activate
(venv) $ python main.py
```

#### Example output

For such an environment:

```python
fair_dice = dice.Dice(np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]))
loaded_dice = dice.Dice(np.array([1/10, 1/10, 1/10, 1/10, 1/10, 1/2]))
initial_dice_probability = np.array([0.5, 0.5])  # Croupier can use either of dices initially
transition_matrix = np.array([
    [0.95, 0.05],  # Fair -> Fair = 0.95, Fair -> Loaded = 0.05
    [0.10, 0.90],  # Loaded -> Fair = 0.10, Loaded -> Loaded = 0.90
])
my_croupier = croupier.Croupier(fair_dice, loaded_dice, initial_dice_probability, transition_matrix)
```

We get such analysis:

```
Observation: 0 Dice: 1 | Viterbi: [0.083 0.05 ] Guess: 0 | Alpha: [0.083 0.05 ] | Beta: [0.301 0.699] | Aposteriori: [0.025 0.035] Guess: 1
Observation: 2 Dice: 1 | Viterbi: [0.746 0.254] Guess: 0 | Alpha: [0.74 0.26] | Beta: [0.183 0.817] | Aposteriori: [0.135 0.212] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.508 0.492] Guess: 0 | Alpha: [0.473 0.527] | Beta: [0.329 0.671] | Aposteriori: [0.156 0.354] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.266 0.734] Guess: 1 | Alpha: [0.252 0.748] | Beta: [0.565 0.435] | Aposteriori: [0.142 0.325] Guess: 1
Observation: 4 Dice: 1 | Viterbi: [0.39 0.61] Guess: 1 | Alpha: [0.433 0.567] | Beta: [0.451 0.549] | Aposteriori: [0.195 0.312] Guess: 1
Observation: 5 Dice: 0 | Viterbi: [0.183 0.817] Guess: 1 | Alpha: [0.227 0.773] | Beta: [0.704 0.296] | Aposteriori: [0.16  0.229] Guess: 1
Observation: 1 Dice: 0 | Viterbi: [0.283 0.717] Guess: 1 | Alpha: [0.408 0.592] | Beta: [0.638 0.362] | Aposteriori: [0.26  0.214] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [0.41 0.59] Guess: 1 | Alpha: [0.574 0.426] | Beta: [0.544 0.456] | Aposteriori: [0.312 0.194] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [0.55 0.45] Guess: 0 | Alpha: [0.704 0.296] | Beta: [0.425 0.575] | Aposteriori: [0.299 0.17 ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [0.683 0.317] Guess: 0 | Alpha: [0.794 0.206] | Beta: [0.296 0.704] | Aposteriori: [0.235 0.145] Guess: 0
Observation: 5 Dice: 0 | Viterbi: [0.431 0.569] Guess: 1 | Alpha: [0.534 0.466] | Beta: [0.521 0.479] | Aposteriori: [0.278 0.223] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [0.571 0.429] Guess: 0 | Alpha: [0.675 0.325] | Beta: [0.398 0.602] | Aposteriori: [0.269 0.196] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [0.701 0.299] Guess: 0 | Alpha: [0.775 0.225] | Beta: [0.27 0.73] | Aposteriori: [0.209 0.165] Guess: 0
Observation: 5 Dice: 0 | Viterbi: [0.452 0.548] Guess: 1 | Alpha: [0.511 0.489] | Beta: [0.482 0.518] | Aposteriori: [0.246 0.253] Guess: 1
Observation: 5 Dice: 0 | Viterbi: [0.225 0.775] Guess: 1 | Alpha: [0.277 0.723] | Beta: [0.734 0.266] | Aposteriori: [0.203 0.192] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [0.338 0.662] Guess: 1 | Alpha: [0.457 0.543] | Beta: [0.684 0.316] | Aposteriori: [0.312 0.172] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [0.473 0.527] Guess: 1 | Alpha: [0.614 0.386] | Beta: [0.608 0.392] | Aposteriori: [0.373 0.151] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [0.612 0.388] Guess: 0 | Alpha: [0.733 0.267] | Beta: [0.504 0.496] | Aposteriori: [0.369 0.133] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [0.735 0.265] Guess: 0 | Alpha: [0.813 0.187] | Beta: [0.38 0.62] | Aposteriori: [0.309 0.116] Guess: 0
Observation: 5 Dice: 0 | Viterbi: [0.494 0.506] Guess: 1 | Alpha: [0.558 0.442] | Beta: [0.628 0.372] | Aposteriori: [0.35  0.164] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [0.632 0.368] Guess: 0 | Alpha: [0.692 0.308] | Beta: [0.531 0.469] | Aposteriori: [0.367 0.145] Guess: 0
Observation: 5 Dice: 0 | Viterbi: [0.377 0.623] Guess: 1 | Alpha: [0.424 0.576] | Beta: [0.776 0.224] | Aposteriori: [0.329 0.129] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [0.516 0.484] Guess: 0 | Alpha: [0.587 0.413] | Beta: [0.753 0.247] | Aposteriori: [0.442 0.102] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [0.652 0.348] Guess: 0 | Alpha: [0.713 0.287] | Beta: [0.715 0.285] | Aposteriori: [0.51  0.082] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [0.767 0.233] Guess: 0 | Alpha: [0.8 0.2] | Beta: [0.655 0.345] | Aposteriori: [0.524 0.069] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [0.853 0.147] Guess: 0 | Alpha: [0.856 0.144] | Beta: [0.567 0.433] | Aposteriori: [0.485 0.063] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0.671 0.329] Guess: 0 | Alpha: [0.615 0.385] | Beta: [0.805 0.195] | Aposteriori: [0.495 0.075] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [0.782 0.218] Guess: 0 | Alpha: [0.733 0.267] | Beta: [0.804 0.196] | Aposteriori: [0.589 0.052] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [0.863 0.137] Guess: 0 | Alpha: [0.813 0.187] | Beta: [0.801 0.199] | Aposteriori: [0.652 0.037] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [0.917 0.083] Guess: 0 | Alpha: [0.863 0.137] | Beta: [0.797 0.203] | Aposteriori: [0.688 0.028] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [0.951 0.049] Guess: 0 | Alpha: [0.893 0.107] | Beta: [0.79 0.21] | Aposteriori: [0.705 0.022] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [0.969 0.031] Guess: 0 | Alpha: [0.911 0.089] | Beta: [0.777 0.223] | Aposteriori: [0.707 0.02 ] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [0.969 0.031] Guess: 0 | Alpha: [0.92 0.08] | Beta: [0.755 0.245] | Aposteriori: [0.694 0.02 ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [0.969 0.031] Guess: 0 | Alpha: [0.926 0.074] | Beta: [0.717 0.283] | Aposteriori: [0.664 0.021] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [0.969 0.031] Guess: 0 | Alpha: [0.929 0.071] | Beta: [0.658 0.342] | Aposteriori: [0.611 0.024] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [0.969 0.031] Guess: 0 | Alpha: [0.931 0.069] | Beta: [0.571 0.429] | Aposteriori: [0.532 0.03 ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [0.969 0.031] Guess: 0 | Alpha: [0.932 0.068] | Beta: [0.458 0.542] | Aposteriori: [0.427 0.037] Guess: 0
Observation: 5 Dice: 0 | Viterbi: [0.864 0.136] Guess: 0 | Alpha: [0.733 0.267] | Beta: [0.711 0.289] | Aposteriori: [0.522 0.077] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [0.918 0.082] Guess: 0 | Alpha: [0.813 0.187] | Beta: [0.649 0.351] | Aposteriori: [0.528 0.066] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [0.951 0.049] Guess: 0 | Alpha: [0.863 0.137] | Beta: [0.558 0.442] | Aposteriori: [0.482 0.06 ] Guess: 0
Observation: 0 Dice: 0 | Viterbi: [0.969 0.031] Guess: 0 | Alpha: [0.893 0.107] | Beta: [0.442 0.558] | Aposteriori: [0.395 0.06 ] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [0.969 0.031] Guess: 0 | Alpha: [0.911 0.089] | Beta: [0.313 0.687] | Aposteriori: [0.285 0.061] Guess: 0
Observation: 5 Dice: 0 | Viterbi: [0.864 0.136] Guess: 0 | Alpha: [0.698 0.302] | Beta: [0.545 0.455] | Aposteriori: [0.38  0.137] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [0.918 0.082] Guess: 0 | Alpha: [0.79 0.21] | Beta: [0.426 0.574] | Aposteriori: [0.337 0.12 ] Guess: 0
Observation: 1 Dice: 0 | Viterbi: [0.951 0.049] Guess: 0 | Alpha: [0.849 0.151] | Beta: [0.297 0.703] | Aposteriori: [0.253 0.106] Guess: 0
Observation: 5 Dice: 0 | Viterbi: [0.864 0.136] Guess: 0 | Alpha: [0.606 0.394] | Beta: [0.522 0.478] | Aposteriori: [0.317 0.188] Guess: 0
Observation: 2 Dice: 1 | Viterbi: [0.918 0.082] Guess: 0 | Alpha: [0.727 0.273] | Beta: [0.4 0.6] | Aposteriori: [0.291 0.164] Guess: 0
Observation: 0 Dice: 1 | Viterbi: [0.951 0.049] Guess: 0 | Alpha: [0.809 0.191] | Beta: [0.272 0.728] | Aposteriori: [0.22  0.139] Guess: 0
Observation: 1 Dice: 1 | Viterbi: [0.969 0.031] Guess: 0 | Alpha: [0.861 0.139] | Beta: [0.158 0.842] | Aposteriori: [0.136 0.117] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0.864 0.136] Guess: 0 | Alpha: [0.622 0.378] | Beta: [0.278 0.722] | Aposteriori: [0.173 0.273] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.69 0.31] Guess: 0 | Alpha: [0.361 0.639] | Beta: [0.494 0.506] | Aposteriori: [0.178 0.323] Guess: 1
Observation: 0 Dice: 1 | Viterbi: [0.797 0.203] Guess: 0 | Alpha: [0.534 0.466] | Beta: [0.368 0.632] | Aposteriori: [0.196 0.295] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.58 0.42] Guess: 0 | Alpha: [0.292 0.708] | Beta: [0.615 0.385] | Aposteriori: [0.18  0.273] Guess: 1
Observation: 2 Dice: 0 | Viterbi: [0.708 0.292] Guess: 0 | Alpha: [0.471 0.529] | Beta: [0.513 0.487] | Aposteriori: [0.242 0.258] Guess: 1
Observation: 3 Dice: 0 | Viterbi: [0.81 0.19] Guess: 0 | Alpha: [0.626 0.374] | Beta: [0.389 0.611] | Aposteriori: [0.244 0.229] Guess: 0
Observation: 4 Dice: 0 | Viterbi: [0.882 0.118] Guess: 0 | Alpha: [0.741 0.259] | Beta: [0.262 0.738] | Aposteriori: [0.194 0.191] Guess: 0
Observation: 5 Dice: 0 | Viterbi: [0.725 0.275] Guess: 0 | Alpha: [0.474 0.526] | Beta: [0.469 0.531] | Aposteriori: [0.222 0.28 ] Guess: 1
Observation: 2 Dice: 0 | Viterbi: [0.823 0.177] Guess: 0 | Alpha: [0.627 0.373] | Beta: [0.341 0.659] | Aposteriori: [0.214 0.245] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.621 0.379] Guess: 0 | Alpha: [0.365 0.635] | Beta: [0.582 0.418] | Aposteriori: [0.213 0.265] Guess: 1
Observation: 1 Dice: 1 | Viterbi: [0.742 0.258] Guess: 0 | Alpha: [0.537 0.463] | Beta: [0.471 0.529] | Aposteriori: [0.253 0.245] Guess: 0
Observation: 5 Dice: 0 | Viterbi: [0.503 0.497] Guess: 0 | Alpha: [0.295 0.705] | Beta: [0.723 0.277] | Aposteriori: [0.213 0.195] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [0.64 0.36] Guess: 0 | Alpha: [0.474 0.526] | Beta: [0.667 0.333] | Aposteriori: [0.316 0.175] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [0.758 0.242] Guess: 0 | Alpha: [0.628 0.372] | Beta: [0.584 0.416] | Aposteriori: [0.367 0.155] Guess: 0
Observation: 2 Dice: 0 | Viterbi: [0.846 0.154] Guess: 0 | Alpha: [0.742 0.258] | Beta: [0.474 0.526] | Aposteriori: [0.352 0.136] Guess: 0
Observation: 3 Dice: 0 | Viterbi: [0.907 0.093] Guess: 0 | Alpha: [0.819 0.181] | Beta: [0.347 0.653] | Aposteriori: [0.284 0.118] Guess: 0
Observation: 0 Dice: 1 | Viterbi: [0.945 0.055] Guess: 0 | Alpha: [0.867 0.133] | Beta: [0.223 0.777] | Aposteriori: [0.193 0.103] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0.857 0.143] Guess: 0 | Alpha: [0.631 0.369] | Beta: [0.403 0.597] | Aposteriori: [0.254 0.22 ] Guess: 0
Observation: 3 Dice: 1 | Viterbi: [0.914 0.086] Guess: 0 | Alpha: [0.745 0.255] | Beta: [0.275 0.725] | Aposteriori: [0.205 0.185] Guess: 0
Observation: 5 Dice: 1 | Viterbi: [0.788 0.212] Guess: 0 | Alpha: [0.478 0.522] | Beta: [0.49 0.51] | Aposteriori: [0.234 0.267] Guess: 1
Observation: 1 Dice: 1 | Viterbi: [0.867 0.133] Guess: 0 | Alpha: [0.631 0.369] | Beta: [0.364 0.636] | Aposteriori: [0.229 0.235] Guess: 1
Observation: 4 Dice: 1 | Viterbi: [0.92 0.08] Guess: 0 | Alpha: [0.744 0.256] | Beta: [0.238 0.762] | Aposteriori: [0.177 0.195] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.802 0.198] Guess: 0 | Alpha: [0.478 0.522] | Beta: [0.43 0.57] | Aposteriori: [0.205 0.298] Guess: 1
Observation: 0 Dice: 1 | Viterbi: [0.877 0.123] Guess: 0 | Alpha: [0.631 0.369] | Beta: [0.301 0.699] | Aposteriori: [0.19  0.258] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.715 0.285] Guess: 0 | Alpha: [0.368 0.632] | Beta: [0.528 0.472] | Aposteriori: [0.194 0.298] Guess: 1
Observation: 2 Dice: 1 | Viterbi: [0.815 0.185] Guess: 0 | Alpha: [0.54 0.46] | Beta: [0.406 0.594] | Aposteriori: [0.219 0.273] Guess: 1
Observation: 2 Dice: 1 | Viterbi: [0.886 0.114] Guess: 0 | Alpha: [0.678 0.322] | Beta: [0.278 0.722] | Aposteriori: [0.189 0.232] Guess: 1
Observation: 3 Dice: 1 | Viterbi: [0.932 0.068] Guess: 0 | Alpha: [0.777 0.223] | Beta: [0.163 0.837] | Aposteriori: [0.127 0.186] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.828 0.172] Guess: 0 | Alpha: [0.514 0.486] | Beta: [0.289 0.711] | Aposteriori: [0.148 0.346] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.628 0.372] Guess: 0 | Alpha: [0.279 0.721] | Beta: [0.51 0.49] | Aposteriori: [0.142 0.354] Guess: 1
Observation: 3 Dice: 1 | Viterbi: [0.748 0.252] Guess: 0 | Alpha: [0.459 0.541] | Beta: [0.386 0.614] | Aposteriori: [0.177 0.333] Guess: 1
Observation: 1 Dice: 1 | Viterbi: [0.839 0.161] Guess: 0 | Alpha: [0.616 0.384] | Beta: [0.258 0.742] | Aposteriori: [0.159 0.285] Guess: 1
Observation: 2 Dice: 1 | Viterbi: [0.902 0.098] Guess: 0 | Alpha: [0.734 0.266] | Beta: [0.147 0.853] | Aposteriori: [0.108 0.227] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.764 0.236] Guess: 0 | Alpha: [0.466 0.534] | Beta: [0.253 0.747] | Aposteriori: [0.118 0.399] Guess: 1
Observation: 3 Dice: 1 | Viterbi: [0.851 0.149] Guess: 0 | Alpha: [0.621 0.379] | Beta: [0.143 0.857] | Aposteriori: [0.089 0.324] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.667 0.333] Guess: 0 | Alpha: [0.36 0.64] | Beta: [0.244 0.756] | Aposteriori: [0.088 0.484] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.413 0.587] Guess: 1 | Alpha: [0.186 0.814] | Beta: [0.44 0.56] | Aposteriori: [0.082 0.456] Guess: 1
Observation: 0 Dice: 1 | Viterbi: [0.553 0.447] Guess: 0 | Alpha: [0.367 0.633] | Beta: [0.312 0.688] | Aposteriori: [0.114 0.436] Guess: 1
Observation: 5 Dice: 1 | Viterbi: [0.304 0.696] Guess: 1 | Alpha: [0.189 0.811] | Beta: [0.542 0.458] | Aposteriori: [0.103 0.371] Guess: 1
Observation: 3 Dice: 1 | Viterbi: [0.434 0.566] Guess: 1 | Alpha: [0.37 0.63] | Beta: [0.423 0.577] | Aposteriori: [0.157 0.363] Guess: 1
Observation: 0 Dice: 1 | Viterbi: [0.574 0.426] Guess: 0 | Alpha: [0.542 0.458] | Beta: [0.295 0.705] | Aposteriori: [0.16  0.323] Guess: 1
Observation: 5 Dice: 0 | Viterbi: [0.322 0.678] Guess: 1 | Alpha: [0.298 0.702] | Beta: [0.518 0.482] | Aposteriori: [0.155 0.338] Guess: 1
Observation: 2 Dice: 0 | Viterbi: [0.455 0.545] Guess: 1 | Alpha: [0.477 0.523] | Beta: [0.396 0.604] | Aposteriori: [0.189 0.316] Guess: 1
Observation: 2 Dice: 0 | Viterbi: [0.595 0.405] Guess: 0 | Alpha: [0.63 0.37] | Beta: [0.268 0.732] | Aposteriori: [0.169 0.271] Guess: 1
Observation: 0 Dice: 0 | Viterbi: [0.721 0.279] Guess: 0 | Alpha: [0.744 0.256] | Beta: [0.155 0.845] | Aposteriori: [0.115 0.216] Guess: 1
Observation: 5 Dice: 0 | Viterbi: [0.476 0.524] Guess: 1 | Alpha: [0.477 0.523] | Beta: [0.27 0.73] | Aposteriori: [0.129 0.382] Guess: 1
Observation: 2 Dice: 0 | Viterbi: [0.615 0.385] Guess: 0 | Alpha: [0.63 0.37] | Beta: [0.157 0.843] | Aposteriori: [0.099 0.312] Guess: 1
Observation: 5 Dice: 0 | Viterbi: [0.36 0.64] Guess: 1 | Alpha: [0.368 0.632] | Beta: [0.274 0.726] | Aposteriori: [0.101 0.459] Guess: 1
Observation: 3 Dice: 0 | Viterbi: [0.498 0.502] Guess: 1 | Alpha: [0.539 0.461] | Beta: [0.16 0.84] | Aposteriori: [0.086 0.387] Guess: 1
Observation: 5 Dice: 0 | Viterbi: [0.259 0.741] Guess: 1 | Alpha: [0.296 0.704] | Beta: [0.282 0.718] | Aposteriori: [0.084 0.505] Guess: 1
Observation: 5 Dice: 0 | Viterbi: [0.109 0.891] Guess: 1 | Alpha: [0.153 0.847] | Beta: [1. 1.] | Aposteriori: [0.153 0.847] Guess: 1
```

