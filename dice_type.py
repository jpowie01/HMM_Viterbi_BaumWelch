import enum

AVAILABLE_DICES = 2


class DiceType(enum.IntEnum):
    FAIR = 0
    LOADED = 1
