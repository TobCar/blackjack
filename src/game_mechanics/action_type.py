from enum import Enum


class ActionType(Enum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    CASHOUT = "cashout"
    INSURANCE = "insurance"
