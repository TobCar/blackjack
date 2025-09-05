from enum import Enum


class ActionType(Enum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    SURRENDER_ANY_TIME = "surrender_any_time"
    INSURANCE = "insurance"
