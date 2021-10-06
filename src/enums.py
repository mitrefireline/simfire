from enum import auto, IntEnum


class BurnStatus(IntEnum):
    UNBURNED = auto()
    BURNING = auto()
    BURNED = auto()