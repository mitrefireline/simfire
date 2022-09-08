import re
from datetime import timedelta
from typing import Tuple, Union

import numpy as np

from .log import create_logger

log = create_logger(__name__)

UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def scale_ms_to_ftpm(ms: np.ndarray) -> np.ndarray:
    ftpm = ms * 196.85
    return ftpm


def mph_to_ms(mph: Union[int, float]) -> Union[int, float]:
    """
    Convert mph to m/s

    Arguments:
        mph: The speed in mph.

    Returns:
        The speed in meters per second
    """
    ftpm = mph / 2.237

    return ftpm


def mph_to_ftpm(mph: Union[int, float]) -> Union[int, float]:
    """
    Convert mph to ft/min

    Arguments:
        mph: The speed in mph.

    Returns:
        The speed in feet per minute.
    """
    ftpm = mph * 88
    return ftpm


def ftpm_to_mph(ftpm: Union[int, float]) -> Union[int, float]:
    """
    Convert ft/min to mph

    Arguments:
        ftpm: The speed in ft/min.

    Returns:
        The speed in mph.
    """
    mph = ftpm / 88
    return mph


def str_to_minutes(string: str) -> int:
    """
    Convert any string representation of time ('1d 2h 3m') into a number of minutes

    Arguments:
        string: The input string represented in any of the following ways and more: `2d`,
        `2days`, `24h`, `1d 23h 60m`.

    Returns:
        The number of minutes represented by the string.
    """
    return int(
        round(
            timedelta(
                **{
                    UNITS.get(m.group("unit").lower(), "minutes"): float(m.group("val"))
                    for m in re.finditer(
                        r"(?P<val>\d+(\.\d+)?)(?P<unit>[smhdw]?)", string, flags=re.I
                    )
                }
            ).total_seconds()
            / 60
        )
    )


def chains_to_feet_handline(chains: float) -> Tuple[int, int]:
    """
    Convert "chains" to (width x hieght) / hour per individual firefighters.

    Calculated by averaging from a 20-person hand-crew.
    https://www.nifc.gov/nicc/logistics/references/Wildland%20Fire%20Incident%20Management%20Field%20Guide.pdf
    pgs: 110-113

    Chains are defined as 66 ft x 3 ft
    """
    return int(chains * 66), 3


def chains_to_feet_dozerline(chains: float) -> Tuple[int, int]:
    """
    Convert "chains" to (width x hieght) / hour per dozer.

    https://www.nifc.gov/nicc/logistics/references/Wildland%20Fire%20Incident%20Management%20Field%20Guide.pdf
    pgs: 114-116

    Chains are defined as 66 ft x 30 ft
    """
    return int(chains * 66), 30
