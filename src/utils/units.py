import re
from typing import Union
from datetime import timedelta

from .log import create_logger

log = create_logger(__name__)

UNITS = {'s': 'seconds', 'm': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}


def mph_to_ftpm(mph: Union[int, float]) -> Union[int, float]:
    '''
    Convert mph to ft/min

    Arguments:
        mph: The speed in mph.

    Returns:
        The speed in per minute.
    '''
    ftpm = mph * 88
    return ftpm


def ftpm_to_mph(ftpm: Union[int, float]) -> Union[int, float]:
    '''
    Convert ft/min to mph

    Arguments:
        ftpm: The speed in ft/min.

    Returns:
        The speed in mph.
    '''
    mph = ftpm / 88
    return mph


def convert_to_minutes(string: str) -> int:
    '''
    Convert any string representation of time ('1d 2h 3m') into a number of minutes

    Arguments:
        string: The input string represented in any of the following ways and more: `2d`,
        `2 d`, `2 days`, `24h`, `1d 23h 60m`.

    Returns:
        The number of minutes represented by the string.
    '''
    return int(
        round(
            timedelta(
                **{
                    UNITS.get(m.group('unit').lower(), 'minutes'): float(m.group('val'))
                    for m in re.finditer(
                        r'(?P<val>\d+(\.\d+)?)(?P<unit>[smhdw]?)', string, flags=re.I)
                }).total_seconds() / 60))
