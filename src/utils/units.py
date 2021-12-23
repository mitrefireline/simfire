from typing import Union


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
