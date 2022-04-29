from typing import Tuple, Union

import numpy as np

from ..world.parameters import Fuel


def random_seed_list(length: int, seed: int = None) -> Tuple[Tuple[int, ...], ...]:
    '''
    Create a tuple of tuples  of random integers (to be used as seeds) based on `length`
    and initial input `seed`

    Arguments:
        length: Length of each of the nested tuples.
        seed: Initial random seed for generating seeds.

    Returns:
        A tuple of tuples containing random integers ranging from 0 to 99_999
    '''
    np.random.seed(seed)
    return tuple(
        tuple(np.random.randint(0, 99_999) for _ in range(length)) for _ in range(length))


def w_0_seed(seed: Union[int, None]) -> float:
    '''
    Create a `w_0` between `0.2` and `0.6` based on an initial `seed` parameter

    Arguments:
        seed: Initial seed.

    Returns:
        The oven-dry fuel load value of the fuel.
    '''
    np.random.seed(seed)
    # Update the test for this function if this range is changed in the future
    w_0 = np.random.uniform(.2, .6)
    return w_0


def chaparral(seed: Union[int, None] = None) -> Fuel:
    '''
    Create a chaparral fuel object using an optional input seed

    The seed only affects the input to `w_0`, the oven-dry fuel load value.

    Arguments:
        seed: Initial seed.

    Returns:
        A fuel with randomized `w_0`, `delta == 6.0`, `M_x == 0.2`, and `sigma == 1739`.
    '''
    w_0 = w_0_seed(seed)
    return Fuel(w_0=w_0, delta=6.000, M_x=0.2000, sigma=1739)


def fuel(seed: int = None) -> Tuple[float, float]:
    '''
    Functionailty to use a random seed to define a center point.
    To be used with operational data layers

    Predefine CA latitude / longitude bounds (N, W, S, E)

    '''

    north = 41.81527476
    south = 32.85980972
    east = 113.8035177
    west = 125.0133402

    np.random.seed(seed)
    longitude = np.random.uniform(east, west)
    latitude = np.random.uniform(south, north)

    return (latitude, longitude)
