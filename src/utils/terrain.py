from typing import Tuple

import numpy as np

from ..world.parameters import Fuel


def random_seed_list(length: int, seed: int = None) -> Tuple[Tuple[int]]:
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


def w_0_seed(seed: int) -> float:
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


def chaparral(seed: int = None) -> Fuel:
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
