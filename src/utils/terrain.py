import numpy as np
from typing import Tuple
from ..world.parameters import Fuel


def RandomSeedList(length: int, seed: int = None) -> Tuple[Tuple[int]]:
    np.random.seed(seed)
    return tuple(
        tuple(np.random.randint(0, 99999) for _ in range(length)) for _ in range(length))


def w_0_seed(seed):
    np.random.seed(seed)
    w_0 = np.random.uniform(.2, .6)
    return w_0


def Chaparral(seed: int = None) -> Fuel:
    w_0 = w_0_seed(seed)
    return Fuel(w_0=w_0, delta=6.000, M_x=0.2000, sigma=1739)
