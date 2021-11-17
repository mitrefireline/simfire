import numpy as np
from ..world.parameters import Fuel


def Chaparral(seed: int = None) -> Fuel:
    if seed is not None:
        np.random.seed(seed)
    return Fuel(w_0=np.random.uniform(.2, .6), delta=6.000, M_x=0.2000, sigma=1739)
