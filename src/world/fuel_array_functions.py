from typing import Callable

from .parameters import Fuel
from ..utils.terrain import chaparral

FuelFn = Callable[[float, float], Fuel]


def chaparral_fn(scale_x: float, scale_y: float, seed: int = None) -> FuelFn:
    '''
    Return a callable that accepts (x, y) coordinates and returns a Fuel with
    Chaparral characterisitics at that coordinate. The w_0 parameter is slightly
    altered/jittered to allow for non-uniform terrains. Specifying a specific seed
    allows for recreateable random terrain.

    Arguments:
        scale_x: The width of the Fuel tile in feet
        scale_y: The height of the Fuel tile in feet
        seed: The seed to initialize the Fuel w_0 randomization

    Returns:
        A FuelFn callable that accepts (x,y) coordinates and returns a Fuel
    '''
    def fn(x: float, y: float) -> Fuel:
        '''
        Use the input coordinates to generate a Fuel for the environment at that
        coordinate.

        Arguments:
            x: The input x coordinate
            y: The input y coorindate

        Returns:
            A Fuel
        '''
        fuel = chaparral(seed)

        return fuel

    return fn
