from typing import Callable

from .parameters import FuelArray, Tile
from ..utils.terrain import chaparral

FuelArrayFn = Callable[[float, float], FuelArray]


def chaparral_fn(scale_x: float, scale_y: float, seed: int = None) -> FuelArrayFn:
    '''
    Return a callable that accepts (x, y) coordinates and returns a FuelArray with
    Chaparral characterisitics at that coordinate. The w_0 parameter is slightly
    altered/jittered to allow for non-uniform terrains. Specifying a specific seed
    allows for recreateable random terrain.

    Arguments:
        scale_x: The width of the FuelArray tile in feet
        scale_y: The height of the FuelArray tile in feet
        seed: The seed to initialize the Fuel w_0 randomization

    Returns:
        A FuelArrayFn callable that accepts (x,y) coordinates and returns a FuelArray
    '''
    def fn(x: float, y: float) -> FuelArray:
        '''
        Use the input coordinates to generate a FuelArray for the environment at that
        coordinate.

        Arguments:
            x: The input x coordinate
            y: The input y coorindate

        Returns:
            A Fuel
        '''
        tile = Tile(x, y, scale_x, scale_y)
        fuel = chaparral(seed)

        return FuelArray(tile, fuel)

    return fn
