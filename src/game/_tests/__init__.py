from typing import Tuple
import numpy as np

from ...world.presets import Chaparral


class DummyTopographyLayer():
    '''Dummy layer to use with unittests. Sets random elevations using numpy'''
    def __init__(self, shape: Tuple[int, int]) -> None:
        # Add the 1-dimension to match what the real layers look like
        self.shape = (shape) + (1, )
        self.data = np.random.uniform(0, 300, self.shape)


class DummyFuelLayer():
    '''Dummy layer to use with unittests. Sets all terrain/fuel to Chaparral'''
    def __init__(self, shape: Tuple[int, int]) -> None:
        # Add the 1-dimension to match what the real layers look like
        self.shape = (shape) + (1, )
        self.data = np.full(self.shape, Chaparral)
