from math import exp
from typing import Callable, Tuple

import numpy as np

from ..enums import ElevationConstants

ElevationFn = Callable[[float, float], float]


def gaussian(amplitude: int, mu_x: int, mu_y: int, sigma_x: int,
             sigma_y: int) -> ElevationFn:
    '''
    Create a callable that returns the value of a Gaussian centered at (mu_x, mu_y) with
    variances given by sigma_x and sigma_y. The input A will modify the final amplitude.

    Arguments:
        amplitude: The Gaussian amplitude
        mu_x: The mean/center in the x direction
        mu_y: The mean/center in the y direction
        sigma_x: The variance in the x direction
        sigma_y: The variance in the y direction

    Returns:
        A callabe that computes z values for (x, y) inputs
    '''
    def fn(x: float, y: float) -> float:
        '''
        Return the gaussian function value at the specified point.

        Arguments:
            x: The input x coordinate
            y: The input y coordinate

        Returns:
            z: The output z coordinate computed by the function
        '''

        exp_term = ((x - mu_x)**2 / (4 * sigma_x**2)) + ((y - mu_y)**2 / (4 * sigma_y**2))
        z = amplitude * exp(-exp_term)
        return z

    return fn


class PerlinNoise2D():
    def __init__(self,
                 amplitude: float,
                 shape: Tuple[int, int],
                 res: Tuple[int, int],
                 seed: int = None) -> None:
        '''
        Create a class to compute perlin noise for given input parameters.

        Arguments:
            amplitude: The amplitude to scale the noise by.
            shape: The output shape of the data.
            res: The resolution of the noise.
            seed: The initialization seed for randomization.
        '''
        self.amplitude = amplitude
        self.shape = shape
        self.res = res
        self.seed = seed
        self.terrain_map = self._precompute()

    def _precompute(self) -> np.ndarray:
        '''
        Precompute the noise at each (x, y) location for faster use later.
        '''
        def f(t):
            return 6 * t**5 - 15 * t**4 + 10 * t**3

        delta = (self.res[0] / self.shape[0], self.res[1] / self.shape[1])
        d = (self.shape[0] // self.res[0], self.shape[1] // self.res[1])
        # Ignore mypy here becuase it thinks the floats in delta are indexing an array,
        # but this is the intended functionality of np.mgrid()
        grid = np.mgrid[0:self.res[0]:delta[0],  # type: ignore
                        0:self.res[1]:delta[1]].transpose(1, 2, 0) % 1  # type: ignore
        # Gradients
        if isinstance(self.seed, int):
            np.random.seed(self.seed)
        angles = 2 * np.pi * np.random.rand(self.res[0] + 1, self.res[1] + 1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        terrain_map = self.amplitude * np.sqrt(2) * (
            (1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
        terrain_map = terrain_map + np.min(terrain_map)
        # Add the mean elevation of CA to the map
        terrain_map = terrain_map + ElevationConstants.MEAN_ELEVATION
        # Apply bounds to the map to prevent elevation values from being greater or less
        # than what is possible for the state
        terrain_map = self._apply_elevation_bounds(terrain_map)

        return terrain_map

    def _apply_elevation_bounds(self, elevation_map: np.ndarray) -> np.ndarray:
        '''
        Apply elevation bounds to the elevation map.

        Arguments:
            elevation_map: The elevation map to apply the bounds to

        Returns:
            The elevation map with bounds applied
        '''
        min_elevation = ElevationConstants.MIN_ELEVATION
        max_elevation = ElevationConstants.MAX_ELEVATION
        elevation_map[elevation_map < min_elevation] = min_elevation
        elevation_map[elevation_map > max_elevation] = max_elevation
        return elevation_map

    def fn(self, x: int, y: int) -> float:
        '''
        Wrapper function to retrieve the perlin noise values at input (x, y) coordinates.

        Arguments:
            x: The x coordinate to retrieve
            y: The y coordinate to retrieve

        Returns:
            The perlin noise value at the (x, y) coordinates
        '''
        if not isinstance(x, int):
            x = int(x)
        if not isinstance(y, int):
            y = int(y)
        return self.terrain_map[x, y]


def flat() -> ElevationFn:
    '''
    Create a callable that returns 0 for all elevations.

    Returns:
        A callable that computes z values for (x, y) inputs
    '''
    def fn(x: float, y: float) -> float:
        '''
        Return a constant, flat elevation value at every x and y point

        Arguments:
            x: The input x location (isn't used).
            y: The input y location (isn't used).

        Returns:
            The constant, flat elevation.
        '''
        return 0

    return fn
