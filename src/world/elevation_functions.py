from math import exp
from typing import Callable

ElevationFn = Callable[[float, float], float]


def gaussian(A, mu_x, mu_y, sigma_x, sigma_y) -> ElevationFn:
    '''
    Create a callable that returns the value of a Gaussian centered at (mu_x, mu_y) with
    variances given by sigma_x and sigma_y. The input A will modify the final amplitude.

    Arguments:
        A: The Gaussian amplitude
        mu_x: The mean/center in the x direction
        mu_y: The mean/center in the y direction
        sigma_x: The variance in the x direction
        sigma_y: The variance in the y direction

    Returns:
        fn: A callabe that computes z values for (x, y) inputs
    '''
    def fn(x: float, y: float) -> float:
        '''
        Return the function value at the specified point.

        Arguments:
            x: The input x coordinate
            y: The input y coordinate

        Returns:
            z: The output z coordinate computed by the function
        '''

        exp_term = ((x - mu_x)**2 / (4 * sigma_x**2)) + ((y - mu_y)**2 / (4 * sigma_y**2))
        z = A * exp(-exp_term)
        return z

    return fn


def flat() -> ElevationFn:
    '''
    Create a callable that returns 0 for all elevations.

    Arguments:
        None

    Returns:
        fn: A callable that computes z values for (x, y) inputs
    '''
    def fn(x: float, y: float) -> float:
        return 0

    return fn
