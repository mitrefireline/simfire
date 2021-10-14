from math import exp
from typing import Callable
import numpy as np

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

class perlin_noise_2d():
    def __init__(self, A = 300, shape=[225,225],res=[225,225]):
        self.amplitude = A
        self.shape = shape
        self.res = res
        self.terrain_map = None
    
    def precompute(self):
        res = self.res
        shape = self.shape
        def f(t):
            return 6*t**5 - 15*t**4 + 10*t**3
    
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        self.terrain_map = self.amplitude*np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)
        self.terrain_map = self.terrain_map + np.min(self.terrain_map)

    def fn(self, x: float, y: float) -> float:
        return self.terrain_map[x,y]

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
