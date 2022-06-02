from math import exp
from typing import Callable

import noise

ElevationFn = Callable[[int, int], float]


def flat() -> ElevationFn:
    """
    Create a callable that returns 0 for all elevations.

    Returns:
        A callable that computes z values for (x, y) inputs
    """

    def fn(x: int, y: int) -> float:
        """
        Return a constant, flat elevation value at every x and y point

        Arguments:
            x: The input x location (isn't used).
            y: The input y location (isn't used).

        Returns:
            The constant, flat elevation of 0.
        """
        return 0

    return fn


def gaussian(
    amplitude: float, mu_x: float, mu_y: float, sigma_x: float, sigma_y: float
) -> ElevationFn:
    """
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
    """

    def fn(x: int, y: int) -> float:
        """
        Return the gaussian function value at the specified point.

        Arguments:
            x: the input x coordinate
            y: the input y coordinate

        Returns:
            The output z coordinate computed by the function
        """

        exp_term = ((x - mu_x) ** 2 / (4 * sigma_x**2)) + (
            (y - mu_y) ** 2 / (4 * sigma_y**2)
        )
        z = amplitude * exp(-exp_term)
        return z

    return fn


def perlin(
    octaves: int,
    persistence: float,
    lacunarity: float,
    seed: int,
    range_min: float,
    range_max: float,
) -> ElevationFn:
    """
    Create a callable that returns the value of a 2D Perlin noise function.

    Arguments:
        octaves: specifies the number of passes, defaults to 1 (simple noise).
        persistence: specifies the amplitude of each successive octave relative
                     to the one below it. Defaults to 0.5 (each higher octave's amplitude
                     is halved). Note the amplitude of the first pass is always 1.0.
        lacunarity: specifies the frequency of each successive octave relative
                    to the one below it, similar to persistence. Defaults to 2.0.
        seed: The seed to used to generate random terrain. `seed` takes the place of the
              `base` argument in the `snoise2()` function, which adds offsets to the
              input (x, y) coordinates to get new terrain
        range_min: The minimum amplitude to scale to
        range_max: The maximum amplitude to scale to

    Returns:
        A callable that computes Perlin Noise z-values for (x, y) inputs
    """
    if range_min >= range_max:
        raise ValueError(f"range_min={range_min} must be less than range_max={range_max}")

    def fn(x: int, y: int) -> float:
        """
        Return the generated Perlin Noise function at the specified value.

        Arguments:
            x: the input x coordinate
            y: the input y coordinate

        Returns:
            The output z coordinate computed by the function
        """
        z = noise.snoise2(x, y, octaves, persistence, lacunarity, base=seed)
        # Normalize to [0, 1]
        z = (z + 1) / 2
        # Scale to [0, range_max-range_min]
        z = z * (range_max - range_min)
        # Add to normalize to [range_min, range_max]
        z = z + range_min
        return z

    return fn
