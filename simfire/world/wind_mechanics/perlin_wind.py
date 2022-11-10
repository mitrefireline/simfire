from typing import Optional

import numpy as np
from noise import snoise2


class WindNoise:
    """
    Class for controlling and fine tuning wind noise generation with the Simplex noise
    algorithm
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        scale: int = 100,
        octaves: int = 2,
        persistence: float = 0.5,
        lacunarity: float = 1.0,
    ) -> None:
        """
        Class that handles and creates the wind layer which specifies the magnitude
        and direction of wind a a given location.  Uses python noise library

        Arguments:
            seed: The value to seed the noise generator
            scale: The "altitude" from which to see the noise
            octaves: number of passes/layers of the algorithm.  Each pass adds more detail
            persistence: How much each pass affects the overall shape
                         High values means each pass is less important on shape.
                         Lower values mean each pass has greater effect on shape.
                         Best to keep between 0-1
            lacunarity: Controls increase in frequency of octaves per pass.
                        Frequency = lacunarity & (pass number).
                        Higher lacunarity, higher frequency per pass.

            screen_size: Size of screen (both heigh and width) MUST BE SQUARE
        """
        if seed is None:
            self.seed = np.random.randint(0, 100)
        else:
            self.seed = seed

        self.scale: int = scale
        self.octaves: int = octaves
        self.persistence: float = persistence
        self.lacunarity: float = lacunarity
        self.range_min: float
        self.range_max: float

    def set_noise_parameters(
        self,
        seed: int,
        scale: int,
        octaves: int,
        persistence: float,
        lacunarity: float,
        range_min: float,
        range_max: float,
    ):
        self.seed = seed
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.range_min = range_min
        self.range_max = range_max

    def generate_map_array(self, screen_size) -> np.ndarray:
        map = []
        map = [
            [self._generate_noise_value(x, y) for x in range(screen_size)]
            for y in range(screen_size)
        ]
        return np.array(map, dtype=np.float32)

    def _denormalize_noise_value(self, noise_value) -> float:
        denormalized_value = (
            ((noise_value + 1) * (self.range_max - self.range_min)) / 2
        ) + self.range_min
        return denormalized_value

    def _generate_noise_value(self, x: int, y: int) -> float:
        scaledX = x / self.scale
        scaledY = y / self.scale

        value = snoise2(
            scaledX,
            scaledY,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
            base=self.seed,
        )

        denormalized_value = self._denormalize_noise_value(value)

        return denormalized_value
