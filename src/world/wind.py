from typing import Callable, Tuple
from noise import pnoise2 ,snoise2
from math import exp
import numpy as np

class WindNoise():
    '''
    Class for controlling and fine tuning wind noise generation with the Simplex noise
    algorithm
    '''
    def __init__(self, seed: int = None, scale: int = 100, octaves: int = 2, persistence: float = 0.5, lacunarity: float = 1.0) -> None:
        '''
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
        '''
        if(seed == None):
            self.seed = np.random.randint(0,100)
        else:
            self.seed = seed

        self.scale: int = scale
        self.octaves: int = octaves
        self.persistence: float = persistence
        self.lacunarity: float = lacunarity

    def set_noise_parameters(self, seed: int, scale: int, octaves: int, persistence: float, lacunarity: float):
        self.seed: int = seed
        self.scale: int = scale
        self.octaves: int = octaves
        self.persistence: float = persistence
        self.lacunarity: float = lacunarity

    def generate_noise_value(self, x: int, y: int) -> float:
        scaledX = x / self.scale
        scaledY = y / self.scale

        value = snoise2(scaledX, 
                        scaledY, 
                        octaves=self.octaves,
                        persistence=self.persistence,
                        lacunarity=self.lacunarity,
                        base=self.seed)

        return value

class WindController():
    '''
    Generates and tracks objects that dictate wind magnitude and wind direction for map given size of the screen
    '''
    def __init__() -> None:
        self.map_wind_speeds = WindNoise()
        self.map_wind_directions = WindNoise()

    def set_wind_speed_generator(self, seed: int, scale: int, octaves: int, persistence: float, lacunarity: float) -> None:
        self.map_wind_speeds.set_noise_parameters(seed, scale, octaves, persistence, lacunarity)
    
    def set_wind_direction_generator(self, seed: int, scale: int, octaves: int, persistence: float, lacunarity: float) -> None:
        self.map_wind_directions.set_noise_parameters(seed, scale, octaves, persistence, lacunarity)
        