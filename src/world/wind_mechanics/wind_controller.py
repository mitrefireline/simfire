from typing import Optional
import numpy as np
import pygame

from .perlin_wind import WindNoise
from .cfd_wind import Fluid

pygame.init()


# For Perlin Implementations
class WindController():
    '''
    Generates and tracks objects that dictate wind magnitude and wind direction for map
    given size of the screen
    '''
    def __init__(self, screen_size: int = 225) -> None:
        self.speed_layer = WindNoise()
        self.direction_layer = WindNoise()
        self.map_wind_speed: Optional[np.ndarray] = None
        self.map_wind_direction: Optional[np.ndarray] = None
        self.screen_size = screen_size

    def init_wind_speed_generator(self, seed: int, scale: int, octaves: int,
                                  persistence: float, lacunarity: float, range_min: float,
                                  range_max: float, screen_size: int) -> None:
        '''
        Set simplex noise values for wind speeds

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
            range_min: The minimum speed of the wind in ft/min.
            range_max: The maximum speed of the wind in ft/min.
            screen_size: Size of screen (both heigh and width) MUST BE SQUARE
        '''
        self.speed_layer.set_noise_parameters(seed, scale, octaves, persistence,
                                              lacunarity, range_min, range_max)

        self.map_wind_speed = self.speed_layer.generate_map_array(screen_size)

    def init_wind_direction_generator(self, seed: int, scale: int, octaves: int,
                                      persistence: float, lacunarity: float,
                                      range_min: float, range_max: float,
                                      screen_size: int) -> None:
        '''
        Set simplex noise values for wind directions
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
            range_min: The minimum angle of wind in degrees clockwise.
            range_min: The maximum angle of wind in degrees clockwise.
            screen_size: Size of screen (both heigh and width) MUST BE SQUARE
        '''
        self.direction_layer.set_noise_parameters(seed, scale, octaves, persistence,
                                                  lacunarity, range_min, range_max)

        self.map_wind_direction = self.direction_layer.generate_map_array(screen_size)


# For CFD Implementations
class WindController2():
    '''
    This is a PRECOMPUTE wind controller.  It generates and tracks objects that dictate
    wind magnitude and wind direction for map given size of the screen.
    '''
    def __init__(self,
                 screen_size: int = 225,
                 result_accuracy: int = 1,
                 scale: int = 1,
                 timestep: float = 1.0,
                 diffusion: float = 0.0,
                 viscosity: float = 0.0000001,
                 terrain_features: np.ndarray = None,
                 wind_speed: float = 27,
                 wind_direction: str = 'North') -> None:
        self.N = screen_size
        self.iterations = result_accuracy
        self.scale = scale
        self.timestep = timestep
        self.diffusion = diffusion
        self.viscosity = viscosity
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

        if terrain_features is None:
            self.terrain_features = np.zeros((self.N, self.N))
        else:
            self.terrain_features = np.array(terrain_features, dtype=np.float32)
        # TODO Load terrain setup here

        self.fvect = Fluid(self.N, self.iterations, self.scale, self.timestep,
                           self.diffusion, self.viscosity, self.terrain_features)

    def iterate_wind_step(self) -> None:
        for v in range(0, self.N):
            if self.wind_direction == 'north':
                self.fvect.addVelocity(v, 1, 0, self.wind_speed)
            elif self.wind_direction == 'east':
                self.fvect.addVelocity(self.N - 1, v, -1 * self.wind_speed, 0)
            elif self.wind_direction == 'south':
                self.fvect.addVelocity(1, v, -1 * self.wind_speed, 0)
            elif self.wind_direction == 'west':
                self.fvect.addVelocity(1, v, self.wind_speed, 0)
            else:
                print('Bad source direction input')

        self.fvect.step()
        return

    def get_wind_density_field(self) -> np.ndarray:
        return self.fvect.density

    def get_wind_velocity_field_x(self) -> np.ndarray:
        return self.fvect.Vx

    def get_wind_velocity_field_y(self) -> np.ndarray:
        return self.fvect.Vy

    def get_wind_scale(self) -> int:
        return self.scale

    def get_screen_size(self) -> int:
        return self.N
