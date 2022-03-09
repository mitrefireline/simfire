import numpy as np
import time
from ..wind_mechanics.perlin_wind import WindNoise
from ..wind_mechanics.cfd_wind import Fluid

import pygame
pygame.init


# For Perlin Implementations
class WindController():
    '''
    Generates and tracks objects that dictate wind magnitude and wind direction for map
    given size of the screen
    '''
    def __init__(self, screen_size: int = 225) -> None:
        self.speed_layer = WindNoise()
        self.direction_layer = WindNoise()
        self.map_wind_speed = []
        self.map_wind_direction = []
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
            range_min: The maximum speed of the wind in ft/min.
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
    Generates and tracks objects that dictate wind magnitude and wind direction for map
    given size of the screen
    '''
    def __init__(self, screen_size: int = 225, result_accuracy: int = 1, scale: int = 1,
                 timestep: float = 1.0, diffusion: float = 0.0,
                 viscosity: float = 0.0000001,
                 terrain_features: np.ndarray = None) -> None:
        self.N: int = screen_size
        self.iterations: int = result_accuracy
        self.scale = scale
        self.timestep = timestep
        self.diffusion = diffusion
        self.viscosity = viscosity

        if terrain_features is None:
            self.terrain_features = np.zeros((self.N, self.N))
        else:
            self.terrain_features = np.array(terrain_features)
        # TODO Load terrain setup here

        self.fvect = Fluid(self.N, self.iterations, self.scale, self.timestep, 
                           self.diffusion, self.viscosity, self.terrain_features)

    def initialize_wind_fields(self, source_direction, source_speed,
                               screen_size: int = 225):
        time_end = time.time() + 100
        screen = pygame.display.set_mode([225, 225])
        screen.fill((255, 255, 255))
        while time.time() < time_end:
            # contiually spawn velocity
            for v in range(0, screen_size):
                if source_direction == 'north':
                    self.fvect.addVelocity(v, 1, 0, source_speed)
                elif source_direction == 'east':
                    self.fvect.addVelocity(screen_size - 1, v, -1 * source_speed, 0)
                elif source_direction == 'south':
                    self.fvect.addVelocity(1, v, -1 * source_speed, 0)
                elif source_direction == 'west':
                    self.fvect.addVelocity(1, v, source_speed, 0)
                else:
                    print('Bad source direction input')
                    return

            self.fvect.step()
            self.fvect.renderD(screen)
            pygame.display.flip()