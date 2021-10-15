from typing import Tuple

import GPUtil
import numpy as np

from .world.elevation_functions import PerlinNoise2D
from .world.parameters import Fuel, FuelArray

# Use GPU if available, else CPU
try:
    if len(GPUtil.getAvailable()) > 0:
        device = 'cuda'
    else:
        device = 'cpu'
except ValueError:
    device = 'cpu'

# Game/Screen parameters
# Screen size in pixels
screen_size: int = 225
# Number of terrain tiles in each row/column
terrain_size: int = 15
# Fire/flame szie in pixels
fire_size: int = 2
# The amount of feet 1 pixel represents (ft)
pixel_scale: float = 50
# Copmute the size of each terrain tile in feet
terrain_scale = terrain_size * pixel_scale

# Create function that returns elevation values at (x, y) points
A = 500
mu_x = 50
mu_y = 50
sigma_x = 50
sigma_y = 50
pnoise = PerlinNoise2D(A, [screen_size, screen_size], [1, 1])
pnoise.precompute()

elevation_fn = pnoise.fn


def Chaparral_r():
    return Fuel(w_0=np.random.uniform(.3, .5), delta=6.000, M_x=0.2000, sigma=1739)


# Create FuelArray tiles to make the terrain/map
chaparral_row = tuple((Chaparral_r() for i in range(terrain_size)))
short_grass_row = tuple((Chaparral_r() for i in range(terrain_size)))
terrain_map: Tuple[Tuple[FuelArray]] = ((chaparral_row, ) * (terrain_size // 2) +
                                        (short_grass_row, ) * (terrain_size // 2) +
                                        (short_grass_row, ) * (terrain_size % 2))

# Fire Manager Parameters
# (x, y) starting coordinates
fire_init_pos: Tuple[int, int] = (110, 110)
# Fires burn for a limited number of frames
max_fire_duration: int = 5

# Environment Parameters:
# Moisture Content
M_f: float = 0.03
# Wind Speed (ft/min)
# ft/min = 88*mi/hour
U: float = 88 * 13
# Wind Direction (degrees clockwise from north)
U_dir: float = 135
