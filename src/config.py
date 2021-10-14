from typing import Tuple

import GPUtil

from .world.elevation_functions import gaussian
from .world.parameters import FuelArray
from .world.presets import Chaparral

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
pixel_scale: float = 500
# Copmute the size of each terrain tile in feet
terrain_scale = terrain_size * pixel_scale

# Create function that returns elevation values at (x, y) points
A = 100
mu_x = 50
mu_y = 50
sigma_x = 50
sigma_y = 50
elevation_fn = gaussian(A, mu_x, mu_y, sigma_x, sigma_y)
# elevation_fn = flat()

# Create FuelArray tiles to make the terrain/map
chaparral_row = (Chaparral, ) * terrain_size
short_grass_row = (Chaparral, ) * terrain_size
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
