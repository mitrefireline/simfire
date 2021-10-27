from typing import Tuple

import GPUtil
import numpy as np

from .world.elevation_functions import PerlinNoise2D, flat
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
# Fireline size in pixels
control_line_size: int = 2
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

# elevation_fn = pnoise.fn
elevation_fn = flat()


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
fire_init_pos: Tuple[int, int] = (65, 65)
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


# Wind Noise Parameters
mw_seed: int = 2345
mw_scale: int = 2
mw_octaves: int = 3
mw_persistence: float = 0.7
mw_lacunarity: float = 2.0
mw_speed_min: float = 616.0 # ft/min
mw_speed_max: float = 4136.0 # ft/min

dw_seed: int = 1203
dw_scale: int = 2
dw_octaves: int = 2
dw_persistence: float = 0.9
dw_lacunarity: float = 1.0
dw_deg_min: float = 0.0 # Degrees, 0/360: North, 90: East, 180: South, 270: West
dw_deg_max: float  = 360.0 # Degrees, 0/360: North, 90: East, 180: South, 270: West