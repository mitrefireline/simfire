from typing import Tuple

import GPUtil

from .world.parameters import FuelArray
from .world.presets import Chaparral, ShortGrass

# Use GPU if available, else CPU
try:
    if len(GPUtil.getAvailable()) > 0:
        device = 'cuda'
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
pixel_scale = 150
# Copmute the size of each terrain tile in feet
terrain_scale = terrain_size * pixel_scale

# Create FuelArray tiles to make the terrain/map
chaparral_row = (Chaparral, ) * terrain_size
short_grass_row = (ShortGrass, ) * terrain_size
terrain_map: Tuple[Tuple[FuelArray]] = ((chaparral_row, ) * (terrain_size // 2) +
                                        (short_grass_row, ) * (terrain_size // 2) +
                                        (short_grass_row, ) * (terrain_size % 2))

# Fire Manager Parameters
# (x, y) starting coordinates
fire_init_pos: Tuple[int, int] = (103, 50)
# Fires burn for a limited number of frames
max_fire_duration: int = 5

# Environment Parameters:
# Moisture Content
M_f: float = 0.03
# Wind Speed (ft/min)
# ft/min = 88*mi/hour
U: float = 88 * 1
# Wind Direction (degrees clockwise from north)
U_dir: float = 45
