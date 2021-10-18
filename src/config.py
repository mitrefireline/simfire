from typing import Tuple

from .world.parameters import FuelArray
from .world.presets import Chaparral, ShortGrass

# Game/Screen parameters
# Screen size in pixels
screen_size: int = 1000
# Number of terrain tiles in each row/column
terrain_size: int = 15
# Fire/flame szie in pixels
fire_size: int = 2
# Fireline size in pixels
control_line_size: int = 10
# The amount of feet 1 pixel represents (ft)
pixel_scale = 500
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
fire_init_pos: Tuple[int, int] = (112, 112)
# Fires burn for 7 frames
max_fire_duration: int = 10

# Environment Parameters:
# Moisture Content
M_f: float = 0.03
# Wind Speed (ft/min)
# ft/min = 88*mi/hour
U: float = 88 * 5
# Wind Direction (degrees clockwise from north)
U_dir: float = 45
