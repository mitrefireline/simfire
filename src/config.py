from typing import Tuple
import numpy as np

# Game/Screen parameters
# Screen size in pixels
screen_size: int = 225
# Terrain size in pixels
terrain_size: int = 15
# Fire/flame szie in pixels
fire_size: int = 2
# The amount of feet 1 pixel represents (ft)
pixel_scale = 10
# Copmute the size of each terrain tile in feet
terrain_scale = terrain_size * pixel_scale

# Fuel Array Parameters
w_0_max: float = 7
delta_max: float = 7
M_x_max: float = 0.2
sigma_max: float = 2000
w_0: np.ndarray = np.random.choice([0.5, w_0_max], p=[0.75, 0.25], size=(terrain_size, terrain_size))
delta: np.ndarray = np.random.choice([0.5, delta_max], p=[0.75, 0.25], size=(terrain_size, terrain_size))
M_x: np.ndarray = np.random.choice([0.1, M_x_max], p=[0.75, 0.25], size=(terrain_size, terrain_size))
sigma: np.ndarray = np.random.choice([750, sigma_max], p=[0.75, 0.25], size=(terrain_size, terrain_size))

# Fire Manager Parameters
# (x, y) starting coordinates
fire_init_pos: Tuple[int, int] = (122, 122)
# Fires burn for 7 frames
max_fire_duration: int = 10

# Environment Parameters:
# Moisture Content
M_f: float = 0.025
# Wind Speed (ft/min)
# ft/min = 88*mi/hour
U: float = 88 * 20
# Wind Direction (degrees clockwise from north)
U_dir: float = 90
