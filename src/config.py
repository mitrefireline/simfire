from typing import Tuple
import numpy as np

# Game/Screen parameters
screen_size: int = 225
terrain_size: int = 15
fire_size: int = 2

# Fuel Array Parameters
w_0_max: float = 10
delta_max: float = 5
M_x_max: float = 0.2
w_0: np.ndarray = np.random.choice([1, w_0_max], p=[0.75, 0.25], size=(terrain_size, terrain_size))
delta: np.ndarray = np.random.choice([1, delta_max], p=[0.75, 0.25], size=(terrain_size, terrain_size))
M_x: np.ndarray = np.random.choice([0.01, M_x_max], p=[0.75, 0.25], size=(terrain_size, terrain_size))

# Fire Manager Parameters
# (x, y) starting coordinates
fire_init_pos: Tuple[int, int] = (10, 10)
# Fires burn for 7 frames
max_fire_duration: int = 7
# Spreads at 1 pixel every 5 frames
rate_of_spread: int = 5
