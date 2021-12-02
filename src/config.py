'''
    Config
    ======

    This file determines all initial configuration for the fire simulation.

    The generic usage will be like so for each of the variables listed below:

    Example
    -------
    ```python
    import rothsim.config as cfg

    cfg.screen_size = 225
    cfg.fire_size = 2
    ...
    ```

    Variables
    ---------

    ### Area Parameters
    Used to determine scale of area in square feet.

    #### `screen_size`
    (`int`, default = 225):

    Determines how large the simulation is in pixels. The `screen_size` sets both the
    height and the width of the screen.


    #### `terrain_size`
    (`int`, default = 15)

    Number of terrain tiles in each row/column.


    #### `pixel_scale`
    (`float`, default = 50)

    The number of feet across that one pixel represents. i.e. for the default value, one
    pixel represents a 50ft x 50ft square of land.


    ### Display Parameters
    Used in determining display scaling.

    #### `fire_size`
    (`int`, default = 2)

    The size of the flame/fire size in pixels. Only used for display purposes, and does
    not change fire dynamics.


    #### `control_line_size`
    (`int`, default = 2)

    The size of the control lines in pixels. Only used for display purposes, and does not
    change how much space the control line takes up in the simulation.


    ### Terrain Parameters
    Used when defining the Terrain class.

    #### `elevation_fn`
    (`Callable`, default = rothsim.world.elevation_functions.PerlinNoise2D)

    The function that determines how elevation is determined throughout the simulation.
    The available elevation functions can be found in `rothsim.world.elevation_functions`
    and the default function is
    `PerlinNoise2D(amplitude=500, shape=(screen_size, screen_size), res=(1, 1))`.

    If this would like to be changed, just the `noise_amplitude` can be modified or look
    to the examples below:

    ```python
    import rothsim.config as cfg
    from rothsim.world.elevation_functions import flat

    cfg.elevation_fn = flat()
    ```

    OR

    ```python
    import rothsim.config as cfg
    from rothsim.world.elevation_functions import gaussian

    noise_amplitude = 500
    mu_x = 50
    mu_y = 50
    sigma_x = 50
    sigma_y = 50

    cfg.elevation_fn = gaussian(noise_amplitude, mu_x, mu_y, sigma_x, sigma_y)
    ```

    ### Simulation Parameters
    Used when determing simulation initial conditions not defined in another class.

    #### `fire_init_pos`
    (`Tuple[int, int]`, default = (65, 65))

    The initial location to start the fire. This should be set every time when running the
    simulation.


    ### Environment Parameters
    Used when defining the Environment class.

    #### `M_f` (Moisture Content)
    (`float`, default = 0.03)

    Used in Rothermel calculation. Most of Southern California has the default value of
    0.03.


    #### `U` (Wind Speed)
    (`float`, default = 88 * 13)

    In ft/min. Used in the Rothermal calculation. To convert from mph to ft/min, multiply
    mph by 88 (e.g.: 88 * 13 mph = 1144 ft/min).


    #### `U_dir` (Wind Direction)
    (`float`, default = 135)

    Degrees clockwise from North (upwards in the simulation window).
'''
from typing import Tuple

from .world.elevation_functions import PerlinNoise2D
from .world.parameters import FuelArray
from .utils.terrain import Chaparral

# Game/Screen parameters

# Number of terrain tiles in each row/column
terrain_size: int = 15
# Screen size in pixels
screen_size: int = terrain_size**2

# Fire/flame szie in pixels
fire_size: int = 2
# Fireline size in pixels
control_line_size: int = 2
# The amount of feet 1 pixel represents (ft)
pixel_scale: float = 50
# Compute the size of each terrain tile in feet
terrain_scale: int = terrain_size * pixel_scale

# The amount of time that passes in the simulation for each frame update (minutes)
update_rate: float = 1

# Create function that returns elevation values at (x, y) points
noise_amplitude: int = 500
pnoise = PerlinNoise2D(noise_amplitude, [screen_size, screen_size], [1, 1])
pnoise.precompute()

elevation_fn = pnoise.fn

# Create FuelArray tiles to make the terrain/map
terrain_map: Tuple[Tuple[FuelArray]] = tuple(
    tuple(Chaparral() for _ in range(terrain_size)) for _ in range(terrain_size))

# Fire Manager Parameters
# (x, y) starting coordinates
fire_init_pos: Tuple[int, int] = (screen_size // 2, screen_size // 2)
# Fires burn for a limited number of frames
max_fire_duration: int = 4

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
mw_scale: int = 400  # High values = Lower Resolution, less noisy
mw_octaves: int = 3
mw_persistence: float = 0.7
mw_lacunarity: float = 2.0
mw_speed_min: float = 616.0  # ft/min
mw_speed_max: float = 4136.0  # ft/min

dw_seed: int = 650  # 1203 (North bias)
dw_scale: int = 1500  # Better to be above 1000 to have less variable wind dir
dw_octaves: int = 2
dw_persistence: float = 0.9
dw_lacunarity: float = 1.0
dw_deg_min: float = 0.0  # Degrees, 0/360: North, 90: East, 180: South, 270: West
dw_deg_max: float = 360.0  # same as above

render_inline: str = False
render_post_agent: str = False
render_post_agent_with_fire: str = True

# random seed for producing same terrain map
seed = 1111  # None
