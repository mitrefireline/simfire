# Configuring the Simulation

This file determines all initial configuration for the fire simulation.

The generic usage will be like so for each of the variables listed below:

## Example
```python
import rothsim.config as cfg

cfg.screen_size = 225
cfg.fire_size = 2
...
```

## Variables

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


### `update_rate`
(`float`, default = 1)

The number of minutes that each game/frame update represents in simulation time. Note
that this can be fractional to account for times that are non-integer and/or less than 1.


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

If this would like to be changed, just the `noise_amplitude` can be modified or look to
the examples below:

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

Used in Rothermel calculation. Most of Southern California has the default value of 0.03.


#### `U` (Wind Speed)
(`float`, default = 88 * 13)

In ft/min. Used in the Rothermal calculation. To convert from mph to ft/min, multiply
mph by 88 (e.g.: 88 * 13 mph = 1144 ft/min).


#### `U_dir` (Wind Direction)
(`float`, default = 135)

Degrees clockwise from North (upwards in the simulation window).
