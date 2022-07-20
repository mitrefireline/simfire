# Configuring the Simulation

The configuration for SimFire is all done through a [YAML](https://yaml.org/) file. All of the configurable settings will be explained in the sections below.

This file can have any name, as a path is required to load the configuration file, but the hierarchy of the file must match the hierarchy found in the example [`config.yml`](https://gitlab.mitre.org/fireline/simfire/-/blob/main/config.yml). This example configuration can be seen in its entirety at the [bottom of this page](#example-config-file).

## Settings

---

### Area Parameters

#### screen_size
(`int`)<br>
Determines how large the simulation is in pixels. The `screen_size` sets both the
height and the width of the screen.

#### terrain_size
(`int`)<br>
Number of terrain tiles in each row/column.

#### pixel_scale
(`float`)<br>
The number of feet across that one pixel represents. i.e. for the default value, one
pixel represents a 50ft x 50ft square of land.

---

### Display Parameters

#### fire_size
(`int`)<br>
The size of the flame/fire size in pixels. Only used for display purposes, and does
not change fire dynamics.

#### control_line_size
(`int`)<br>
The size of the control lines in pixels. Only used for display purposes, and does not
change how much space the control line takes up in the simulation.

---

### Simulation Parameters

#### update_rate
(`float`)<br>
The number of minutes that each game/frame update represents in simulation time. Note
that this can be fractional to account for times that are non-integer and/or less than 1.

#### runtime
(`str`)<br>
The amount of time that the simulation is supposed to run. This can be expressed in days, hours, and minutes in any of the following ways and more: `2d`, `2day`, `2days`, `24h`, `1d 23h 60m`, etc. If no designation is given or if the value is given with a space in between the integer and the time span (e.g. `2 days`), the program will assume the units are in minutes.

#### headless
(`bool`)<br>
Whether or not to run the simulation in a headless state.

#### save_path
(`str`)<br>
Where to save GIF recordings and burn graphs of simulation [runs](https://fireline.pages.mitre.org/simfire/autoapi/simfire/sim/simulation/index.html#simfire.sim.simulation.FireSimulation.run). It will create subfolders in `save_path` that are of the format `<month>-<day>-<year>_<hour>-<minute>-<second>` based on when the outputs were saved to file.

---

### Mitigation Parameters

#### ros_attenuation
(`bool`)<br>
Whether or not to attenuate rate of spread based on the type of line being used. These attenuation values can be seen in [`enums.py`](https://gitlab.mitre.org/fireline/simfire/-/blob/main/simfire/enums.py#L48). These values will be subtracted from the rate of spread for a given pixel based on the control line type. If this value is set to `false`, all lines will completely stop a fire and the rate of spread for any pixel with a control line will be set to zero.

---

### Operational Parameters
These are the operational parameters that will be used for the operational data layers that rely on a location, if those layers' types are set to `operational`.
#### seed
(`int`)<br>
The seed that would pick a random latitude and longitude. Leave empty if using the [latitude](#latitude) and [longitude](#longitude) below.

#### latitude
(`float`)<br>
The latitude given in decimal degrees.

#### longitude
(`float`)<br>
The longitude given in decimal degrees.

#### height
(`int`)<br>
The height of the screen in meters.

#### width
(`int`)<br>
The width of the screen in meters.

#### resolution
(`int`)<br>
The resolution of each pixel in meters. Data is measured in pixels corresponding to the resolution i.e: resolution = 10m = 1 pixel.

---

### Terrain Parameters

#### topography
All configuration settings for the topography in the simulation area.

##### type
(`str`)<br>
Can be either `operational` or `functional`.

If `operational`, will use the parameters in the [Operational Parameters](#operational-parameters) section to determine topography in the simulation area.

If `functional`, will use the parameters in [terrain.topography.functional](#functional) to determine topography in the simulation area.

##### functional
All configuration settings for determining the functional topography data layer.

###### function
(`str`)<br>
The function that determines how elevation is distributed throughout the simulation area.
The available elevation functions are currently:

  - perlin
  - gaussian
  - flat

###### perlin
All arguments that would be passed into the PerlinNoise2D elevation class.

- **amplitude** (`int`):<br>
  The amplitude of the perlin noise function.

- **shape** (`Tuple[int, int]`):<br>
  The output shape of the noise. Most often, should probably be `(area.screen_size, area.screen_size)`.

- **resolution** (`Tuple[int, int]`):<br>
  The output resolution of the noise. Could be thought of as the "altitude" and aspect ratio at which the data `(1, 1)` is seen.

- **seed** (`int`):<br>
  The random seed used to determine the terrain elevation so that the user can recreate repeatable terrain.

###### gaussian
All arguments that would be passed into the gaussian function.

- **amplitude** (`int`):<br>
  The amplitude of the gaussian noise function.

- **mu_x** (`int`):<br>
  The mean of the 2D normal distribution in the `x` direction.

- **mu_y** (`int`):<br>
  The mean of the 2D normal distribution in the `y` direction.

- **sigma_x** (`int`):<br>
  The variance of the 2D normal distribution in the `x` direction.

- **sigma_x** (`int`):<br>
  The variance of the 2D normal distribution in the `y` direction.

#### fuel
All configuration settings for the fuel in the simulation area.

##### type
(`str`)<br>
Can be either `operational` or `functional`.

If `operational`, will use the parameters in the [Operational Parameters](#operational-parameters) section to determine fuel in the simulation area.

If `functional`, will use the parameters in [terrain.fuel.functional](#functional-1) to determine fuel in the simulation area.

##### functional
All configuration settings for determining the functional topography data layer.

###### function
(`str`)<br>
The function that determines how fuel is distributed throughout the simulation area.
The available fuel functions are currently:

  - chaparral

###### chaparral
All arguments that would be passed into the chaparral fuel array function.

- **seed** (`int`):<br>
  The random seed used to determine the fuel distribution so that the user can recreate repeatable fuel patterns.

---

### Fire Parameters

#### fire_initial_position
(`Tuple[int, int]`)<br>
The initial location to start the fire. This should be set every time when running the
simulation.

#### max_fire_duration
(`int`)<br>
The maximum number of frames that a single pixel can be on fire.

---

### Environment Parameters
Defines the Environment class.

#### moisture
(`float`)<br>
Used in simulator fire spread calculation. Most of Southern California has the default value of 0.03.

---

### Wind Parameters
Defines wind speed and direction generation.

#### function
(`str`)<br>
The function that determines how wind is distributed throughout the simulation area.
The available wind functions are currently:

  - cfd
  - simple
  - perlin

#### cfd
A function for wind that uses a computational fluid dynamics (CFD) algorithm for wind modeling.

- **time_to_train** (`int`): <br>
  @ckempis: The time to train the CFD algorithm when preprocessing.

- **iterations** (`int`): <br>
  @ckempis

- **scale** (`int`): <br>
  @ckempis

- **timestep_dt** (`int`): <br>
  @ckempis

- **diffusion** (`int`): <br>
  @ckempis

- **viscosity** (`int`): <br>
  @ckempis

- **speed** (`int`): <br>
  @ckempis

- **direction** (`int`): <br>
  @ckempis

#### simple
A function for wind that keeps direction and speed constant throughout the whole simulation area.

- **speed** (`int` | `float`):<br>
  The wind speed in **miles per hour**.

- **direction** (`int` | `float`):<br>
  The wind direction expressed in **degrees clockwise from North** (E.g. East == 90.0, South == 180.0, etc.).

#### perlin
All arguments that would be passed into the WindController class.

##### speed
All arguments that define wind speed layer generation.

- **seed** (`int`):<br>
  The random seed used to determine the wind speed layer so that the user can recreate repeatable wind speed layers.

- **scale** (`int`):<br>
  How large to make the noise. Can be seen as an "elevation", but don't take that literally.

- **octaves** (`int`):<br>
  How many passes/layers to the noise algorithm. Each pass adds more detail.

- **persistence** (`float`):<br>
  How much more each successive value brings. Keep between 0 and 1.0.

- **lacunarity** (`float`):<br>
  The level of detail added per pass. Usually kept at 2.0.

- **min** (`int`| `float`):<br>
  The *minimum* wind speed for any individual pixel location. Expressed in **miles per hour**.

- **max** (`int` | `float`):<br>
  The *maximum* wind speed for any individual pixel location. Expressed in **miles per hour**.

##### direction
All arguments that define wind direction layer generation.

- **seed** (`int`):<br>
  The random seed used to determine the wind speed layer so that the user can recreate repeatable wind speed layers.

- **scale** (`int`):<br>
  How large to make the noise. Can be seen as an "elevation", but don't take that literally.

- **octaves** (`int`):<br>
  How many passes/layers to the noise algorithm. Each pass adds more detail.

- **persistence** (`float`):<br>
  How much more each successive value brings. Keep between 0 and 1.0.

- **lacunarity** (`float`):<br>
  The level of detail added per pass. Usually kept at 2.0.

- **min** (`int`):<br>
  The *minimum* wind direction for any individual pixel location. Expressed in **degrees clockwise from North** (E.g. East == 90.0, South == 180.0, etc.).

- **max** (`int`):<br>
  The *maximum* wind direction for any individual pixel location. Expressed in **degrees clockwise from North** (E.g. East == 90.0, South == 180.0, etc.).

---

### Render Parameters
Defines rendering.

#### inline
(`bool`)<br>
Whether or not to render at each call to the `step()` function in `FirelineEnv`.

#### post_agent
(`bool`)<br>
Whether or not to render with post agent in place.

#### post_agent_with_fire
(`bool`)<br>
Whether or not to render with post agent and fire in place.

---

## Example Config File

Go [here](https://gitlab.mitre.org/fireline/simfire/-/blob/main/configs) for more examples.

```yaml
area:
  screen_size: 225
  pixel_scale: 50

display:
  fire_size: 2
  control_line_size: 2

simulation:
  update_rate: 1
  runtime: 24h
  headless: false
  record: true

mitigation:
  ros_attenuation: true

operational:
  seed:
  latitude: 39.67
  longitude: 119.8
  height: 4000
  width: 4000
  resolution: 30

terrain:
  topography:
    type: functional
    functional:
      function: perlin
      perlin:
        octaves: 3
        persistence: 0.7
        lacunarity: 2.0
        seed: 827
        range_min: 100.0
        range_max: 300.0
      gaussian:
        amplitude: 500
        mu_x: 50
        mu_y: 50
        sigma_x: 50
        sigma_y: 50
  fuel:
    type: functional
    functional:
      function: chaparral
      chaparral:
        seed: 1113

fire:
  fire_initial_position: (16, 16)
  max_fire_duration: 4

environment:
  moisture: 0.03

wind:
  function: perlin
  cfd:
    time_to_train: 1000
    iterations: 1
    scale: 1
    timestep_dt: 1.0
    diffusion: 0.0
    viscosity: 0.0000001
    speed: 19
    direction: north
  simple:
    speed: 7
    direction: 90.0
  perlin:
    speed:
      seed: 2345
      scale: 400
      octaves: 3
      persistence: 0.7
      lacunarity: 2.0
      range_min: 7
      range_max: 47
    direction:
      seed: 650
      scale: 1500
      octaves: 2
      persistence: 0.9
      lacunarity: 1.0
      range_min: 0.0
      range_max: 360.0
```
