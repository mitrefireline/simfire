import numpy as np

import yaml
import os.path
from pathlib import Path
from typing import Any, Tuple
from yaml.parser import ParserError

from ..utils.log import create_logger
from ..utils.layers import (FunctionalFuelLayer, LatLongBox, OperationalTopographyLayer,
                            FunctionalTopographyLayer, OperationalFuelLayer)
from ..world.wind_mechanics.wind_controller import WindController, WindController2
from ..utils.units import str_to_minutes, mph_to_ftpm, mph_to_ms, scale_ms_to_ftpm
from ..world.elevation_functions import PerlinNoise2D, flat, gaussian
from ..world.fuel_array_functions import chaparral_fn
from ..utils.terrain import fuel

log = create_logger(__name__)


class ConfigError(Exception):
    '''
    Exception class for Config class
    '''
    pass


class ConfigType:
    '''
    An object-relational mapper (ORM) type class used in `config.Config` to create nested
    object attributes. For more information on Python ORMs, go
    [here](https://www.fullstackpython.com/object-relational-mappers-orms.html).
    '''
    def __init__(self, **kwargs) -> None:
        '''
        All `kwargs` passed in will be nested into attributes if the `kwargs` consist of
        nested dictionaries.
        '''
        self.__dict__ = kwargs
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = ConfigType(**value)
            setattr(self, key, self._type(value))

    def _type(self, obj: Any) -> Any:
        '''
        Type strings into tuples when setting attributes.
        '''
        if isinstance(obj, str):
            if '(' in obj:
                obj = tuple(map(int, obj[1:-1].split(', ')))
        return obj


class Config:
    '''
    Reads in a YAML file and assigns nested attributes to itself based on structure of
    YAML
    '''
    def __init__(self, path: Path, cfd_precompute: bool = False) -> None:
        '''
        Arguments:
            path: The path to the `config.yml` file. This file can have any name, but
                  needs to be a valid YAML.
        '''
        self.path = path
        self._possible_layers = ('operational', 'functional')
        self._possible_functional_topography = ('perlin', 'gaussian', 'flat')
        self._possible_wind = ('perlin', 'simple')
        self._possible_functional_fuel = ('chaparral')
        self._load()
        self._set_attributes()
        self._set_runtime()
        self._set_terrain_scale()
        self._set_topography_layer()
        self._set_fuel_layer()
        if cfd_precompute is False:
            self._set_wind_function()

    def _load(self) -> None:
        '''
        Loads the YAML file and assigns to `self.data`
        '''
        try:
            with open(self.path, 'r') as f:
                try:
                    self.data = yaml.safe_load(f)
                except ParserError:
                    message = f'Error parsing YAML file at {self.path}'
                    log.error(message)
                    raise ConfigError(message)
        except FileNotFoundError:
            message = f'Error opening YAML file at {self.path}. Does it exist?'
            log.error(message)
            raise ConfigError(message)

    def _set_attributes(self) -> None:
        '''
        Set all nested attributes using the `config.ConfigType` class
        '''
        for key, value in self.data.items():
            if isinstance(value, dict):
                value = ConfigType(**value)
            setattr(self, key, value)

    def _set_terrain_scale(self) -> None:
        '''
        Set the terrain scale and other simulation variables based on type of data layer.
        If both Fuel and Topography are `functional` data layers, then use screen_size
            and pixel scale
        If either Fuel or Topography is Operational, use calculated pixel scale and
            screen size.
        '''
        if self.terrain.topography.type.lower() and self.terrain.fuel.type.lower(
        ) == 'functional':
            setattr(self.area, 'terrain_scale',
                    self.area.pixel_scale * self.area.terrain_size)
        elif self.terrain.topography.type.lower() or self.terrain.fuel.type.lower(
        ) == 'operational':
            args = self.operational
            center = (args.latitude, args.longitude)
            if args.seed is not None:
                center = fuel(args.seed)
                try:
                    # TODO: center point may not include any data in CA, check
                    # and/or re-try a new point
                    self.lat_long_box = LatLongBox(center, args.height, args.width,
                                                   args.resolution)
                except ValueError:
                    message = ('Latitude and longitude are not contained in center point '
                               f'{center} in the database, retrying.')
                    log.error(message)
                    raise ConfigError(message)
            else:
                self.lat_long_box = LatLongBox(center, args.height, args.width,
                                               args.resolution)
            self.height = self.lat_long_box.tr[0] - self.lat_long_box.bl[0]
            self.width = self.height[0]
            self.screen_size = self.width
            # Convert to feet for use with rothermel
            self.pixel_scale = 3.28084 * args.resolution

            setattr(self.area, 'screen_size', self.width)
            setattr(self.area, 'pixel_scale', self.pixel_scale)
        else:
            message = ('Unable to load topography of type '
                       f'{self.terrain.topography.type}. Please check your config file '
                       f'and select from the following: {self._possible_layers}')
            log.error(message)
            raise ConfigError(message)

    def _set_topography_layer(self) -> None:
        '''
        Sets the `config.terrain.topography.layer` as either a `functional` or
        `operational` type

        The `config.terrain.topography.layer` value will be set to
        `FunctionalTopographyLayer` or a `OperationalTopographyLayer`, depending on the
        how the `config.terrain.topography.type` is set.

        If the `config.terrain.topography.type` is set to `functional`, the function is
        chosen by `config.terrain.topography.functional.function` and the arguments are
        used from their respective sections.
        '''
        if self.terrain.topography.type.lower() == 'functional':
            elevation = self.terrain.topography.functional.function.lower()
            # Now we can set the function again
            if elevation == 'perlin':
                # Reset the value, if we are resetting the function
                args = self.terrain.topography.functional.perlin
                noise = PerlinNoise2D(args.amplitude, args.shape, args.resolution,
                                      args.seed)
                noise.precompute()
                topo_layer = FunctionalTopographyLayer(self.area.screen_size,
                                                       self.area.screen_size,
                                                       noise.fn,
                                                       name='perlin')
                setattr(self.terrain.topography, 'layer', topo_layer)
            elif elevation == 'gaussian':
                # Reset the value, if we are resetting the function
                args = self.terrain.topography.functional.gaussian
                noise = gaussian(args.amplitude, args.mu_x, args.mu_y, args.sigma_x,
                                 args.sigma_y)
                topo_layer = FunctionalTopographyLayer(self.area.screen_size,
                                                       self.area.screen_size,
                                                       noise,
                                                       name='gaussian')
                setattr(self.terrain.topography, 'layer', topo_layer)
            elif elevation == 'flat':
                # Reset the value, if we are resetting the function
                topo_layer = FunctionalTopographyLayer(self.area.screen_size,
                                                       self.area.screen_size,
                                                       flat(),
                                                       name='flat')
                setattr(self.terrain.topography, 'layer', topo_layer)
            else:
                message = ('The user-defined topography is set to '
                           f'{self.terrain.topography.functional.function} when it can '
                           'only be one of these values: '
                           f'{self._possible_functional_topography}')
                log.error(message)
                raise ConfigError(message)
        elif self.terrain.topography.type.lower() == 'operational':
            topo_layer = OperationalTopographyLayer(self.lat_long_box)
            setattr(self.terrain.topography, 'layer', topo_layer)
        else:
            message = ('Unable to load topography of type '
                       f'{self.terrain.topography.type}. Please check your config file '
                       f'and select from the following: {self._possible_layers}')
            log.error(message)
            raise ConfigError(message)

    def _set_fuel_layer(self) -> None:
        '''
        Sets the `config.terrain.fuel.layer` as either a `functional` or `operational`
        type

        The `config.terrain.fuel.layer` value will be set to `FunctionalFuelLayer`
        or a `OperationalFuelLayer`, depending on the how the
        `config.terrain.fuel.type` is set.

        If the `config.terrain.fuel.type` is set to `functional`, the function is
        chosen by `config.terrain.fuel.functional.function` and the arguments are
        used from their respective sections.
        '''
        if self.terrain.fuel.type.lower() == 'functional':
            fuel = self.terrain.fuel.functional.function.lower()
            if fuel == 'chaparral':
                args = self.terrain.fuel.functional.chaparral
                fn = chaparral_fn(args.seed)
                fuel_layer = FunctionalFuelLayer(self.area.screen_size,
                                                 self.area.screen_size,
                                                 fn,
                                                 name='chaparral')
                setattr(self.terrain.fuel, 'layer', fuel_layer)
            else:
                message = ('The user-defined fuel array function is set to '
                           f'{self.terrain.fuel.type}, when it can only be one of '
                           f'these values: {self._possible_functional_fuel}')
                log.error(message)
        elif self.terrain.fuel.type.lower() == 'operational':
            fuel_layer = OperationalFuelLayer(self.lat_long_box)
            setattr(self.terrain.fuel, 'layer', fuel_layer)
        else:
            message = ('Unable to load fuel of type '
                       f'{self.terrain.fuel.type}. Please check your config file '
                       f'and select from the following: {self._possible_layers}')
            log.error(message)
            raise ConfigError(message)

    def _set_wind_function(self) -> None:
        '''
        Reset the attributes `self.wind.speed` and `self.wind.direction` based on the
        wind function assigned in the config

        Before, as read in from the YAML, the wind function was just a string. After
        calling this, it sets the `self.wind.speed` and `self.wind.direction` to the
        values generated by the wind function or to constant values.

        All subsequent wind functions should be instantiated here and set
        `self.wind.speed` and `self.wind.direction` to arrays of size
        (`self.area.screen_size`, `self.area.screen_size`) with wind values at each pixel.
        '''
        if self.wind.function.lower() == 'cfd':
            # Check if wind files have been generated
            cfd_generated = os.path.isfile(
                'generated_wind_directions.npy') and os.path.isfile(
                    'generated_wind_magnitudes.npy')
            if cfd_generated is False:
                log.error('Missing pregenerated cfd npy files, switching to perlin')
                self.wind_function = 'perlin'
            else:
                map_wind_speed = np.load('generated_wind_magnitudes.npy')
                map_wind_direction = np.load('generated_wind_directions.npy')
                map_wind_speed = scale_ms_to_ftpm(map_wind_speed)
                setattr(self.wind, 'speed', map_wind_speed)
                setattr(self.wind, 'direction', np.rint(map_wind_direction))
        elif self.wind.function.lower() == 'perlin':
            speed_min = mph_to_ftpm(self.wind.perlin.speed.min)
            speed_max = mph_to_ftpm(self.wind.perlin.speed.max)
            wind_map = WindController()
            wind_map.init_wind_speed_generator(
                self.wind.perlin.speed.seed, self.wind.perlin.speed.scale,
                self.wind.perlin.speed.octaves, self.wind.perlin.speed.persistence,
                self.wind.perlin.speed.lacunarity, speed_min, speed_max,
                self.area.screen_size)
            wind_map.init_wind_direction_generator(
                self.wind.perlin.direction.seed, self.wind.perlin.direction.scale,
                self.wind.perlin.direction.octaves,
                self.wind.perlin.direction.persistence,
                self.wind.perlin.direction.lacunarity, self.wind.perlin.direction.min,
                self.wind.perlin.direction.max, self.area.screen_size)
            setattr(self.wind, 'speed', wind_map.map_wind_speed)
            setattr(self.wind, 'direction', wind_map.map_wind_direction)
        elif self.wind.function.lower() == 'simple':
            # Convert wind speed to ft/min
            wind_speed = mph_to_ftpm(self.wind.simple.speed)
            speed = np.full((self.area.screen_size, self.area.screen_size),
                            wind_speed,
                            dtype=np.float32)
            direction = np.full((self.area.screen_size, self.area.screen_size),
                                self.wind.simple.direction,
                                dtype=np.float32)
            setattr(self.wind, 'speed', speed)
            setattr(self.wind, 'direction', direction)
        else:
            message = ('The user-defined wind function is set to '
                       f'{self.wind.function} when it can only be one of '
                       f'these values: {self._possible_wind}')
            log.error(message)
            raise ConfigError(message)

    def _set_runtime(self) -> None:
        '''
        Set the `simulation.runtime` variable to a set number of minutes based on a
        string
        '''
        if isinstance(self.simulation.runtime, int):
            runtime = f'{self.simulation.runtime}m'
        else:
            runtime = self.simulation.runtime
        setattr(self.simulation, 'runtime', str_to_minutes(runtime))

    def get_cfd_wind_map(self) -> WindController2:
        args = self.terrain.perlin
        terrain_map = np.zeros((args.shape[0], args.shape[1]))
        for x in range(0, args.shape[0]):
            for y in range(0, args.shape[1]):
                terrain_map[x][y] = self.terrain.layer.data[y, x]
        '''
        TODO: Need to optimize cfd to work on 3d space.  For now we get the average
        terrain height and for values slightly greater than that average we will
        count as terrain features for cfd
        '''
        terrain_space = np.average(terrain_map) + (
            (np.max(terrain_map) - np.average(terrain_map)) / 4)

        def create_cfd_terrain(e):
            if e < terrain_space:
                return 0
            return 1

        cfd_func = np.vectorize(create_cfd_terrain)

        terrain_map = cfd_func(terrain_map)

        # Assumption: CFD Algorithm uses m/s
        source_speed = mph_to_ms(self.wind.cfd.speed)
        source_direction = self.wind.cfd.direction
        wind_map = WindController2(terrain_features=terrain_map,
                                   wind_direction=source_direction,
                                   wind_speed=source_speed)
        return wind_map

    def reset_topography_layer(self,
                               seed: int = None,
                               location: Tuple[float, float] = None) -> None:
        '''
        Reset the topography layer with a different seed or a different location

        Arguments:
            seed: The input used in generating the random elevation function.
            type: The type of topography layer to use.
            location: The location of the topography layer i.e. (latitude, longitude)
        '''
        # Assume functional type
        if seed is None:
            message = 'No seed provided for topography layer'
            log.warning(message)
        else:
            # Set the seed class attribute so that the function uses it correctly
            self.terrain.topography.functional.perlin.seed = seed
            # Set the seed dictionary value so that if the config is later saved, it is
            # reflected in the saved config.yml
            self.data['terrain']['topography']['functional']['perlin']['seed'] = seed

        # Also set the seed for the operational topography layer
        self._reset_operational_variables(seed, location)

        self._set_terrain_scale()
        self._set_topography_layer()

    def reset_fuel_layer(self,
                         seed: int = None,
                         location: Tuple[float, float] = None) -> None:
        '''
        Reset the fuel layer with a different seed or a different location

        Arguments:
            seed: The input used in generating the random fuel array function.
        '''
        # Assume functional type
        if seed is None:
            message = 'No seed provided for functional topography layer'
            log.warning(message)
        else:
            # Set the seed class attribute so that the function uses it correctly
            self.terrain.fuel.functional.chaparral.seed = seed
            # Set the seed dictionary value so that if the config is later saved, it is
            # reflected in the saved config.yml
            self.data['terrain']['fuel']['functional']['chaparral']['seed'] = seed

        # Also set the seed for the operational topography layer
        self._reset_operational_variables(seed, location)

        self._set_terrain_scale()
        self._set_fuel_layer()

    def _reset_operational_variables(self, seed: int, location: Tuple[float,
                                                                      float]) -> None:
        '''
        Reset the operational variables for the topography and fuel layers and handle
        error logging and messaging

        Arguments:
            seed: The input used in generating the random topography or fuel function or
                  location.
            location: The location of the topography or fuel layer
                      i.e. (latitude, longitude)
        '''
        if seed is not None and location is not None:
            message = ('Cannot set seed and location for operational topography '
                       'layer at the same time')
            log.error(message)
            raise ValueError(message)
        elif seed is not None:
            self.operational.seed = seed
        elif location is not None:
            self.operational.latitude = int(location[0])
            self.operational.longitude = int(location[1])

    def reset_wind_function(self,
                            speed_seed: int = None,
                            direction_seed: int = None) -> None:
        '''
        Reset the wind function with a different seed.

        Arguments:
            speed_seed: The input used in generating the random wind speed function.
            direction_seed: The input used in generating the random wind direction
                            function.
        '''
        # Set the seed class attribute so that the function uses it correctly
        # Only set each seed if it has been passed into the function to be changed
        if speed_seed is not None:
            self.wind.perlin.speed.seed = speed_seed
            # Set the seed dictionary value so that if the config is later saved, it is
            # reflected in the saved config.yml
            self.data['wind']['perlin']['speed']['seed'] = speed_seed

        if direction_seed is not None:
            self.wind.perlin.direction.seed = direction_seed
            # Set the seed dictionary value so that if the config is later saved, it is
            # reflected in the saved config.yml
            self.data['wind']['perlin']['direction']['seed'] = direction_seed

        # No reason to run _set_wind_function if it doesn't change
        if speed_seed is not None and direction_seed is not None:
            self._set_wind_function()

    def save(self, path: Path) -> None:
        '''
        Save the current config to the specified `path`

        Arguments:
            path: The path and filename of the output YAML file.
        '''
        with open(path, 'w') as f:
            yaml.dump(self.data, f)
