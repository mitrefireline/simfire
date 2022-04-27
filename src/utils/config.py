import numpy as np

import yaml
import os.path
from typing import Any
from pathlib import Path
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
        self._possible_elevations = ('perlin', 'gaussian', 'flat')
        self._possible_wind = ('perlin', 'simple')
        self._possible_fuel_arrays = ('chaparral')
        self._load()
        self._set_attributes()
        self._set_runtime()
        self._set_terrain_scale()
        self._set_elevation_function()
        self._set_fuel_array_function()
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
                    log.error(f'Error parsing YAML file at {self.path}')
        except FileNotFoundError:
            log.error(f'Error opening YAML file at {self.path}. Does it exist?')

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
        Set the attribute `self.area.terrain_scale` defined as
        `self.area.pixel_scale * self.area.terrain_size`
        '''
        if 'functional' in str(self.data['terrain']['terrain']).lower() and str(
                self.data['fuel']['fuel']).lower():
            setattr(self.area, 'terrain_scale',
                    self.area.pixel_scale * self.area.terrain_size)
        elif 'operational' in str(self.data['terrain']['terrain']).lower() and str(
                self.data['fuel']['fuel']).lower():
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
                    log.error('Latitude and longitude are not contained in center point '
                              f'{center} in the database, retrying.')
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

    def _set_elevation_function(self) -> None:
        '''
        Reset the attribute `self.terrain.elevation_function`

        Before, as read in from the YAML, the elevation function was just a string. After
        calling this, it becomes an actual function with all of the precompute values
        from the config passed in.
        '''
        if 'functional' in str(self.data['terrain']['terrain']).lower():
            # Now we can set the function again
            if 'perlin' in str(self.data['terrain']['elevation_function']).lower():
                # Reset the value, if we are resetting the function
                self.terrain.elevation_function = 'perlin'
                args = self.terrain.perlin
                noise = PerlinNoise2D(args.amplitude, args.shape, args.resolution,
                                      args.seed)
                noise.precompute()
                elevation_layer = FunctionalTopographyLayer(self.area.screen_size,
                                                            self.area.screen_size,
                                                            noise.fn)
                setattr(self.terrain, 'elevation_function', elevation_layer)
            elif 'gaussian' in str(self.data['terrain']['elevation_function']).lower():
                # Reset the value, if we are resetting the function
                self.terrain.elevation_function = 'gaussian'
                args = self.terrain.gaussian
                noise = gaussian(args.amplitude, args.mu_x, args.mu_y, args.sigma_x,
                                 args.sigma_y)
                elevation_layer = FunctionalTopographyLayer(self.area.screen_size,
                                                            self.area.screen_size, noise)
                setattr(self.terrain, 'elevation_function', elevation_layer)
            elif 'flat' in str(self.data['terrain']['elevation_function']).lower():
                # Reset the value, if we are resetting the function
                self.terrain.elevation_function = 'flat'
                elevation_layer = FunctionalTopographyLayer(self.area.screen_size,
                                                            self.area.screen_size, flat())
                setattr(self.terrain, 'elevation_function', elevation_layer)
            else:
                log.error('The user-defined elevation function is set to '
                          f'{self.terrain.elevation_function} when it can only be one of '
                          f'these values: {self._possible_elevations}')
                raise ValueError
        elif 'operational' in str(self.terrain.terrain).lower():
            # Reset the value, if we are resetting the function
            self.terrain.elevation_function = 'operational'
            topo_layer = OperationalTopographyLayer(self.lat_long_box)
            setattr(self.terrain, 'elevation_function', topo_layer)

    def _set_fuel_array_function(self) -> None:
        '''
        Reset the attribute `self.terrain.fuel_array_fn`

        Before, as read in from the YAML, the fuel array function was just a string. After
        calling this, it becomes an actual function with all of the precompute values
        from the config passed in.
        '''
        if 'functional' in str(self.data['fuel']['fuel']).lower():
            # Now we can set the function again
            if 'chaparral' in str(self.data['fuel']['fuel_array_function']).lower():
                self.fuel.fuel_array_function = 'chaparral'
                args = self.fuel.chaparral
                fn = chaparral_fn(args.seed)
                fuel_layer = FunctionalFuelLayer(self.area.screen_size,
                                                 self.area.screen_size, fn)
                setattr(self.fuel, 'fuel_array_function', fuel_layer)
            else:
                log.error('The user-defined fuel array function is set to '
                          f'{self.fuel.fuel_array_function}, when it can only be one of '
                          f'these values: {self._possible_fuel_arrays}')
        elif 'operational' in str(self.fuel.fuel).lower():
            # Reset the value, if we are resetting the function
            self.fuel.fuel_array_function = 'operational'
            topo_layer = OperationalFuelLayer(self.lat_long_box)
            setattr(self.fuel, 'fuel_array_function', topo_layer)

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
        if self.wind.wind_function.lower() == 'cfd':
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
        if self.wind.wind_function.lower() == 'perlin':
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
        elif self.wind.wind_function.lower() == 'simple':
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
            log.error('The user-defined wind function is set to '
                      f'{self.wind.wind_function} when it can only be one of '
                      f'these values: {self._possible_wind}')

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
                terrain_map[x][y] = self.terrain.elevation_function(x, y)
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
        # wind_map.generate_wind_field(source_direction, source_speed,
        #                                 self.area.screen_size)
        return wind_map

    def reset_elevation_function(self, seed: int) -> None:
        '''
        Reset the elevation function with a different seed.

        Arguments:
            seed: The input used in generating the random elevation function.
        '''
        # Set the seed class attribute so that the function uses it correctly
        self.terrain.perlin.seed = seed
        # Set the seed dictionary value so that if the config is later saved, it is
        # reflected in the saved config.yml
        self.data['terrain']['perlin']['seed'] = seed
        self._set_elevation_function()

    def reset_fuel_array_function(self, seed: int) -> None:
        '''
        Reset the fuel array function with a different seed.

        Arguments:
            seed: The input used in generating the random fuel array function.
        '''
        # Set the seed class attribute so that the function uses it correctly
        self.fuel.chaparral.seed = seed
        # Set the seed dictionary value so that if the config is later saved, it is
        # reflected in the saved config.yml
        self.data['fuel']['chaparral']['seed'] = seed
        self._set_fuel_array_function()

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
