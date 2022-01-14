import yaml
from pathlib import Path
from typing import Any
from yaml.scanner import ScannerError

from ..utils.log import create_logger
from ..utils.units import convert_to_minutes
from ..world.elevation_functions import PerlinNoise2D, flat, gaussian
from ..world.fuel_array_functions import chaparral_fn

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
    def __init__(self, path: Path) -> None:
        '''
        Arguments:
            path: The path to the `config.yml` file. This file can have any name, but
                  needs to be a valid YAML.

        Returns:
            None
        '''
        self.path = path
        self._possible_elevations = ('perlin', 'gaussian', 'flat')
        self._possible_fuel_arrays = ('chaparral')
        self._load()
        self._set_attributes()
        self._set_runtime()
        self._set_terrain_scale()
        self._set_elevation_function()
        self._set_fuel_array_function()

    def _load(self) -> None:
        '''
        Loads the YAML file and assigns to `self.data`
        '''
        try:
            with open(self.path, 'r') as f:
                try:
                    self.data = yaml.safe_load(f)
                except ScannerError as e:
                    log.error(f'Error parsing YAML file at {self.path}:\n' f'{e.error}')
                    raise ScannerError
        except FileNotFoundError:
            log.error(f'Error opening YAML file at {self.path}. Does it exist?')
            raise FileNotFoundError

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
        setattr(self.area, 'terrain_scale',
                self.area.pixel_scale * self.area.terrain_size)

    def _set_elevation_function(self) -> None:
        '''
        Reset the attribute `self.terrain.elevation_function`

        Before, as read in from the YAML, the elevation function was just a string. After
        calling this, it becomes an actual function with all of the precompute values
        from the config passed in.
        '''
        if self.terrain.elevation_function.lower() == 'perlin':
            args = self.terrain.perlin
            noise = PerlinNoise2D(args.amplitude, args.shape, args.resolution, args.seed)
            noise.precompute()
            setattr(self.terrain, 'elevation_function', noise.fn)
        elif self.terrain.elevation_function.lower() == 'gaussian':
            args = self.terrain.gaussian
            noise = gaussian(args.amplitude, args.mu_x, args.mu_y, args.sigma_x,
                             args.sigma_y)
            setattr(self.terrain, 'elevation_function', noise.fn)
        elif self.terrain.elevation_function.lower() == 'flat':
            setattr(self.terrain, 'elevation_function', flat)
        else:
            log.error('The user-defined elevation function is set to '
                      f'{self.terrain.elevation_function} when it can only be one of '
                      f'these values: {self._possible_elevations}')
            raise ValueError

    def _set_fuel_array_function(self) -> None:
        '''
        Reset the attribute `self.terrain.fuel_array_fn`

        Before, as read in from the YAML, the fuel array function was just a string. After
        calling this, it becomes an actual function with all of the precompute values
        from the config passed in.
        '''
        if self.terrain.fuel_array_function.lower() == 'chaparral':
            args = self.terrain.chaparral
            fn = chaparral_fn(self.area.pixel_scale, self.area.pixel_scale, args.seed)
            setattr(self.terrain, 'fuel_array_function', fn)
        else:
            log.error('The user-defined fuel array function is set to '
                      f'{self.terrain.fuel_array_function}, when it can only be one of '
                      f'these values: {self._possible_fuel_arrays}')
            raise ValueError

    def _set_runtime(self) -> None:
        '''
        Set the `simulation.runtime` variable to a set number of minutes based on a
        string
        '''
        if isinstance(self.simulation.runtime, int):
            runtime = f'{self.simulation.runtime}m'
        else:
            runtime = self.simulation.runtime
        setattr(self.simulation, 'runtime', convert_to_minutes(runtime))

    def save(self, path: Path) -> None:
        '''
        Save the current config to the specified `path`

        Arguments:
            path: The path and filename of the output YAML file.

        Returns:
            None
        '''
        with open(path, 'w') as f:
            yaml.dump(self.data, f)
