from copy import deepcopy
import dataclasses
import numpy as np

import yaml
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
from yaml.parser import ParserError

from ..utils.log import create_logger
from ..utils.layers import (FuelLayer, FunctionalFuelLayer, LatLongBox,
                            OperationalTopographyLayer, FunctionalTopographyLayer,
                            OperationalFuelLayer, TopographyLayer)
from ..world.elevation_functions import PerlinNoise2D, flat, gaussian
from ..world.fuel_array_functions import chaparral_fn

log = create_logger(__name__)


class ConfigError(Exception):
    '''
    Exception class for Config class
    '''
    pass


@dataclasses.dataclass
class AreaConfig:
    screen_size: int
    pixel_scale: float

    def __post_init__(self) -> None:
        self.screen_size = int(self.screen_size)
        self.pixel_scale = float(self.pixel_scale)


@dataclasses.dataclass
class DisplayConfig:
    fire_size: int
    control_line_size: int

    def __post_init__(self) -> None:
        self.fire_size = int(self.fire_size)
        self.control_line_size = int(self.control_line_size)


@dataclasses.dataclass
class SimulationConfig:
    update_rate: float
    runtime: str
    headless: bool

    def __post_init__(self) -> None:
        self.update_rate = float(self.update_rate)
        self.runtime = str(self.runtime)
        self.headless = str(self.headless)


@dataclasses.dataclass
class MitigationConfig:
    ros_attenuation: bool

    def __post_init__(self) -> None:
        self.ros_attenuation = bool(self.ros_attenuation)


@dataclasses.dataclass
class OperationalConfig:
    latitude: float
    longitude: float
    height: float
    width: float
    resolution: float  # TODO: Make enum for resolution?

    def __post_init__(self) -> None:
        self.latitude = float(self.latitude)
        self.longitude = float(self.longitude)
        self.height = float(self.height)
        self.width = float(self.width)
        self.resolution = float(self.resolution)


@dataclasses.dataclass
class TerrainConfig:
    topography_layer: TopographyLayer
    fuel_layer: FuelLayer


@dataclasses.dataclass
class FireConfig:
    fire_initial_position: Tuple[int, int]
    max_fire_duration: int

    def __post_init__(self) -> None:
        self.fire_initial_position = tuple(
            map(int, self.fire_initial_position[1:-1].split(',')))
        self.max_fire_duration = int(self.max_fire_duration)


@dataclasses.dataclass
class EnvironmentConfig:
    moisture: float

    def __post_init__(self) -> None:
        self.moisture = float(self.moisture)


@dataclasses.dataclass
class WindConfig:
    speed: np.ndarray
    direction: np.ndarray


@dataclasses.dataclass
class Config:
    def __init__(self, path: Union[str, Path], cfd_precompute: bool = False) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.yaml_data = self._load_yaml()

        self.lat_long_box = self._make_lat_long_box()

        self.area = self._load_area()
        self.display = self._load_display()
        self.simulation = self._load_simulation()
        self.mitigation = self._load_mitigation()
        self.operational = self._load_operational()
        self.terrain = self._load_terrain()
        self.fire = self._load_fire()
        self.environment = self._load_environment()
        self.wind = self._load_wind()

    def _load_yaml(self) -> Dict[str, Any]:
        '''
        Loads the YAML file specified in self.path and returns the data as a dictionary.

        Returns:
            The YAML data as a dictionary
        '''
        try:
            with open(self.path, 'r') as f:
                try:
                    yaml_data = yaml.safe_load(f)
                except ParserError:
                    message = f'Error parsing YAML file at {self.path}'
                    log.error(message)
                    raise ConfigError(message)
        except FileNotFoundError:
            log.error(f'Error opening YAML file at {self.path}. Does it exist?')
        return yaml_data

    def _make_lat_long_box(self) -> Union[LatLongBox, None]:
        '''
        Optionally create the LatLongBox used by the data layers if any
        of the data layers are of type `operational`.

        Returns:
            None if none of the data layers are `operational`
            LatLongBox with the config coordinates and shape if any of
            the data layers are `operational`
        '''
        if self.yaml_data['terrain']['topography']['type'] == 'operational' \
           or self.yaml_data['terrain']['fuel']['type'] == 'operational':
            lat = self.yaml_data['operational']['latitude']
            lon = self.yaml_data['operational']['longitude']
            height = self.yaml_data['operational']['height']
            width = self.yaml_data['operational']['width']
            resolution = self.yaml_data['operational']['resolution']
            return LatLongBox((lat, lon), height, width, resolution)
        else:
            return None

    def _load_area(self) -> AreaConfig:
        '''
        Load the AreaConfig from the YAML data.

        returns:
            The YAML data converted to an AreaConfig dataclass
        '''
        # No processing needed for the AreaConfig
        return AreaConfig(**self.yaml_data['area'])

    def _load_display(self) -> DisplayConfig:
        '''
        Load the DisplayConfig from the YAML data.

        returns:
            The YAML data converted to a DisplayConfig dataclass
        '''
        # No processing needed for the DisplayConfig
        return DisplayConfig(**self.yaml_data['display'])

    def _load_simulation(self) -> SimulationConfig:
        '''
        Load the SimulationConfig from the YAML data.

        returns:
            The YAML data converted to a SimulationConfig dataclass
        '''
        # No processing needed for the SimulationConfig
        return SimulationConfig(**self.yaml_data['simulation'])

    def _load_mitigation(self) -> MitigationConfig:
        '''
        Load the MitigationConfig from the YAML data.

        returns:
            The YAML data converted to a MitigationConfig dataclass
        '''
        # No processing needed for the MitigationConfig
        return MitigationConfig(**self.yaml_data['mitigation'])

    def _load_operational(self) -> OperationalConfig:
        '''
        Load the OperationalConfig from the YAML data.

        returns:
            The YAML data converted to an OperationalConfig dataclass
        '''
        # No processing needed for the OperationalConfig
        return OperationalConfig(**self.yaml_data['operational'])

    def _load_terrain(self) -> TerrainConfig:
        '''
        Load the TerrainConfig from the YAML data.

        returns:
            The YAML data converted to a TerrainConfig dataclass
        '''
        topo_layer = self._create_topography_layer(init=True)

        fuel_layer = self._create_fuel_layer(init=True)

        return TerrainConfig(topo_layer, fuel_layer)

    def _create_topography_layer(self,
                                 init: bool = False,
                                 seed: Optional[int] = None) -> TopographyLayer:
        '''
        Create a TopographyLayer given the config parameters.
        This is used for initalization and after resetting the layer seeds.

        Arguments:
            seed: A randomization seed used by

        Returns:
            A FunctionalTopographyLayer that utilizes the fuction specified by
            fn_name and the keyword arguments in kwargs
        '''
        topo_type = self.yaml_data['terrain']['topography']['type']
        if topo_type == 'operational':
            topo_layer = OperationalTopographyLayer(self.lat_long_box)
        elif topo_type == 'functional':
            fn_name = self.yaml_data['terrain']['topography']['functional']['function']
            try:
                kwargs = self.yaml_data['terrain']['topography']['functional'][fn_name]
            # No kwargs found (flat is an example of this)
            except KeyError:
                kwargs = {}
            # Reset the seed if this isn't the inital creation
            if 'seed' in kwargs and not init:
                kwargs['seed'] = seed
            if fn_name == 'perlin':
                new_kwargs = deepcopy(kwargs)
                new_kwargs['shape'] = (self.yaml_data['area']['screen_size'],
                                       self.yaml_data['area']['screen_size'])
                # Convert the input from string `(x, y)` to tuple of ints (x, y)
                if isinstance(nk := new_kwargs['res'], str):
                    new_kwargs['res'] = tuple(map(int, nk[1:-1].split(',')))
                noise = PerlinNoise2D(**new_kwargs)
                fn = noise.fn
            elif fn_name == 'gaussian':
                fn = gaussian(**kwargs)
            elif fn_name == 'flat':
                fn = flat()
            else:
                raise ValueError(f'The specified topography function ({fn_name}) '
                                 'is not valid.')
            topo_layer = FunctionalTopographyLayer(self.yaml_data['area']['screen_size'],
                                                   self.yaml_data['area']['screen_size'],
                                                   fn)
        else:
            raise ValueError(f'The specified topography type ({topo_type}) '
                             'is not supported')

        return topo_layer

    def _create_fuel_layer(self,
                           init: bool = False,
                           seed: Optional[int] = None) -> FuelLayer:
        '''
        Create a FuelLayer given the config parameters.
        This is used for initalization and after resetting the layer seeds.

        Returns:
            A FunctionalFuelLayer that utilizes the fuction specified by
            fn_name and the keyword arguments in kwargs
        '''
        fuel_type = self.yaml_data['terrain']['fuel']['type']
        if fuel_type == 'operational':
            fuel_layer = OperationalFuelLayer(self.lat_long_box)
        elif fuel_type == 'functional':
            fn_name = self.yaml_data['terrain']['fuel']['functional']['function']
            try:
                kwargs = self.yaml_data['terrain']['fuel']['functional'][fn_name]
            # No kwargs found (some functions don't need input arguments)
            except KeyError:
                kwargs = {}
            # Reset the seed if this isn't the inital creation
            if 'seed' in kwargs and not init:
                kwargs['seed'] = seed
            if fn_name == 'chaparral':
                fn = chaparral_fn(**kwargs)
            else:
                raise ValueError(f'The specified fuel function ({fn_name}) '
                                 'is not valid.')
            fuel_layer = FunctionalFuelLayer(self.yaml_data['area']['screen_size'],
                                             self.yaml_data['area']['screen_size'], fn)
        else:
            raise ValueError(f'The specified fuel type ({fuel_type}) '
                             'is not supported')

        return fuel_layer

    def _load_fire(self) -> FireConfig:
        '''
        Load the FireConfig from the YAML data.

        Returns:
            The YAML data converted to a FireConfig dataclass
        '''
        # No processing needed for the FireConfig
        return FireConfig(**self.yaml_data['fire'])

    def _load_environment(self) -> EnvironmentConfig:
        '''
        Load the EnvironmentConfig from the YAML data.

        Returns:
            The YAML data converted to a EnvironmentConfig dataclass
        '''
        # No processing needed for the EnvironmentConfig
        return EnvironmentConfig(**self.yaml_data['environment'])

    def _load_wind(self) -> WindConfig:
        '''
        Load the WindConfig from the YAML data.

        Returns:
            The YAML data converted to a WindConfig dataclass
        '''
        # Only support simple for now
        # TODO: Figure out how Perlin and CFD create wind
        fn_name = self.yaml_data['wind']['function']
        if fn_name == 'simple':
            arr_shape = (self.yaml_data['area']['screen_size'],
                         self.yaml_data['area']['screen_size'])
            speed = self.yaml_data['wind']['simple']['speed']
            direction = self.yaml_data['wind']['simple']['direction']
            speed_arr = np.full(arr_shape, speed)
            direction_arr = np.full(arr_shape, direction)
        else:
            raise ValueError(f'Wind type {fn_name} is not supported')

        return WindConfig(speed_arr, direction_arr)

    def reset_terrain(self,
                      topography_seed: Optional[int] = None,
                      fuel_seed: Optional[int] = None,
                      location: Optional[Tuple[float, float]] = None) -> None:
        '''
        Reset the terrain functional generation seeds if using functional data,
        or reset the terrain lat/long location if using operational data.

        Arguments:
            topography_seed: The seed used to randomize functional topography generation
            fuel_seed: The seed used to randomize functional fuel generation
            location: A new center-point for the operational topography and fuel data
        '''
        if location is not None:
            # Since all operational layers use the LatLongBox, we can update
            # the yaml data and the LatLongBox at the class level
            lat, long = location
            self.yaml_data['operational']['latitude'] = lat
            self.yaml_data['operational']['longitude'] = long
            self.lat_long_box = self._make_lat_long_box()

        # Since not all functional layers use a seed (such as flat and gaussian),
        # we update the seed values during layer creation
        topo_layer = self._create_topography_layer(seed=topography_seed)
        fuel_layer = self._create_fuel_layer(seed=fuel_seed)

        self.terrain = TerrainConfig(topo_layer, fuel_layer)

    def reset_wind(self,
                   speed_seed: Optional[int] = None,
                   direction_seed: Optional[int] = None) -> None:
        '''
        Reset the wind speed and direction seeds.

        Arguments:
            speed_seed: The seed used to randomize wind speed generation
            direction_seed: The seed used to randomize wind direction generation
        '''
        # TODO: Implement wind stuff that's not just `simple`
        pass

    def save(self, path: Union[str, Path]) -> None:
        '''
        Save the current config to the specified path.

        Arguments:
            path: The path and filename of the output YAML file
        '''
        with open(path, 'w') as f:
            yaml.dump(self.yaml_data, f)
