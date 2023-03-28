"""
Module containing all config parsing and dataclass logic.
"""
import dataclasses
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import yaml  # type: ignore
from yaml.parser import ParserError  # type: ignore

from ..world.elevation_functions import flat, gaussian, perlin
from ..world.fuel_array_functions import chaparral_fn
from ..world.wind_mechanics.wind_controller import (
    WindController,
    WindControllerCFD,
)
from .layers import (
    BurnProbabilityLayer,
    DataLayer,
    FuelLayer,
    FunctionalBurnProbabilityLayer,
    FunctionalFuelLayer,
    FunctionalTopographyLayer,
    HistoricalLayer,
    LatLongBox,
    OperationalBurnProbabilityLayer,
    OperationalFuelLayer,
    OperationalTopographyLayer,
    TopographyLayer,
)
from .log import create_logger
from .units import mph_to_ftpm, scale_ms_to_ftpm, str_to_minutes

log = create_logger(__name__)


class ConfigError(Exception):
    """
    Exception class for Config class
    """

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
    agent_size: int
    rescale_size: Optional[int] = None

    def __post_init__(self) -> None:
        self.fire_size = int(self.fire_size)
        self.control_line_size = int(self.control_line_size)
        self.agent_size = int(self.agent_size)
        if self.rescale_size is not None:
            try:
                self.rescale_size = int(self.rescale_size)
            except ValueError:
                if isinstance(self.rescale_size, str):
                    if self.rescale_size.upper() == "NONE":
                        self.rescale_size = None
                    else:
                        raise ValueError(
                            f"Specified value  of {self.rescale_size} for "
                            "config:display:rescale_size is not valid. "
                            "Specify either an integer value or None"
                        )
                else:
                    raise TypeError(
                        "Speicified type of config:display:rescale_size "
                        f"({type(self.rescale_size)}) with value "
                        f"{self.rescale_size} is invalid. rescale_size "
                        "should be int or None."
                    )


@dataclasses.dataclass
class SimulationConfig:
    def __init__(
        self,
        update_rate: str,
        runtime: str,
        headless: bool,
        draw_spread_graph: bool,
        record: bool,
        save_data: bool,
        data_type: str,
        save_path: str,
    ) -> None:
        self.update_rate = float(update_rate)
        self.runtime = str_to_minutes(runtime)
        self.headless = headless
        self.draw_spread_graph = draw_spread_graph
        self.record = record
        self.save_data = save_data
        data_type = data_type.lower()
        if data_type not in ["npy", "h5"]:
            raise ConfigError(
                f"Specified data_type {data_type} is not valid. "
                "Specify either 'npy' or 'h5'."
            )
        self.data_type = data_type
        self.save_path = Path(save_path)


@dataclasses.dataclass
class MitigationConfig:
    ros_attenuation: bool

    def __post_init__(self) -> None:
        self.ros_attenuation = bool(self.ros_attenuation)


@dataclasses.dataclass
class OperationalConfig:
    seed: Optional[int]
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
class HistoricalConfig:
    """
    Class that tracks historical layer.
    """

    use: bool
    fire_init_pos_lat: float
    fire_init_pos_long: float
    name: str
    year: str
    historical_layer = DataLayer


@dataclasses.dataclass
class FunctionalConfig:
    """
    Class that tracks functional layer names and keyword arguments.
    """

    name: str
    kwargs: Dict[str, Any]


@dataclasses.dataclass
class TerrainConfig:
    """
    Class that tracks the terrain topography and fuel layers.
    The fuel and terrain function fields are optional. They are used for
    functional layers and ignored for operational layers.
    """

    topography_type: str
    topography_layer: TopographyLayer
    fuel_type: str
    fuel_layer: FuelLayer
    topography_function: Optional[FunctionalConfig] = None
    fuel_function: Optional[FunctionalConfig] = None


@dataclasses.dataclass
class FireConfig:
    fire_initial_position: Tuple[int, int]
    max_fire_duration: int
    seed: Optional[int] = None


@dataclasses.dataclass
class EnvironmentConfig:
    moisture: float

    def __post_init__(self) -> None:
        self.moisture = float(self.moisture)


@dataclasses.dataclass
class WindConfig:
    speed: np.ndarray
    direction: np.ndarray
    speed_function: Optional[FunctionalConfig] = None
    direction_function: Optional[FunctionalConfig] = None


@dataclasses.dataclass
class Config:
    def __init__(self, path: Union[str, Path], cfd_precompute: bool = False) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.yaml_data = self._load_yaml()

        # Save the original screen size in case the simulation changes from
        # operational to functional
        self.original_screen_size = self.yaml_data["area"]["screen_size"]

        self.lat_long_box, self.historical_layer = self._make_lat_long_box()

        self.area = self._load_area()
        self.display = self._load_display()
        self.simulation = self._load_simulation()
        self.mitigation = self._load_mitigation()
        self.operational = self._load_operational()
        self.terrain = self._load_terrain()
        self.fire = self._load_fire()
        self.environment = self._load_environment()
        if cfd_precompute is False:
            self.wind = self._load_wind()
        else:
            self.cfd_setup = self._cfd_wind_setup()

    def _load_yaml(self) -> Dict[str, Any]:
        """
        Loads the YAML file specified in self.path and returns the data as a dictionary.

        Returns:
            The YAML data as a dictionary
        """
        try:
            with open(self.path, "r") as f:
                try:
                    yaml_data = yaml.safe_load(f)
                except ParserError:
                    message = f"Error parsing YAML file at {self.path}"
                    log.error(message)
                    raise ConfigError(message)
        except FileNotFoundError:
            message = f"Error opening YAML file at {self.path}. Does it exist?"
            log.error(message)
            raise ConfigError(message)
        return yaml_data

    def _make_lat_long_box(
        self,
    ) -> Tuple[Optional[LatLongBox], Optional[HistoricalLayer]]:
        """
        Optionally create the LatLongBox used by the data layers if any
        of the data layers are of type `operational`.

        Returns:
            None if none of the data layers are `operational`
            LatLongBox with the config coordinates and shape if any of
            the data layers are `operational`
        """

        if (
            self.yaml_data["terrain"]["topography"]["type"] == "operational"
            or self.yaml_data["terrain"]["fuel"]["type"] == "operational"
        ):
            self._set_all_combos()
            if self.yaml_data["operational"]["seed"] is not None:
                lat_long_box: Optional[LatLongBox] = self._randomly_select_box(
                    self.yaml_data["operational"]["seed"]
                )
                valid = self._check_lat_long(lat_long_box)
                if not valid:
                    if lat_long_box is None:
                        message = (
                            "Lat/Long box is not valid and was not created successfully."
                        )
                        log.error(message)
                        raise ConfigError(message)
                    message = (
                        "Lat/Long box is not valid. Data does not "
                        f"exist for latitude {lat_long_box.center[0]} and "
                        f"longitude {lat_long_box.center[1]} with "
                        f"a height of {lat_long_box.height} and width of "
                        f"{lat_long_box.width}. Checking another location "
                        f"with seed {self.yaml_data['operational']['seed'] + 1}."
                    )
                    log.warning(message)
                    self.yaml_data["operational"]["seed"] += 1
                    lat_long_box, _ = self._make_lat_long_box()
            else:
                lat = self.yaml_data["operational"]["latitude"]
                lon = self.yaml_data["operational"]["longitude"]
                height = self.yaml_data["operational"]["height"]
                width = self.yaml_data["operational"]["width"]
                resolution = self.yaml_data["operational"]["resolution"]
                lat_long_box = LatLongBox((lat, lon), height, width, resolution)
                valid = self._check_lat_long(lat_long_box)
                if not valid:
                    message = (
                        "Lat/Long box is not valid. Data does not "
                        f"exist for latitude {lat} and longitude {lon} with "
                        f"a height of {height} and width of {width}."
                    )
                    log.error(message)
                    raise ConfigError(message)
            return lat_long_box, None
        else:
            return None, None

    def _check_lat_long(self, lat_long_box: Optional[LatLongBox]) -> bool:
        """
        Check that the lat/long box is within the bounds of the data.

        Args:
            lat_long_box: The LatLongBox to check

        Returns:
            True if the LatLongBox is within the bounds of the data
        """
        if lat_long_box is None:
            return False
        for tile in lat_long_box.tiles.values():
            for range in tile:
                if range not in self._all_combos:
                    return False
        return True

    def _set_all_combos(self) -> None:
        """
        Get all possible combinations of latitude and longitude for the
        data layers.
        """
        data_path = Path("/nfs/lslab2/fireline/data/fuel/")
        res = str(self.yaml_data["operational"]["resolution"]) + "m"
        data_path = data_path / res / "old_2020"
        all_files = [
            f.stem
            for f in data_path.iterdir()
            if f.is_dir() and "n" in f.stem and "w" in f.stem
        ]
        self._all_combos = [
            (
                int(str(f).split(".")[0][1:].split("w")[0]),
                int(str(f).split(".")[0][1:].split("w")[1]),
            )
            for f in all_files
        ]

    def _randomly_select_box(self, seed: int) -> LatLongBox:
        """
        Randomly select a latitude and longitude for the LatLongBox.

        Args:
            seed: The seed to use for the random number generator

        Returns:
            A LatLongBox with a random latitude and longitude
        """
        random.seed(seed)  # nosec
        lat, lon = random.choice(self._all_combos)  # nosec
        lat = round(random.random(), 4) + lat  # nosec
        lon = round(random.random(), 4) + lon  # nosec
        height = self.yaml_data["operational"]["height"]
        width = self.yaml_data["operational"]["width"]
        resolution = self.yaml_data["operational"]["resolution"]
        lat_long_box = LatLongBox((lat, lon), height, width, resolution)
        return lat_long_box

    def _load_area(self) -> AreaConfig:
        """
        Load the AreaConfig from the YAML data.

        returns:
            The YAML data converted to an AreaConfig dataclass
        """
        # No processing needed for the AreaConfig
        if self.lat_long_box is not None:
            # Overwite the screen_size since operational data will determine
            # its own screen_size based on lat/long input
            # Height and width are the same since we assume square input
            height = int(self.lat_long_box.tr[0][0] - self.lat_long_box.bl[0][0])
            self.yaml_data["area"]["screen_size"] = height
        return AreaConfig(**self.yaml_data["area"])

    def _load_display(self) -> DisplayConfig:
        """
        Load the DisplayConfig from the YAML data.

        returns:
            The YAML data converted to a DisplayConfig dataclass
        """
        # No processing needed for the DisplayConfig
        return DisplayConfig(**self.yaml_data["display"])

    def _load_simulation(self) -> SimulationConfig:
        """
        Load the SimulationConfig from the YAML data.

        returns:
            The YAML data converted to a SimulationConfig dataclass
        """
        # No processing needed for the SimulationConfig
        return SimulationConfig(**self.yaml_data["simulation"])

    def _load_mitigation(self) -> MitigationConfig:
        """
        Load the MitigationConfig from the YAML data.

        returns:
            The YAML data converted to a MitigationConfig dataclass
        """
        # No processing needed for the MitigationConfig
        return MitigationConfig(**self.yaml_data["mitigation"])

    def _load_operational(self) -> OperationalConfig:
        """
        Load the OperationalConfig from the YAML data.

        returns:
            The YAML data converted to an OperationalConfig dataclass
        """
        # No processing needed for the OperationalConfig
        return OperationalConfig(**self.yaml_data["operational"])

    def _load_historical(self) -> HistoricalConfig:
        """
        Load the HistoricalConfig from the YAML data.

        returns:
            The YAML data converted to an HistoricalConfig dataclass
        """
        return HistoricalConfig(**self.yaml_data["historical"])

    def _load_terrain(self) -> TerrainConfig:
        """
        Load the TerrainConfig from the YAML data.

        returns:
            The YAML data converted to a TerrainConfig dataclass
        """
        topo_type = self.yaml_data["terrain"]["topography"]["type"]
        fuel_type = self.yaml_data["terrain"]["fuel"]["type"]

        topo_type, topo_layer, topo_name, topo_kwargs = self._create_topography_layer(
            init=True
        )
        if topo_name is not None and topo_kwargs is not None:
            topo_fn = FunctionalConfig(topo_name, topo_kwargs)
        else:
            topo_fn = None

        fuel_type, fuel_layer, fuel_name, fuel_kwargs = self._create_fuel_layer(init=True)
        if fuel_name is not None and fuel_kwargs is not None:
            fuel_fn = FunctionalConfig(fuel_name, fuel_kwargs)
        else:
            fuel_fn = None

        return TerrainConfig(
            topo_type, topo_layer, fuel_type, fuel_layer, topo_fn, fuel_fn
        )

    def _create_topography_layer(
        self, init: bool = False, seed: Optional[int] = None
    ) -> Tuple[str, TopographyLayer, Optional[str], Optional[Dict[str, Any]]]:
        """
        Create a TopographyLayer given the config parameters.
        This is used for initalization and after resetting the layer seeds.

        Arguments:
            seed: A randomization seed used by

        Returns:
            A tuple containing:
                A string representing the `type` of the layer (`operational`,
                    `functional`, etc.)
                A FunctionalTopographyLayer that utilizes the fuction specified by
                    fn_name and the keyword arguments in kwargs
                The function name if a functional layer is used. Otherwise None
                The keyword arguments for the function if a functinoal layer is used.
                    Otherwise None
        """
        topo_layer: TopographyLayer
        topo_type = self.yaml_data["terrain"]["topography"]["type"]
        if topo_type == "operational":
            if self.lat_long_box is not None:
                topo_layer = OperationalTopographyLayer(self.lat_long_box)
            else:
                raise ConfigError(
                    "The topography layer type is `operational`, "
                    "but self.lat_long_box is None"
                )
            fn_name = None
            kwargs = None
        elif topo_type == "functional":
            fn_name = self.yaml_data["terrain"]["topography"]["functional"]["function"]
            try:
                kwargs = self.yaml_data["terrain"]["topography"]["functional"][fn_name]
            # No kwargs found (flat is an example of this)
            except KeyError:
                kwargs = {}
            # Reset the seed if this isn't the inital creation
            if "seed" in kwargs and not init:
                kwargs["seed"] = seed
            if fn_name == "perlin":
                fn = perlin(**kwargs)
            elif fn_name == "gaussian":
                fn = gaussian(**kwargs)
            elif fn_name == "flat":
                fn = flat()
            else:
                raise ConfigError(
                    f"The specified topography function ({fn_name}) " "is not valid."
                )
            topo_layer = FunctionalTopographyLayer(
                self.yaml_data["area"]["screen_size"],
                self.yaml_data["area"]["screen_size"],
                fn,
                fn_name,
            )
        else:
            raise ConfigError(
                f"The specified topography type ({topo_type}) " "is not supported"
            )

        return topo_type, topo_layer, fn_name, kwargs

    def _create_burn_probability_layer(
        self, init: bool = False, seed: Optional[int] = None
    ) -> Tuple[str, BurnProbabilityLayer, Optional[str], Optional[Dict[str, Any]]]:
        """
        Create a BurnProbabilityLayer given the config parameters.
        This is used for initalization and after resetting the layer seeds.

        Arguments:
            seed: A randomization seed used by

        Returns:
            A tuple containing:
                A string representing the `type` of the layer (`operational`,
                    `functional`, etc.)
                A FunctionalTopographyLayer that utilizes the fuction specified by
                    fn_name and the keyword arguments in kwargs
                The function name if a functional layer is used. Otherwise None
                The keyword arguments for the function if a functinoal layer is used.
                    Otherwise None
        """
        burn_prob_layer: BurnProbabilityLayer
        bp_type = self.yaml_data["terrain"]["burn_probability"]["type"]
        if bp_type == "operational":
            if self.lat_long_box is not None:
                burn_prob_layer = OperationalBurnProbabilityLayer(self.lat_long_box)
            else:
                raise ConfigError(
                    "The burn probability layer type is `operational`, "
                    "but self.lat_long_box is None"
                )
            fn_name = None
            kwargs = None
        elif bp_type == "functional":
            fn_name = self.yaml_data["terrain"]["burn_probability"]["functional"][
                "function"
            ]
            try:
                kwargs = self.yaml_data["terrain"]["burn_probability"]["functional"][
                    fn_name
                ]
            # No kwargs found (flat is an example of this)
            except KeyError:
                kwargs = {}
            # Reset the seed if this isn't the inital creation
            if "seed" in kwargs and not init:
                kwargs["seed"] = seed
            if fn_name == "perlin":
                fn = perlin(**kwargs)
            elif fn_name == "gaussian":
                fn = gaussian(**kwargs)
            elif fn_name == "flat":
                fn = flat()
            else:
                raise ConfigError(
                    f"The specified topography function ({fn_name}) " "is not valid."
                )
            burn_prob_layer = FunctionalBurnProbabilityLayer(
                self.yaml_data["area"]["screen_size"],
                self.yaml_data["area"]["screen_size"],
                fn,
                fn_name,
            )
        else:
            raise ConfigError(
                f"The specified topography type ({bp_type}) " "is not supported"
            )

        return bp_type, burn_prob_layer, fn_name, kwargs

    def _create_fuel_layer(
        self, init: bool = False, seed: Optional[int] = None
    ) -> Tuple[str, FuelLayer, Optional[str], Optional[Dict[str, Any]]]:
        """
        Create a FuelLayer given the config parameters.
        This is used for initalization and after resetting the layer seeds.

        Returns:
            A FunctionalFuelLayer that utilizes the fuction specified by
            fn_name and the keyword arguments in kwargs
        """
        fuel_layer: FuelLayer
        fuel_type = self.yaml_data["terrain"]["fuel"]["type"]
        if fuel_type == "operational":
            if self.lat_long_box is not None:
                fuel_layer = OperationalFuelLayer(self.lat_long_box)
            else:
                raise ConfigError(
                    "The topography layer type is `operational`, "
                    "but self.lat_long_box is None"
                )
            fn_name = None
            kwargs = None
        elif fuel_type == "functional":
            fn_name = self.yaml_data["terrain"]["fuel"]["functional"]["function"]
            try:
                kwargs = self.yaml_data["terrain"]["fuel"]["functional"][fn_name]
            # No kwargs found (some functions don't need input arguments)
            except KeyError:
                kwargs = {}
            # Reset the seed if this isn't the inital creation
            if "seed" in kwargs and not init:
                kwargs["seed"] = seed
            if fn_name == "chaparral":
                fn = chaparral_fn(**kwargs)
            else:
                raise ConfigError(
                    f"The specified fuel function ({fn_name}) " "is not valid."
                )
            fuel_layer = FunctionalFuelLayer(
                self.yaml_data["area"]["screen_size"],
                self.yaml_data["area"]["screen_size"],
                fn,
                fn_name,
            )
        else:
            raise ConfigError(
                f"The specified fuel type ({fuel_type}) " "is not supported"
            )

        return fuel_type, fuel_layer, fn_name, kwargs

    def _load_fire(self, pos: Optional[Tuple[int, int]] = None) -> FireConfig:
        """
        Load the FireConfig from the YAML data.

        Returns:
            The YAML data converted to a FireConfig dataclass
        """
        max_fire_duration = int(self.yaml_data["fire"]["max_fire_duration"])
        fire_init_pos_type = self.yaml_data["fire"]["fire_initial_position"]["type"]
        if fire_init_pos_type == "static":
            # If pos is unspecified, read from the YAML data
            if pos is None:
                fire_pos = self.yaml_data["fire"]["fire_initial_position"]["static"][
                    "position"
                ]
                fire_pos = fire_pos[1:-1].split(",")
                fire_initial_position = (int(fire_pos[0]), int(fire_pos[1]))
            # Pos is specified, so use that
            else:
                fire_initial_position = pos
            return FireConfig(fire_initial_position, max_fire_duration)
        elif fire_init_pos_type == "random":
            if pos is not None:
                log.warn(
                    "`pos` is specified, but the initialization type is `random`. "
                    "Ignoring `pos`."
                )
            screen_size = self.yaml_data["area"]["screen_size"]
            seed = self.yaml_data["fire"]["fire_initial_position"]["random"]["seed"]
            rng = np.random.default_rng(seed)
            pos_x = rng.integers(screen_size, dtype=int)
            pos_y = rng.integers(screen_size, dtype=int)
            return FireConfig((pos_x, pos_y), max_fire_duration, seed)
        else:
            raise ConfigError(
                "The specified fire initial position type "
                f"({fire_init_pos_type}) is not supported"
            )

    def _load_environment(self) -> EnvironmentConfig:
        """
        Load the EnvironmentConfig from the YAML data.

        Returns:
            The YAML data converted to a EnvironmentConfig dataclass
        """
        # No processing needed for the EnvironmentConfig
        return EnvironmentConfig(**self.yaml_data["environment"])

    def _load_wind(self) -> WindConfig:
        """
        Load the WindConfig from the YAML data.

        Returns:
            The YAML data converted to a WindConfig dataclass
        """
        # Only support simple for now
        # TODO: Figure out how Perlin and CFD create wind
        fn_name = self.yaml_data["wind"]["function"]
        if fn_name == "simple":
            arr_shape = (
                self.yaml_data["area"]["screen_size"],
                self.yaml_data["area"]["screen_size"],
            )
            speed = self.yaml_data["wind"]["simple"]["speed"]
            direction = self.yaml_data["wind"]["simple"]["direction"]
            speed_arr = np.full(arr_shape, speed)
            direction_arr = np.full(arr_shape, direction)
            speed_kwargs = None
            dir_kwargs = None
        elif fn_name == "cfd":
            # Check if wind files have been generated
            cfd_generated = os.path.isfile(
                "generated_wind_directions.npy"
            ) and os.path.isfile("generated_wind_magnitudes.npy")
            if cfd_generated is False:
                log.error("Missing pregenerated cfd npy files, switching to perlin")
                self.wind_function = "perlin"
            else:
                speed_arr = np.load("generated_wind_magnitudes.npy")
                direction_arr = np.load("generated_wind_directions.npy")
                speed_arr = scale_ms_to_ftpm(speed_arr)
            speed_kwargs = self.yaml_data["wind"]["cfd"]
            dir_kwargs = self.yaml_data["wind"]["cfd"]
        elif fn_name == "perlin":
            wind_map = WindController()
            speed_kwargs = deepcopy(self.yaml_data["wind"]["perlin"]["speed"])
            range_min = mph_to_ftpm(
                self.yaml_data["wind"]["perlin"]["speed"]["range_min"]
            )
            range_max = mph_to_ftpm(
                self.yaml_data["wind"]["perlin"]["speed"]["range_max"]
            )
            speed_kwargs["range_min"] = range_min
            speed_kwargs["range_max"] = range_max
            wind_map.init_wind_speed_generator(
                **speed_kwargs, screen_size=self.yaml_data["area"]["screen_size"]
            )

            direction_kwargs = self.yaml_data["wind"]["perlin"]["direction"]
            wind_map.init_wind_direction_generator(
                **direction_kwargs, screen_size=self.yaml_data["area"]["screen_size"]
            )
            if wind_map.map_wind_speed is not None:
                speed_arr = wind_map.map_wind_speed
            else:
                raise ConfigError(
                    "The Perlin WindController is specified in the config, "
                    "but returned None for the wind speed"
                )
            if wind_map.map_wind_direction is not None:
                direction_arr = wind_map.map_wind_direction
            else:
                raise ConfigError(
                    "The Perlin WindController is specified in the config, "
                    "but returned None for the wind direction"
                )
            direction_arr = wind_map.map_wind_direction
            speed_kwargs = self.yaml_data["wind"]["perlin"]["speed"]
            dir_kwargs = self.yaml_data["wind"]["perlin"]["direction"]
        else:
            raise ConfigError(f"Wind type {fn_name} is not supported")

        if fn_name is not None and speed_kwargs is not None:
            speed_fn = FunctionalConfig(fn_name, speed_kwargs)
        else:
            speed_fn = None
        if fn_name is not None and dir_kwargs is not None:
            direction_fn = FunctionalConfig(fn_name, dir_kwargs)
        else:
            direction_fn = None

        # Convert to float to get correct type
        speed_arr = speed_arr.astype(np.float64)
        direction_arr = direction_arr.astype(np.float64)

        return WindConfig(speed_arr, direction_arr, speed_fn, direction_fn)

    def _cfd_wind_setup(self) -> WindControllerCFD:
        screen_size: int = self.yaml_data["area"]["screen_size"]
        result_accuracy: int = self.yaml_data["wind"]["cfd"]["result_accuracy"]
        # scale: int = self.yaml_data['wind']['cfd']['scale']
        scale: int = self.yaml_data["area"]["pixel_scale"]
        timestep: float = self.yaml_data["wind"]["cfd"]["timestep_dt"]
        diffusion: float = self.yaml_data["wind"]["cfd"]["diffusion"]
        viscosity: float = self.yaml_data["wind"]["cfd"]["viscosity"]
        terrain_features: np.ndarray = self.terrain.topography_layer.data
        wind_speed: float = self.yaml_data["wind"]["cfd"]["speed"]
        wind_direction: str = self.yaml_data["wind"]["cfd"]["direction"]
        time_to_train = self.yaml_data["wind"]["cfd"]["time_to_train"]

        wind_map = WindControllerCFD(
            screen_size,
            result_accuracy,
            scale,
            timestep,
            diffusion,
            viscosity,
            terrain_features,
            wind_speed,
            wind_direction,
            time_to_train,
        )
        return wind_map

    def reset_terrain(
        self,
        topography_seed: Optional[int] = None,
        topography_type: Optional[str] = None,
        fuel_seed: Optional[int] = None,
        fuel_type: Optional[str] = None,
        location: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Reset the terrain functional generation seeds if using functional data,
        or reset the terrain lat/long location if using operational data.

        Arguments:
            topography_seed: The seed used to randomize functional topography generation
            fuel_seed: The seed used to randomize functional fuel generation
            location: A new center-point for the operational topography and fuel data
        """
        # We want to update the YAML terrain data so that the call to _load_terrain()
        # re-create the layers with the updated parameters

        # Do the location first, as the creation of the LatLongBox depends on it
        if location is not None:
            # Since all operational layers use the LatLongBox, we can update
            # the yaml data and the LatLongBox at the class level
            lat, long = location
            self.yaml_data["operational"]["latitude"] = lat
            self.yaml_data["operational"]["longitude"] = long
            self.lat_long_box, self.historical_layer = self._make_lat_long_box()

        # Can only reset functional topography seeds, since operational is updated
        # via the `location` argument
        if topography_seed is not None:
            # Working with functional data
            if self.terrain.topography_function is not None:
                topo_fn_name = self.terrain.topography_function.name
                self.yaml_data["terrain"]["topography"]["functional"][topo_fn_name][
                    "seed"
                ] = topography_seed
        # Can only reset functional fuel seeds, since operational is updated
        # via the `location` argument
        if fuel_seed is not None:
            # Working with functional data
            if self.terrain.fuel_function is not None:
                fuel_fn_name = self.terrain.fuel_function.name
                self.yaml_data["terrain"]["fuel"]["functional"][fuel_fn_name][
                    "seed"
                ] = fuel_seed

        # Need to check if any data layer types are changing, since the
        # screen_size could be affected
        if topography_type is not None and fuel_type is not None:
            # Special case when going from all operational to all functional, so
            # we need to revert back to the original screen_size from the config file
            if topography_type == "operational" and fuel_type == "operational":
                if (
                    self.terrain.topography_type == "functional"
                    and self.terrain.fuel_type == "functional"
                ):
                    self.yaml_data["screen_size"] = self.original_screen_size
        if topography_type is not None:
            # Update the yaml data
            self.yaml_data["terrain"]["topography"]["type"] = topography_type
        if fuel_type is not None:
            # Update the yaml data
            self.yaml_data["terrain"]["fuel"]["type"] = fuel_type

        # Remake the LatLongBox
        self.lat_long_box, self.historical_layer = self._make_lat_long_box()
        # Remake the AreaConfig since operational/functional could have changed
        self.area = self._load_area()
        # Remake the terrain
        self.terrain = self._load_terrain()

    def reset_wind(
        self, speed_seed: Optional[int] = None, direction_seed: Optional[int] = None
    ) -> None:
        """
        Reset the wind speed and direction seeds.

        Arguments:
            speed_seed: The seed used to randomize wind speed generation
            direction_seed: The seed used to randomize wind direction generation
        """
        # We want to update the YAML wind data so that the call to _load_wind()
        # re-create the WindConfig with the updated parameters
        if speed_seed is not None:
            # Working with functional data
            if self.wind.speed_function is not None:
                speed_fn_name = self.wind.speed_function.name
                if "seed" in self.yaml_data["wind"][speed_fn_name]["speed"]:
                    self.yaml_data["wind"][speed_fn_name]["speed"]["seed"] = speed_seed
                else:
                    log.warn(
                        "Attempted to reset speed seed for wind fucntion "
                        f"{speed_fn_name}, but no seed parameter exists in the config"
                    )

        if direction_seed is not None:
            if self.wind.direction_function is not None:
                direction_fn_name = self.wind.direction_function.name
                if "seed" in self.yaml_data["wind"][direction_fn_name]["direction"]:
                    self.yaml_data["wind"][direction_fn_name]["direction"][
                        "seed"
                    ] = direction_seed
                else:
                    log.warn(
                        "Attempted to reset direction seed for wind fucntion "
                        f"{direction_fn_name}, but no seed parameter exists in the "
                        "config"
                    )

        self.wind = self._load_wind()

    def reset_fire(
        self, seed: Optional[int] = None, pos: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Reset the fire initial position seed. Note that both `seed` and `pos` cannot
        be specified together since `seed` is used for random/dynamic cases and `pos`
        is used for static cases.

        Arguments:
            seed: The seed used to randomize fire initial position generation.
            pos: The static position to start the fire at
        """
        if seed is None and pos is None:
            raise ValueError("Both `seed` and `pos` cannot be None")
        elif seed is not None and pos is None:
            try:
                # Change the seed for the current fire initital position type
                fire_init_pos_type = self.yaml_data["fire"]["fire_initial_position"][
                    "type"
                ]
                self.yaml_data["fire"]["fire_initial_position"][fire_init_pos_type][
                    "seed"
                ] = seed
                # Reload the FireConfig with the updated seed in the yaml data
                self.fire = self._load_fire()
            except KeyError:
                log.warning(
                    "Trying to set a seed for fire initial position type "
                    f"({fire_init_pos_type}), which does not support the use of a "
                    "seed. The seed value will be ignored."
                )
        elif seed is None and pos is not None:
            self.fire = self._load_fire(pos=pos)
        else:
            raise ValueError("Both `seed` and `pos` cannot be specified together")

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the current config to the specified path.

        Arguments:
            path: The path and filename of the output YAML file
        """
        with open(path, "w") as f:
            yaml.dump(self.yaml_data, f)
