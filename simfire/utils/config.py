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

from ..utils.generate_cfd_wind_layer import generate_cfd_wind_layer
from ..world.elevation_functions import flat, gaussian, perlin
from ..world.fuel_array_functions import chaparral_fn
from ..world.wind_mechanics.wind_controller import WindController, WindControllerCFD
from .layers import (
    BurnProbabilityLayer,
    FuelLayer,
    FunctionalBurnProbabilityLayer,
    FunctionalFuelLayer,
    FunctionalTopographyLayer,
    HistoricalLayer,
    LandFireLatLongBox,
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
    screen_size: Tuple[int, int]
    pixel_scale: float

    def __post_init__(self) -> None:
        self.screen_size = (int(self.screen_size[0]), int(self.screen_size[1]))
        self.pixel_scale = float(self.pixel_scale)


@dataclasses.dataclass
class DisplayConfig:
    fire_size: int
    control_line_size: int
    agent_size: int
    rescale_factor: Optional[int] = None

    def __post_init__(self) -> None:
        self.fire_size = int(self.fire_size)
        self.control_line_size = int(self.control_line_size)
        self.agent_size = int(self.agent_size)
        if self.rescale_factor is not None:
            try:
                self.rescale_factor = int(self.rescale_factor)
            except ValueError:
                if isinstance(self.rescale_factor, str):
                    if self.rescale_factor.upper() == "NONE":
                        self.rescale_factor = None
                    else:
                        raise ValueError(
                            f"Specified value  of {self.rescale_factor} for "
                            "config:display:rescale_factor is not valid. "
                            "Specify either an integer value or None"
                        )
                else:
                    raise TypeError(
                        "Speicified type of config:display:rescale_factor "
                        f"({type(self.rescale_factor)}) with value "
                        f"{self.rescale_factor} is invalid. rescale_factor "
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
        sf_home: str,
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
        self.sf_home = Path(sf_home)


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
    year: int  # TODO: Make enum for year?

    def __post_init__(self) -> None:
        self.latitude = float(self.latitude)
        self.longitude = float(self.longitude)
        self.height = float(self.height)
        self.width = float(self.width)
        self.resolution = float(self.resolution)
        self.year = int(self.year)


@dataclasses.dataclass
class HistoricalConfig:
    path: Union[Path, str]
    year: int
    state: str
    fire: str
    height: int
    width: int


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
    diagonal_spread: bool
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
    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        cfd_precompute: bool = False,
    ) -> None:
        if path is not None:
            if isinstance(path, str):
                path = Path(path)
        self.path = path
        if config_dict is None and path is not None:
            self.yaml_data = self._load_yaml()
        elif config_dict is not None and path is None:
            self.yaml_data = config_dict
        else:
            raise ValueError("Either a path or a config dictionary must be specified.")

        # Save the original screen size in case the simulation changes from
        # operational to functional
        self.original_screen_size = self.yaml_data["area"]["screen_size"]

        # If using 'historical', BOTH topo and fuel must be 'historical'!
        # The only use case I can imagine is using 'functional' for one and
        # 'historical' for the other. I don't see how that would be useful,
        # so we will force FULL usage of historical data, when specified. Otherwise,
        # raise an error!
        topo_type = self.yaml_data["terrain"]["topography"]["type"]
        fuel_type = self.yaml_data["terrain"]["fuel"]["type"]

        if topo_type == "historical" and fuel_type != "historical":
            log.error(f"Invalid config: historical topography, but {fuel_type} fuel.")
            raise ConfigError(
                "If using 'historical' data for topography type, the fuel type must "
                "also be 'historical'!"
            )
        elif fuel_type == "historical" and topo_type != "historical":
            log.error(f"Invalid config: historical fuel, but {topo_type} topography.")
            raise ConfigError(
                "If using 'historical' data for fuel type, the topography type must "
                "also be 'historical'!"
            )

        # Load the historical data layers, if necessary.
        if topo_type == "historical" and fuel_type == "historical":
            self.historical = self._load_historical()
            self.historical_layer = self._create_historical_layer()

        # This can take up to 30 seconds to pull LandFire data directly from source
        self.landfire_lat_long_box = self._make_lat_long_box()

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
            if self.path is not None:
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

    def _make_lat_long_box(self) -> Optional[LandFireLatLongBox]:
        """
        Optionally create the LatLongBox used by the data layers if any
        of the data layers are of type `operational`.

        Returns:
            None if none of the data layers are `operational`
            LatLongBox with the config coordinates and shape if any of
            the data layers are `operational`
        """

        # temporarily instantiate LatLongBox for the BurnProbabiltyLayer
        # TODO: remove this layer until LandFire supports the data
        self.lat_long_box = LatLongBox()

        if (
            self.yaml_data["terrain"]["topography"]["type"] == "operational"
            or self.yaml_data["terrain"]["fuel"]["type"] == "operational"
        ):
            year = self.yaml_data["operational"]["year"]
            self._set_all_combos()
            if self.yaml_data["operational"]["seed"] is not None:
                points: Tuple[Tuple[float, float], Tuple[float, float]] = (
                    self._randomly_select_box(self.yaml_data["operational"]["seed"])
                )
                valid = self._check_lat_long(points)
                if not valid:
                    if self.landfire_lat_long_box is None:
                        message = (
                            "Lat/Long box is not valid and was not created successfully."
                        )
                        log.error(message)
                        raise ConfigError(message)
                    message = (
                        "Latitude and Longitude box is not valid. Data does not "
                        f"exist within the bounding box: ({points[0]}), "
                        f" ({points[1]}) and the year {year}."
                        "Checking another location "
                        f"with seed {self.yaml_data['operational']['seed'] + 1}."
                    )
                    log.warning(message)
                    self.yaml_data["operational"]["seed"] += 1
                    landfire_lat_long_box = self._make_lat_long_box()
            else:
                tl_lat = self.yaml_data["operational"]["latitude"]
                tl_lon = self.yaml_data["operational"]["longitude"]
                height = self.yaml_data["operational"]["height"]
                width = self.yaml_data["operational"]["width"]
                # calculate the bottom right corner
                br_lat = tl_lat - ((height / 30) * 0.00027777777803598015)
                br_long = tl_lon + ((width / 30) * 0.00027777777803598015)
                valid = self._check_lat_long(((tl_lat, tl_lon), (br_lat, br_long)))
                if not valid:
                    message = (
                        "Lat/Long box is not valid. Data does not "
                        "exist between the bounding box: "
                        f"({(tl_lat, tl_lon), (br_lat, br_long) }), "
                        f"and the year {year}."
                    )
                    log.error(message)
                    raise ConfigError(message)
                else:
                    landfire_lat_long_box = LandFireLatLongBox(
                        points=((tl_lat, tl_lon), (br_lat, br_long)),
                        year=year,
                        height=height,
                        width=width,
                    )
            return landfire_lat_long_box
        elif (
            self.yaml_data["terrain"]["topography"]["type"] == "historical"
            or self.yaml_data["terrain"]["fuel"]["type"] == "historical"
        ):
            return self.historical_layer.lat_lon_box
        else:
            return None

    def _check_lat_long(
        self, points: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> bool:
        """
        Check that the points going to be queried are within the bounds of
            the CONUS data.

        CONUS extent:
                West_Bounding_Coordinate: -127.9878
                East_Bounding_Coordinate: -65.2544
                North_Bounding_Coordinate: 51.6497
                South_Bounding_Coordinate: 22.7654

        Args:
            points: The bounding box that will be passed into LandFireLatLongBox

        Returns:
            True if the LandFireLatLongBox is within the bounds of the data
        """
        TLW = -127.9878  # west bounding coord
        BRW = -65.2544  # east bounding coord
        TLN = 51.6497  # north bounding coord
        BRN = 22.7654  # south bounding coord

        tlw = points[0][1]
        brw = points[1][1]
        tln = points[0][0]
        brn = points[1][0]
        # If top-left inner box corner is inside the bounding box
        if TLN > tln and TLW < tlw:
            # If bottom-right inner box corner is inside the bounding box
            if BRN < brn and BRW > brw:
                return True
            else:
                return False
        else:
            return False

    def _set_all_combos(self) -> None:
        """
        Get all possible combinations of latitude and longitude for the
        data layers.

        TODO: This needs to get re-vsisted since we no longer use the tiles
        CONUS extent:
                West_Bounding_Coordinate: -127.9878
                East_Bounding_Coordinate: -65.2544
                North_Bounding_Coordinate: 51.6497
                South_Bounding_Coordinate: 22.7654
        """

        res = str(self.yaml_data["operational"]["resolution"]) + "m"
        year = str(self.yaml_data["operational"]["year"])
        if res not in ["30m"]:
            message = "Resolution must be 30m"
            log.error(message)
            raise ConfigError(message)
        if year not in ["2019", "2020", "2022"]:
            message = "Year must be 2019, 2020, or 2022"
            log.error(message)
            raise ConfigError(message)

        # create a random point within CONUS bounds
        y = random.choice(np.linspace(-127.9878, -65.2544, 100000))  # nosec
        x = random.choice(np.linspace(22.7654, 51.6497, 100000))  # nosec
        self._all_combos = (x, y)

    def _randomly_select_box(
        self, seed: int
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Randomly select a latitude and longitude for the LandFireLatLongBox.

        Args:
            seed: The seed to use for the random number generator

        Returns:
            A LandFireLatLongBox with a random latitude and longitude
        """
        random.seed(seed)  # nosec
        lat, lon = self._all_combos  # nosec
        tl_lat = round(random.random(), 4) + lat  # nosec
        tl_lon = round(random.random(), 4) + lon  # nosec
        height = self.yaml_data["operational"]["height"]
        width = self.yaml_data["operational"]["width"]

        # calculate the bottom right corner
        br_lat = tl_lat - ((height / 30) * 0.00027777777803598015)
        br_long = tl_lon + ((width / 30) * 0.00027777777803598015)

        return ((tl_lat, tl_lon), (br_lat, br_long))

    def _load_area(self) -> AreaConfig:
        """
        Load the AreaConfig from the YAML data.

        This is in pixel space, to convert to actual physical space multiple the
            height/width by the resolution: 0.000277..., which represents 30m
            per decimal degree of the lat/lon input.

        returns:
            The YAML data converted to an AreaConfig dataclass
        """

        # No processing needed for the AreaConfig
        if self.landfire_lat_long_box is not None:
            # Overwite the screen_size since operational data will determine
            self.yaml_data["area"]["screen_size"] = (
                self.landfire_lat_long_box.fuel.shape[0],
                self.landfire_lat_long_box.fuel.shape[1],
            )
            self.yaml_data["area"]["pixel_scale"] = int(
                self.yaml_data["operational"]["resolution"] / 0.3048
            )
            # "Clear" the geotiff_data to enable making deepcopy of Config object
            self.landfire_lat_long_box.geotiff_data = None
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
            if self.landfire_lat_long_box is not None:
                topo_layer = OperationalTopographyLayer(self.landfire_lat_long_box)
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
                self.yaml_data["area"]["screen_size"][0],
                self.yaml_data["area"]["screen_size"][1],
                fn,
                fn_name,
            )
        elif topo_type == "historical":
            topo_layer = self.historical_layer.topography
            fn_name = None
            kwargs = None
        else:
            raise ConfigError(
                f"The specified topography type ({topo_type}) " "is not supported"
            )

        return topo_type, topo_layer, fn_name, kwargs

    def _create_burn_probability_layer(
        self, init: bool = False, seed: Optional[int] = None
    ) -> Tuple[
        str, Optional[BurnProbabilityLayer], Optional[str], Optional[Dict[str, Any]]
    ]:
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
        burn_prob_layer: Optional[BurnProbabilityLayer]
        bp_type = self.yaml_data["terrain"]["burn_probability"]["type"]
        if bp_type == "operational":
            if self.lat_long_box is not None:
                path = Path(self.yaml_data["operational"]["path"])
                burn_prob_layer = OperationalBurnProbabilityLayer(self.lat_long_box, path)
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
                self.yaml_data["area"]["screen_size"][0],
                self.yaml_data["area"]["screen_size"][1],
                fn,
                fn_name,
            )
        elif bp_type == "historical":
            burn_prob_layer = None
            fn_name = None
            kwargs = None
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
            if self.landfire_lat_long_box is not None:
                fuel_layer = OperationalFuelLayer(self.landfire_lat_long_box)
            else:
                raise ConfigError(
                    "The fuel layer type is `operational`, "
                    "but self.landfire_lat_long_box is None"
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
                self.yaml_data["area"]["screen_size"][0],
                self.yaml_data["area"]["screen_size"][1],
                fn,
                fn_name,
            )
        elif fuel_type == "historical":
            fuel_layer = self.historical_layer.fuel
            fn_name = None
            kwargs = None
        else:
            raise ConfigError(
                f"The specified fuel type ({fuel_type}) " "is not supported"
            )

        return fuel_type, fuel_layer, fn_name, kwargs

    def _create_historical_layer(self):
        """Create a HistoricalLayer given the config parameters.
        This is an optional dataclass.

        Returns:
            A HistoricalLayer that utilizes the data specified in the config.
        """
        historical_layer = HistoricalLayer(
            self.historical.year,
            self.historical.state,
            self.historical.fire,
            self.historical.path,
            self.historical.height,
            self.historical.width,
        )
        return historical_layer

    def _load_fire(self, pos: Optional[Tuple[int, int]] = None) -> FireConfig:
        """
        Load the FireConfig from the YAML data.

        Returns:
            The YAML data converted to a FireConfig dataclass
        """
        max_fire_duration = int(self.yaml_data["fire"]["max_fire_duration"])
        diagonal_spread = bool(self.yaml_data["fire"]["diagonal_spread"])
        fire_init_pos_type = self.yaml_data["fire"]["fire_initial_position"]["type"]
        if fire_init_pos_type == "static":
            # If pos is unspecified, read from the YAML data
            if pos is None:
                fire_pos = self.yaml_data["fire"]["fire_initial_position"]["static"][
                    "position"
                ]
                if isinstance(fire_pos, str):
                    fire_pos = fire_pos[1:-1].split(",")
                if len(fire_pos) > 2:
                    raise ConfigError(
                        "`fire_initial_position` should only be a Tuple of length 2"
                    )
                fire_initial_position = (int(fire_pos[0]), int(fire_pos[1]))
            # Pos is specified, so use that
            else:
                fire_initial_position = pos
            return FireConfig(fire_initial_position, diagonal_spread, max_fire_duration)
        elif fire_init_pos_type == "random":
            if pos is not None:
                log.warn(
                    "`pos` is specified, but the initialization type is `random`. "
                    "Ignoring `pos`."
                )
            screen_size = self.yaml_data["area"]["screen_size"]
            seed = self.yaml_data["fire"]["fire_initial_position"]["random"]["seed"]
            rng = np.random.default_rng(seed)
            pos_x = rng.integers(screen_size[1], dtype=int)
            pos_y = rng.integers(screen_size[0], dtype=int)
            return FireConfig((pos_x, pos_y), diagonal_spread, max_fire_duration, seed)
        elif fire_init_pos_type == "historical":
            return FireConfig(
                (self.historical_layer.fire_start_y, self.historical_layer.fire_start_x),
                diagonal_spread,
                max_fire_duration,
                None,
            )
        else:
            raise ConfigError(
                "The specified fire initial position type "
                f"({fire_init_pos_type}) is not supported"
            )

    def _load_historical(self) -> HistoricalConfig:
        """Load the HistoricalConfig from the YAML data.

        Returns:
            The YAML data converted to a HistoricalConfig dataclass
        """
        return HistoricalConfig(**self.yaml_data["historical"])

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
                self.yaml_data["area"]["screen_size"][0],
                self.yaml_data["area"]["screen_size"][1],
            )
            speed = mph_to_ftpm(self.yaml_data["wind"]["simple"]["speed"])
            direction = self.yaml_data["wind"]["simple"]["direction"]
            speed_arr = np.full(arr_shape, speed)
            direction_arr = np.full(arr_shape, direction)
            speed_kwargs = None
            dir_kwargs = None
        elif fn_name == "cfd":
            # Check if wind files have been generated
            cfd_generated = os.path.isfile(
                "pregenerated_wind_files/generated_wind_directions.npy"
            ) and os.path.isfile("pregenerated_wind_files/generated_wind_magnitudes.npy")
            if cfd_generated is False:
                log.info("Generating CFD wind data")
                time_to_train = self.yaml_data["wind"]["cfd"]["time_to_train"]
                cfd_setup = WindControllerCFD(
                    self.yaml_data["area"]["screen_size"],
                    self.yaml_data["wind"]["cfd"]["result_accuracy"],
                    self.yaml_data["wind"]["cfd"]["scale"],
                    self.yaml_data["wind"]["cfd"]["timestep_dt"],
                    self.yaml_data["wind"]["cfd"]["diffusion"],
                    self.yaml_data["wind"]["cfd"]["viscosity"],
                    self.terrain.topography_layer.data,
                    self.yaml_data["wind"]["cfd"]["speed"],
                    self.yaml_data["wind"]["cfd"]["direction"],
                    time_to_train,
                )
                generate_cfd_wind_layer(time_to_train, cfd_setup)
            speed_arr = np.load("pregenerated_wind_files/generated_wind_magnitudes.npy")
            direction_arr = np.load(
                "pregenerated_wind_files/generated_wind_directions.npy"
            )
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
        screen_size: tuple[int, int] = self.yaml_data["area"]["screen_size"]
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
            screen_size=screen_size,
            result_accuracy=result_accuracy,
            scale=scale,
            timestep=timestep,
            diffusion=diffusion,
            viscosity=viscosity,
            terrain_features=terrain_features,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            time_to_train=time_to_train,
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

        # Do the location first, as the creation of the LandFireLatLongBox depends on it
        if location is not None:
            # Since all operational layers use the LandFireLatLongBox, we can update
            # the yaml data and the LandFireLatLongBox at the class level
            lat, long = location
            self.yaml_data["operational"]["latitude"] = lat
            self.yaml_data["operational"]["longitude"] = long
            self.landfire_lat_long_box = self._make_lat_long_box()

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

        # Remake the LandFireLatLongBox
        self.landfire_lat_long_box = self._make_lat_long_box()
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
        fire_init_pos_type = self.yaml_data["fire"]["fire_initial_position"]["type"]

        if seed is None and pos is None:
            raise ValueError("Both `seed` and `pos` cannot be None")
        elif seed is not None and pos is None:
            try:
                # Change the seed for the current fire initital position type
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
            try:
                # For consistency, ensure YAML data contains the same position value.
                self.yaml_data["fire"]["fire_initial_position"][fire_init_pos_type][
                    "position"
                ] = pos
                # Reload the FireConfig with the updated position in the yaml data
                self.fire = self._load_fire(pos=pos)
            except KeyError:
                log.warning(
                    "Trying to set a position for fire initial position type "
                    f"({fire_init_pos_type}), which does not support the use of a "
                    "position. The position value will be ignored."
                )
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
