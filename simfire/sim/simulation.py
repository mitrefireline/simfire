import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ..enums import (
    BurnStatus,
    ElevationConstants,
    FuelConstants,
    GameStatus,
    WindConstants,
)
from ..game.game import Game
from ..game.managers.fire import RothermelFireManager
from ..game.managers.mitigation import (
    FireLineManager,
    ScratchLineManager,
    WetLineManager,
)
from ..game.sprites import Terrain
from ..utils.config import Config
from ..utils.layers import FunctionalFuelLayer
from ..utils.log import create_logger
from ..utils.units import str_to_minutes
from ..world.parameters import Environment, FuelParticle

log = create_logger(__name__)


class Simulation(ABC):
    """
    Base class with several built in methods for interacting with different simulators.

    Current simulators using this API:
      - `SimFire <https://gitlab.mitre.org/fireline/simfire>`_
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the Simulation object for interacting with the RL harness.

        Arguments:
            config: The `Config` that specifies simulation parameters, read in from a
                    YAML file.
        """
        self.config = config
        # Create a _now time to use for the simulation object. This is used to
        # create folders based on individual simulation runs.
        self._now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    @abstractmethod
    def run(self, time: Union[str, int]) -> Tuple[np.ndarray, bool]:
        """
        Runs the simulation.

        Arguments:
            time: Either how many updates to run the simulation, based on the config
                  value, `config.simulation.update_rate`, or a length of time expressed
                  as a string (e.g. `120m`, `2h`, `2hour`, `2hours`, `1h 60m`, etc.)
        Returns:
            A tuple of the following:
                - The Burned/Unburned/ControlLine pixel map (`self.fire_map`). Values
                  range from [0, 6] (see simfire/enums.py:BurnStatus).
                - A boolean indicating whether the simulation has reached the end.
        """
        pass

    @abstractmethod
    def get_actions(self) -> Dict[str, int]:
        """
        Returns the action space for the simulation.

        Returns:
            The action / mitgiation strategies available: Dict[str, int]
        """
        pass

    @abstractmethod
    def get_attribute_data(self) -> Dict[str, np.ndarray]:
        """
        Initialize and return the observation space for the simulation.

        Returns:
            The dictionary of observations containing NumPy arrays.
        """
        pass

    @abstractmethod
    def get_attribute_bounds(self) -> Dict[str, object]:
        """
        Initialize and return the observation space bounds for the simulation.

        Returns:
            The dictionary of observation space bounds containing NumPy arrays.
        """
        pass

    @abstractmethod
    def get_seeds(self) -> Dict[str, Optional[int]]:
        """
        Returns the available randomization seeds for the simulation.

        Returns:
            The dictionary with all available seeds to change and their values.
        """
        pass

    @abstractmethod
    def set_seeds(self, seeds: Dict[str, int]) -> bool:
        """
        Sets the seeds for different available randomization parameters.

        Which randomization parameters can be  set depends on the simulator being used.
        Available seeds can be retreived by calling the `self.get_seeds` method.

        Arguments:
            seeds: The dictionary of seed names and their current seed values.

        Returns:
            Whether or not the method successfully set a seed value.
        """
        pass

    @abstractmethod
    def update_mitigation(self, points: Iterable[Tuple[int, int, int]]) -> None:
        """
        Update the `self.fire_map` with new mitigation points

        Arguments:
            points: A list of `(x, y, mitigation)` tuples. These will be added to
                   `self.fire_map`.
        """
        pass

    @abstractmethod
    def load_mitigation(self, mitigation_map: np.ndarray) -> None:
        """
        Set the 'self.fire_map' to the new mitigation map

        Arguments:
            mitigation_map: A numpy array of mitigations to be set as 'self.fire_map'
        """
        pass

    def get_disaster_categories(self) -> Dict[str, int]:
        """
        Returns all possible categories that a location in the map can be in.

        Returns:
            A dictionary of enum name to enum value.
        """
        return {i.name: i.value for i in self.disaster_categories}

    @property
    @abstractmethod
    def disaster_categories(self) -> Iterable[IntEnum]:
        """
        Returns the possible categories that a location in the map can be in.

        Returns:
            An enum of possible categories.
        """
        pass


class FireSimulation(Simulation):
    def __init__(self, config: Config) -> None:
        """
        Initialize the `FireSimulation` object for interacting with the RL harness.

        Arguments:
            config: The `Config` that specifies simulation parameters, read in from a
                    YAML file.
        """
        super().__init__(config)
        self._rendering: bool = False
        self.game_status: GameStatus = GameStatus.RUNNING
        self.fire_status: GameStatus = GameStatus.RUNNING
        self.fire_map: np.ndarray
        self.reset()

    def reset(self) -> None:
        """
        Reset the `self.fire_map`, `self.terrain`, `self.fire_manager`,
        and all mitigations to initial conditions
        """
        self._create_fire_map()
        self._create_terrain()
        self._create_fire()
        self._create_mitigations()

    def _create_terrain(self) -> None:
        """
        Initialize the terrain.
        """
        self.fuel_particle = FuelParticle()

        self.terrain = Terrain(
            self.config.terrain.fuel_layer,
            self.config.terrain.topography_layer,
            (self.config.area.screen_size, self.config.area.screen_size),
            headless=self.config.simulation.headless,
        )

        self.environment = Environment(
            self.config.environment.moisture,
            self.config.wind.speed,
            self.config.wind.direction,
        )

    def _create_mitigations(self) -> None:
        """
        Initialize the mitigation strategies.
        """
        # initialize all mitigation strategies
        self.fireline_manager = FireLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain,
            headless=self.config.simulation.headless,
        )

        self.scratchline_manager = ScratchLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain,
            headless=self.config.simulation.headless,
        )

        self.wetline_manager = WetLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain,
            headless=self.config.simulation.headless,
        )

        self.fireline_sprites = self.fireline_manager.sprites
        self.fireline_sprites_empty = self.fireline_sprites.copy()
        self.scratchline_sprites = self.scratchline_manager.sprites
        self.wetline_sprites = self.wetline_manager.sprites

    def _create_fire(self) -> None:
        """
        This function will initialize the fire strategies.
        """
        self.fire_manager = RothermelFireManager(
            self.config.fire.fire_initial_position,
            self.config.display.fire_size,
            self.config.fire.max_fire_duration,
            self.config.area.pixel_scale,
            self.config.simulation.update_rate,
            self.fuel_particle,
            self.terrain,
            self.environment,
            max_time=self.config.simulation.runtime,
            attenuate_line_ros=self.config.mitigation.ros_attenuation,
            headless=self.config.simulation.headless,
        )
        self.fire_sprites = self.fire_manager.sprites

    def get_actions(self) -> Dict[str, int]:
        """
        Return the action space for the fire simulation.

        Returns:
            The action / mitigation strategies available: Dict[str, int]
        """
        return {
            "fireline": BurnStatus.FIRELINE,
            "scratchline": BurnStatus.SCRATCHLINE,
            "wetline": BurnStatus.WETLINE,
        }

    @property
    def disaster_categories(self) -> Iterable[BurnStatus]:
        """
        Returns all possible categories that a location in the map can be in.

        Returns:
            A dictionary of enum name to enum value.
        """
        return BurnStatus

    def get_attribute_bounds(self) -> Dict[str, object]:
        """
        Return the observation space bounds for the fire simulation

        Returns:
            The dictionary of observation space bounds containing NumPy arrays.
        """
        bounds = {}
        if isinstance(self.terrain.fuel_layer, FunctionalFuelLayer):
            fuel_bounds = {
                "w_0": {"min": FuelConstants.W_0_MIN, "max": FuelConstants.W_0_MAX},
                "sigma": {"min": FuelConstants.SIGMA, "max": FuelConstants.SIGMA},
                "delta": {"min": FuelConstants.DELTA, "max": FuelConstants.DELTA},
                "M_x": {"min": FuelConstants.M_X, "max": FuelConstants.M_X},
            }
            bounds.update(fuel_bounds)
        else:
            log.error("Fuel layer type not yet supported")
            raise NotImplementedError

        elevation_bounds = {
            "elevation": {
                "min": ElevationConstants.MIN_ELEVATION,
                "max": ElevationConstants.MAX_ELEVATION,
            }
        }

        bounds.update(elevation_bounds)

        wind_dir_min = 0.0
        wind_dir_max = 360.0

        wind_bounds = {
            "wind": {
                "speed": {"min": WindConstants.MIN_SPEED, "max": WindConstants.MAX_SPEED},
                "direction": {"min": wind_dir_min, "max": wind_dir_max},
            }
        }
        bounds.update(wind_bounds)
        return bounds

    def get_attribute_data(self) -> Dict[str, np.ndarray]:
        """
        Initialize and return the observation space for the simulation.

        Returns:
            The dictionary of observation data containing NumPy arrays.
        """
        w_0 = np.zeros_like(self.terrain.fuels)
        sigma = np.zeros_like(self.terrain.fuels)
        delta = np.zeros_like(self.terrain.fuels)
        M_x = np.zeros_like(self.terrain.fuels)
        for y in range(self.config.area.screen_size):
            for x in range(self.config.area.screen_size):
                fuel = self.terrain.fuels[y][x]
                w_0[y][x] = fuel.w_0
                sigma[y][x] = fuel.sigma
                delta[y][x] = fuel.delta
                M_x[y][x] = fuel.M_x

        return {
            "w_0": w_0,
            "sigma": sigma,
            "delta": delta,
            "M_x": M_x,
            "elevation": self.terrain.elevations,
            "wind_speed": self.config.wind.speed,
            "wind_direction": self.config.wind.direction,
        }

    def _correct_pos(self, position: np.ndarray) -> np.ndarray:
        """
        Correct the position to be the same shape as
        `(self.config.area.screen_size, self.config.area.screen_size)`

        Arguments:
            position: The position to be corrected.

        Returns:
            The corrected position.
        """
        pos = position.flatten()
        current_pos = np.where(pos == 1)[0]
        prev_pos = current_pos - 1
        pos[prev_pos] = 1
        pos[current_pos] = 0
        position = np.reshape(
            pos, (self.config.area.screen_size, self.config.area.screen_size)
        )

        return position

    def load_mitigation(self, mitigation_map: np.ndarray) -> None:
        """
        Set the 'self.fire_map' to the new mitigation map

        Arguments:
            mitigation_map: A numpy array of mitigations to be set as 'self.fire_map'
        """
        category_values = [status.value for status in BurnStatus]

        if np.isin(mitigation_map, category_values).all():
            message = (
                "You are overwriting the current fire map with the given "
                "mitigation map - the current fire map data will be erased."
            )
            self.fire_map = mitigation_map
        else:
            message = (
                f"Invalid values in {mitigation_map} - values need to be "
                f"within {category_values}... Skipping"
            )

        warnings.warn(message)
        log.warning(message)

    def update_mitigation(self, points: Iterable[Tuple[int, int, int]]) -> None:
        """
        Update the `self.fire_map` with new mitigation points

        Arguments:
            points: A list of `(column, row, mitigation)` tuples. These will be added to
                   `self.fire_map`.
        """
        firelines = []
        scratchlines = []
        wetlines = []

        # Loop through all points, and add the mitigations to their respective lists
        for i, (column, row, mitigation) in enumerate(points):
            if mitigation == BurnStatus.FIRELINE:
                firelines.append((column, row))
            elif mitigation == BurnStatus.SCRATCHLINE:
                scratchlines.append((column, row))
            elif mitigation == BurnStatus.WETLINE:
                wetlines.append((column, row))
            else:
                log.warning(
                    f"The mitigation,{mitigation}, provided at location[{i}] is "
                    "not an available mitigation strategy... Skipping"
                )

        # Update the self.fire_map using the managers
        self.fire_map = self.fireline_manager.update(self.fire_map, firelines)
        self.fire_map = self.scratchline_manager.update(self.fire_map, scratchlines)
        self.fire_map = self.wetline_manager.update(self.fire_map, wetlines)

    def run(
        self,
        time: Union[str, int],
        render: bool = False,
        record: bool = False,
        spread_graph: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Runs the simulation with or without mitigation lines.

        Use `self.terrain` to either:

          1. Place agent's mitigation lines and then spread fire
          2. Only spread fire, with no mitigation line
                (to compare for reward calculation)

        Arguments:
            time: Either how many updates to run the simulation, based on the config
                  value, `config.simulation.update_rate`, or a length of time expressed
                  as a string (e.g. `120m`, `2h`, `2hour`, `2hours`, `1h 60m`, etc.)
            render: Whether or not to render the simulation during the current `run`.
            record: Whether or not to record the simulation's run in a GIF. Will be output
                    to the location specified in the config section:
                    `simulation.save_path`
                    (`docs <fireline.pages.mitre.org/simfire/config.html#save_path>`_)
            spread_graph: Whether or not to save the spread graph of the simulation.

        Returns:
            A tuple of the following:
                - The Burned/Unburned/ControlLine pixel map (`self.fire_map`). Values
                  range from [0, 6] (see simfire/enums.py:BurnStatus).
                - A boolean indicating whether the simulation has reached the end.
        """
        if not render and (record or spread_graph):
            log.warning(
                "Cannot record a simulation without rendering it: the simulation "
                "will not create a GIF recording or a spread graph"
            )
        # reset the fire status to running
        self.fire_status = GameStatus.RUNNING

        if isinstance(time, str):
            # Convert the string to a number of minutes
            time = str_to_minutes(time)
            # Then determine how many times to step through the loop
            total_updates = round(time / self.config.simulation.update_rate)
        elif isinstance(time, int):
            total_updates = time

        num_updates = 0
        self.elapsed_time = self.fire_manager.elapsed_time

        # Create the Game and switch the internal variable to track if we're
        # currently rendering
        if render:
            self._game = Game(
                (self.config.area.screen_size, self.config.area.screen_size),
                record=record,
            )
            self._rendering = True

        while self.fire_status == GameStatus.RUNNING and num_updates < total_updates:
            self.fire_sprites = self.fire_manager.sprites
            self.fire_map, self.fire_status = self.fire_manager.update(self.fire_map)
            if render:
                self._render()
            num_updates += 1
            # elapsed_time is in minutes
            self.elapsed_time = self.fire_manager.elapsed_time

        self.active = True if self.fire_status == GameStatus.RUNNING else False

        if render:
            if record:
                self._save_gif()
            if spread_graph:
                self._save_spread_graph()
            self._rendering = False
            self._game.quit()

        # Save the spread graph

        return self.fire_map, self.active

    def _create_fire_map(self) -> None:
        """
        Resets the `self.fire_map` attribute to entirely `BurnStatus.UNBURNED`,
        except for self.config.fire.fire_initial_position, which is set to
        `BurnStatus.BURNING`.
        """
        self.fire_map = np.full(
            (self.config.area.screen_size, self.config.area.screen_size),
            BurnStatus.UNBURNED,
        )
        x, y = self.config.fire.fire_initial_position
        self.fire_map[y, x] = BurnStatus.BURNING

    def get_seeds(self) -> Dict[str, Optional[int]]:
        """
        Returns the available randomization seeds for the simulation.

        Returns:
            The dictionary with all available seeds to change and their values.
        """
        seeds = {
            "elevation": self._get_topography_seed(),
            "fuel": self._get_fuel_seed(),
            "wind_speed": self._get_wind_speed_seed(),
            "wind_direction": self._get_wind_direction_seed(),
            "fire_initial_position": self._get_fire_initial_position_seed(),
        }
        # Make sure to delete all the seeds that are None, so the user knows not to try
        # and set them
        del_keys: List[str] = []
        for key, seed in seeds.items():
            if seed is None:
                del_keys.append(key)
        for key in del_keys:
            del seeds[key]

        return seeds

    def _get_topography_seed(self) -> Optional[int]:
        """
        Returns the seed for the current elevation function.

        Only the 'perlin' option has a seed value associated with it.

        Returns:
            The seed for the currently configured elevation function.
        """
        if self.config.terrain.topography_type == "functional":
            if self.config.terrain.topography_function is not None:
                if self.config.terrain.topography_function.name == "perlin":
                    return self.config.terrain.topography_function.kwargs["seed"]
                elif self.config.terrain.topography_function.name == "flat":
                    return None
                else:
                    raise RuntimeError(
                        f"The topography function name "
                        f"{self.config.terrain.topography_function.name} "
                        "is not valid"
                    )
            else:
                raise RuntimeError(
                    "The topography type is set as functional, but "
                    "self.config.terrain.topography_function is not set"
                )
        elif self.config.terrain.topography_type == "operational":
            return self.config.operational.seed
        else:
            raise RuntimeError(
                f"The value of {self.config.terrain.topography_type} "
                "for self.config.terrain.topography_type is not valid"
            )

    def _get_fuel_seed(self) -> Optional[int]:
        """
        Returns the seed for the current fuel array function.

        Only the 'chaparral' option has a seed value associated with it, because it's
        currently the only one.

        Returns:
            The seed for the currently configured fuel array function.
        """
        if self.config.terrain.fuel_type == "functional":
            if self.config.terrain.fuel_function is not None:
                if self.config.terrain.fuel_function.name == "chaparral":
                    return self.config.terrain.fuel_function.kwargs["seed"]
                else:
                    raise RuntimeError(
                        "The fuel function name "
                        f"{self.config.terrain.fuel_function.name} is "
                        "not valid"
                    )
            else:
                raise RuntimeError(
                    "The fuel type is set as functional, but "
                    "self.config.terrain.fuel_function is not set"
                )
        elif self.config.terrain.fuel_type == "operational":
            return self.config.operational.seed
        else:
            raise RuntimeError(
                f"The value of {self.config.terrain.fuel_type} "
                "for self.config.terrain.fuel_type is not valid"
            )

    def _get_wind_speed_seed(self) -> Optional[int]:
        """
        Returns the seed for the current wind speed function.

        Only the 'perlin' option has a seed value associated with it.

        Returns:
            The seed for the currently configured wind speed function.
        """
        if self.config.wind.speed_function is not None:
            if self.config.wind.speed_function.name == "perlin":
                return self.config.wind.speed_function.kwargs["seed"]
            else:
                return None
        else:
            return None

    def _get_wind_direction_seed(self) -> Optional[int]:
        """
        Returns the seed for the current wind direction function.

        Only the 'perlin' option has a seed value associated with it.

        Returns:
            The seed for the currently configured wind direction function.
        """
        if self.config.wind.direction_function is not None:
            if self.config.wind.direction_function.name == "perlin":
                return self.config.wind.direction_function.kwargs["seed"]
            else:
                return None
        else:
            return None

    def _get_fire_initial_position_seed(self) -> Optional[int]:
        """
        Returns the seed for the current fire start location.

        Only the 'random' option has a seed value associated with it.

        Returns:
            The seed for the currently configured fire start location.
        """
        # The seed is set to None for static start locations
        # The seed is set to an int value for random start locations
        return self.config.fire.seed

    def set_seeds(self, seeds: Dict[str, int]) -> bool:
        """
        Sets the seeds for different available randomization parameters.

        Which randomization parameters can be  set depends on the simulator being used.
        Available seeds can be retreived by calling the `self.get_seeds` method.

        Arguments:
            seeds: The dictionary of seed names and the values they will be set to.

        Returns:
            Whether or not the method successfully set a seed value.
        """
        success = False
        keys = list(seeds.keys())
        if "elevation" in keys:
            self.config.reset_terrain(topography_seed=seeds["elevation"])
            success = True
        if "fuel" in keys:
            self.config.reset_terrain(fuel_seed=seeds["fuel"])
            success = True
        if "wind_speed" in keys and "wind_direction" in keys:
            self.config.reset_wind(
                speed_seed=seeds["wind_speed"], direction_seed=seeds["wind_direction"]
            )
            success = True
        if "wind_speed" in keys and "wind_direction" not in keys:
            self.config.reset_wind(speed_seed=seeds["wind_speed"])
            success = True
        if "wind_direction" in keys and "wind_speed" not in keys:
            self.config.reset_wind(direction_seed=seeds["wind_direction"])
            success = True
        if "fire_initial_position" in keys:
            self.config.reset_fire(seeds["fire_initial_position"])

        valid_keys = list(self.get_seeds().keys())
        for key in keys:
            if key not in valid_keys:
                message = (
                    "No valid keys in the seeds dictionary were given to the "
                    "set_seeds method. No seeds will be changed. Valid keys are: "
                    f"{valid_keys}"
                )
                log.warning(message)
                warnings.warn(message)
                success = False
        return success

    def get_layer_types(self) -> Dict[str, str]:
        """
        Returns the current layer types for the simulation

        Returns:
            A dictionary of the current layer type.
        """
        types = {
            "elevation": self.config.terrain.topography_type,
            "fuel": self.config.terrain.fuel_type,
        }

        return types

    def set_layer_types(self, types: Dict[str, str]) -> bool:
        """
        Set the type of layers to be used in the simulation

        Available keys are 'elevation' and 'fuel' and available values are 'functional'
        and 'operational'.

        Arguments:
            types: The dictionary of layer names and the data type they will be set to.

        Returns:
            Whether or not the method successfully set a data type.
        """
        keys = list(types.keys())
        if "elevation" in keys:
            self.config.reset_terrain(topography_type=types["elevation"])
            success = True
        if "fuel" in keys:
            self.config.reset_terrain(fuel_type=types["fuel"])
            success = True

        valid_keys = list(self.get_layer_types().keys())
        for key in keys:
            if key not in valid_keys:
                message = (
                    "No valid keys in the types dictionary were given to the "
                    "set_data_types method. No data types will be changed. Valid "
                    f"keys are: {valid_keys}"
                )
                log.warning(message)
                warnings.warn(message)
                success = False

        if success:
            # all keys are valid
            self.config.reset_terrain(
                topography_type=types["elevation"], fuel_type=types["fuel"]
            )

        return success

    def _render(self) -> None:
        """
        Render `self._game` frame with `self._game.update`
        """
        self._game.update(
            self.terrain,
            self.fire_sprites,
            self.fireline_sprites,
            self.config.wind.speed,
            self.config.wind.direction,
        )
        self._game.fire_map = self.fire_map

    def _create_out_path(self) -> Path:
        """
        Creates the output path if it does not exist.
        """
        out_path = Path(self.config.simulation.save_path).expanduser() / self._now
        if not out_path.parent.is_dir():
            log.warning(
                "Designated save path from the config does not exist, "
                "creating parent directories"
            )
            parents = True
        else:
            parents = False

        log.info(f"Creating directory {out_path}")
        out_path.mkdir(parents=parents) if not out_path.is_dir() else None
        return out_path

    def _save_gif(self) -> None:
        """
        Save the GIF from `self._game`
        """
        out_path = self._create_out_path()
        log.info("Saving GIF...")
        # Save the GIF created by self._game
        gif_out_path = out_path / f"simulation_{datetime.now().strftime('%H-%M-%S')}.gif"
        self._game.save(gif_out_path, duration=100)  # 0.1s
        log.info("Finished saving GIF")

    def _save_spread_graph(self) -> None:
        """
        Save the fire spread graph from `self._game`
        """
        out_path = self._create_out_path()
        log.info("Saving fire spread graph...")
        # Create the fire_spread_graph and save it to PNG
        fig_out_path = (
            out_path / f"fire_spread_graph_{datetime.now().strftime('%H-%M-%S')}.png"
        )
        fig = self.fire_manager.draw_spread_graph(self._game.screen)
        fig.savefig(fig_out_path)
        log.info("Done saving fire spread graph")
