import json
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import h5py
import jsonlines
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
from ..game.sprites import Agent, Terrain
from ..utils.config import Config
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
        self.start_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

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
        self.agent_positions: np.ndarray
        self.agents: Dict[int, Agent] = {}
        self.reset()

    def reset(self) -> None:
        """
        Reset the `self.fire_map`, `self.terrain`, `self.fire_manager`,
        and all mitigations to initial conditions
        """
        self._create_fire_map()
        self._create_agent_positions()
        self._create_terrain()
        self._create_fire()
        self._create_mitigations()
        self.elapsed_steps = 0

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
        if True:
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
            "w_0": w_0.astype(np.float32),
            "sigma": sigma.astype(np.uint32),
            "delta": delta.astype(np.float32),
            "M_x": M_x.astype(np.float32),
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

    def update_agent_positions(self, points: Iterable[Tuple[int, int, int]]) -> None:
        """
        Update the `self.agent_positions` with new agent positions

        Arguments:
            points: A list of `(column, row, agent_id)` tuples. These will be added to
                    `self.agent_positions`.
        """
        for column, row, agent_id in points:
            # Resets current agent positions to 0 before updating the new positions
            self.agent_positions[self.agent_positions == agent_id] = 0
            self.agent_positions[row][column] = agent_id
            try:
                self.agents[agent_id].pos = (column, row)
            except KeyError:
                self.agents[agent_id] = Agent(
                    (column, row),
                    size=self.config.display.agent_size,
                    headless=self.config.simulation.headless,
                )

    def run(self, time: Union[str, int]) -> Tuple[np.ndarray, bool]:
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

        Returns:
            A tuple of the following:
                - The Burned/Unburned/ControlLine pixel map (`self.fire_map`). Values
                  range from [0, 6] (see simfire/enums.py:BurnStatus).
                - A boolean indicating whether the simulation has reached the end.
        """
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

        while self.fire_status == GameStatus.RUNNING and num_updates < total_updates:
            self.fire_sprites = self.fire_manager.sprites
            self.fire_map, self.fire_status = self.fire_manager.update(self.fire_map)
            if self._rendering:
                self._render()
            num_updates += 1

            # elapsed_time is in minutes
            self.elapsed_time = self.fire_manager.elapsed_time

            # elapsed steps
            self.elapsed_steps += 1

            # If we're saving data, make sure to append the data to the output JSON after
            # each update
            if self.config.simulation.save_data:
                self._save_data()

        self.active = True if self.fire_status == GameStatus.RUNNING else False

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

    def _create_agent_positions(self) -> None:
        """
        Resets the `self.agent_positions` attribute to entirely `0`
        """
        self.agent_positions = np.zeros_like(self.fire_map)

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

    def set_fire_initial_position(self, pos: Tuple[int, int]) -> None:
        """
        Manually set the fire intial position for a static fire.

        Arguments:
            pos: The (x, y) coordinates to start the fire at
        """
        self.config.reset_fire(pos=pos)

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
        success = False
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

    def save_gif(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Saves the most recent simulation as a GIF.

        Will save a GIF of all calls to `run` from the last time `self.rendering` was set
        to `True`.
        """
        if path is None:
            path = self._create_out_path()

        # Convert to a Path object for easy manipulation
        if not isinstance(path, Path) and path is not None:
            path = Path(path).expanduser()

        # If the path does not end in a filename, create a default filename
        if path.suffix == "":
            now = datetime.now().strftime("%H-%M-%S")
            filename = f"simulation_{now}.gif"
            path = path / filename

        # If the parent path does not exist, create it
        if not path.parent.is_dir() and not path.parent.exists():
            log.info(f"Creating directory '{path.parent}'")
            path.parent.mkdir(parents=True)

        log.info(f"Saving GIF to '{path}'...")

        if path.suffix != ".gif":
            path = path.with_suffix(".gif")
        self._game.save(path, duration=100)  # 0.1s
        log.info("Finished saving GIF")

    def save_spread_graph(self, filename: Optional[Union[str, Path]] = None) -> None:
        """
        Saves the most recent simulation as a PNG.

        Will save a PNG of the spread graph from the last time `self.rendering` was set
        to `True`.
        """
        # Convert to a Path object for easy manipulation
        if not isinstance(filename, Path) and filename is not None:
            filename = Path(filename)
        out_path = self._create_out_path()
        log.info("Saving fire spread graph...")
        # Create the fire_spread_graph and save it to PNG
        if filename is None:
            now = datetime.now().strftime("%H-%M-%S")
            filename = f"fire_spread_graph_{now}.png"
        else:
            if filename.suffix != ".png":
                filename = filename.with_suffix(".png")
        fig_out_path = out_path / filename
        fig = self.fire_manager.draw_spread_graph(self._game.screen)
        fig.savefig(fig_out_path)
        log.info("Done saving fire spread graph")

    def _save_data(self) -> None:
        """
        Save the data into a JSON file.
        """
        # Create the output path if it doesn't exist
        out_path = self._create_out_path()
        # Create the data directory if it doesn't exist
        datapath = out_path / "data"
        datapath.mkdir(parents=True, exist_ok=True)

        # Get the filepath, depending on the data type
        dtype = self.config.simulation.data_type
        if dtype == "npy":
            ext = "npy"
        elif dtype == "h5":
            ext = "h5"
        elif dtype in ["json", "jsonl"]:
            ext = "jsonl"
        else:
            raise ValueError(
                f"Invalid data type '{dtype}' given. Valid types are 'npy', 'h5', "
                f"'json', and 'jsonl'."
            )
        fire_map_path = datapath / f"fire_map.{ext}"

        # Binarize the static data layers
        static = self._load_static_data(datapath)

        # Create the metadata
        metadata = {
            "config": self.config.yaml_data,
            "seeds": self.get_seeds(),
            "layer_types": self.get_layer_types(),
            "shape": static["shape"],
            "static_data": static,
            "fire_map": fire_map_path.name,
        }

        # Save the metadata
        with open(datapath / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Load the previous fire map if it's already been saved
        # This would be the case
        if dtype in ["npy", "h5"]:
            loaded_fire_map = self._load_fire_map(fire_map_path)

            # Load the current fire map
            current_fire_map = self.fire_map
            current_fire_map = np.expand_dims(current_fire_map, axis=0)

            # Append to the loaded fire map
            if loaded_fire_map is not None:
                if len(loaded_fire_map.shape) == 2:
                    loaded_fire_map = np.expand_dims(loaded_fire_map, axis=0)
                fire_map = np.append(loaded_fire_map, current_fire_map, axis=0)
            else:
                fire_map = current_fire_map
        # If we're saving JSON Lines files, we don't need to load the previous fire map
        else:
            fire_map = self.fire_map

        # Save the fire map data
        if dtype == "npy":
            np.save(fire_map_path, fire_map.astype(np.int8))
        elif dtype == "h5":
            with h5py.File(fire_map_path, "w") as f:
                f.create_dataset("data", data=fire_map)
        elif dtype in ["json", "jsonl"]:
            # Convert the fire map to a list of lists
            fire_map = fire_map.tolist()
            with jsonlines.open(fire_map_path, "a") as writer:
                writer.write({self.elapsed_steps: fire_map})

    @property
    def rendering(self) -> bool:
        """
        Returns whether or not the simulator is currently rendering.

        Returns:
            Whether or not the simulator is currently rendering.
        """
        return self._rendering

    @rendering.setter
    def rendering(self, value: bool) -> None:
        """
        Sets whether or not the simulator is currently rendering.

        Arguments:
            value: Whether or not the simulator is currently rendering.
        """
        self._rendering = value
        if value:
            # Create the Game and switch the internal variable to track if we're
            # currently rendering
            self._game = Game(
                (self.config.area.screen_size, self.config.area.screen_size),
                record=True,
            )
        else:
            self._game.quit()

    def _render(self) -> None:
        """
        Render `self._game` frame with `self._game.update`
        """
        agent_sprites = list(self.agents.values())
        self._game.update(
            self.terrain,
            self.fire_sprites,
            self.fireline_sprites,
            agent_sprites,
            self.config.wind.speed,
            self.config.wind.direction,
        )
        self._game.fire_map = self.fire_map
        self._last_screen = self._game.screen

    def _create_out_path(self) -> Path:
        """
        Creates the output path if it does not exist.
        """
        out_path = Path(self.config.simulation.save_path).expanduser()
        if not out_path.parent.is_dir():
            log.warning(
                "Designated save path from the config does not exist, "
                "creating parent directories"
            )
            parents = True
        else:
            parents = False

        if not out_path.is_dir():
            log.info(f"Creating directory '{out_path}'")
            out_path.mkdir(parents=parents) if not out_path.is_dir() else None
        return out_path

    def _load_fire_map(self, filepath: Path) -> Optional[np.ndarray]:
        """
        Load the fire map from the data directory.

        Arguments:
            filepath: The path to the fire map in the data directory.

        Returns:
            The fire map if it exists, otherwise None.
        """
        # Check if the file exists, if not return None
        if not filepath.is_file():
            return None

        # Load the fire map, depending on the data type
        if filepath.suffix == ".npy":
            fire_map = np.load(filepath)
        else:
            fire_map = h5py.File(filepath)["data"]

        # Make sure the fire map is a numpy array
        fire_map = np.array(fire_map)
        return fire_map

    def _load_static_data(self, datapath: Path) -> Dict[str, Any]:
        """
        Load the static data from `self.get_attribute_data` and save it to the
        data directory if it does not exist.

        Arguments:
            datapath: The path to the data directory.

        Returns:
            The static data.
        """
        # Get the static data
        data = self.get_attribute_data()

        # Create the data locations based on the keys
        data_locs = {k: "" for k in data.keys()}

        # Get the shape of the data (can get it from any key)
        shape = data[list(data.keys())[0]].shape

        # Create the data locations based on the data type
        for key in data.keys():
            if self.config.simulation.data_type == "npy":
                filename = f"{key}.npy"
            elif self.config.simulation.data_type == "h5":
                filename = f"{key}.h5"
            elif self.config.simulation.data_type in ["json", "jsonl"]:
                filename = f"{key}.json"
            else:
                raise ValueError(
                    f"Invalid data type '{self.config.simulation.data_type}' given. "
                    "Valid types are 'npy', 'h5', 'json', and 'jsonl'."
                )
            data_locs[key] = filename

        # Save the static data if it does not exist
        for key, loc in data_locs.items():
            path = datapath / loc
            if not path.is_file():
                log.info(f"Creating static data file '{path}'")
                if self.config.simulation.data_type == "npy":
                    np.save(path, data[key])
                elif self.config.simulation.data_type == "h5":
                    with h5py.File(path, "w") as f:
                        f.create_dataset("data", data=data[key])
                else:
                    with open(path, "w") as f:
                        json.dump({"data": data[key].tolist()}, f)

        static_dict = {"data": data_locs, "shape": shape}
        return static_dict
