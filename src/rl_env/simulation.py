import numpy as np

from typing import Dict, Optional
from abc import ABC, abstractmethod

from ..game.game import Game
from ..utils.config import Config
from ..game.sprites import Terrain
from ..utils.log import create_logger
from ..enums import GameStatus, BurnStatus
from ..game.managers.fire import RothermelFireManager
from ..world.parameters import Environment, FuelParticle
from ..game.managers.mitigation import (FireLineManager, ScratchLineManager,
                                        WetLineManager)

log = create_logger(__name__)


class Simulation(ABC):
    '''
    Base class with several built in methods for interacting with different simulators.

    Current simulators using this API:
      - [RothermelSimulator](https://gitlab.mitre.org/fireline/rothermel-modeling)
    '''
    def __init__(self, config: Config) -> None:
        '''
        Initialize the Simulation object for interacting with the RL harness.

        Arguments:
            config: The `Config` that specifies simulation parameters, read in from a
                    YAML file.
        '''
        self.config = config

    @abstractmethod
    def run(self) -> np.ndarray:
        '''
        Runs the simulation
        '''
        pass

    @abstractmethod
    def render(self) -> None:
        '''
        Runs the simulation and displays the simulation's and agent's actions.
        '''
        pass

    @abstractmethod
    def get_actions(self) -> Dict[str, int]:
        '''
        Returns the action space for the simulation.

        Returns:
            The action / mitgiation strategies available: Dict[str, int]
        '''
        pass

    @abstractmethod
    def get_attributes(self) -> Dict[str, np.ndarray]:
        '''
        Initialize and return the observation space for the simulation.

        Arguments:
            None

        Returns:
            The dictionary of observations containing NumPy arrays.
        '''
        pass

    @abstractmethod
    def get_seeds(self) -> Dict[str, int]:
        '''
        Returns the available randomization seeds for the simulation.

        Arguments:
            None

        Returns:
            The dictionary with all available seeds to change and their values.
        '''
        pass

    @abstractmethod
    def set_seeds(self, seeds: Dict[str, int]) -> None:
        '''
        Sets the seeds for different available randomization parameters.

        Which randomization parameters can be  set depends on the simulator being used.
        Available seeds can be retreived by calling the `self.get_seeds` method.

        Arguments:
            seeds: The dictionary of seed names and their current seed values.

        Returns:
            None
        '''
        pass


class RothermelSimulation(Simulation):
    def __init__(self, config: Config) -> None:
        '''
        Initialize the `RothermelSimulation` object for interacting with the RL harness.
        '''
        super().__init__(config)
        self.game_status = GameStatus.RUNNING
        self.fire_status = GameStatus.RUNNING
        self.points = set([])
        self._create_terrain()
        self._create_fire()
        self._create_mitigations()

    def _create_terrain(self) -> None:
        '''
        Initialize the terrain.
        '''
        self.fuel_particle = FuelParticle()
        self.fuel_arrs = [[
            self.config.terrain.fuel_array_function(x, y)
            for x in range(self.config.area.terrain_size)
        ] for y in range(self.config.area.terrain_size)]
        self.terrain = Terrain(self.fuel_arrs,
                               self.config.terrain.elevation_function,
                               self.config.area.terrain_size,
                               self.config.area.screen_size,
                               headless=self.config.simulation.headless)

        self.environment = Environment(self.config.environment.moisture,
                                       self.config.wind.speed, self.config.wind.direction)

    def _create_mitigations(self) -> None:
        '''
        Initialize the mitigation strategies.
        '''
        # initialize all mitigation strategies
        self.fireline_manager = FireLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain,
            headless=self.config.simulation.headless)

        self.scratchline_manager = ScratchLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain,
            headless=self.config.simulation.headless)

        self.wetline_manager = WetLineManager(size=self.config.display.control_line_size,
                                              pixel_scale=self.config.area.pixel_scale,
                                              terrain=self.terrain,
                                              headless=self.config.simulation.headless)

        self.fireline_sprites = self.fireline_manager.sprites
        self.fireline_sprites_empty = self.fireline_sprites.copy()
        self.scratchline_sprites = self.scratchline_manager.sprites
        self.wetline_sprites = self.wetline_manager.sprites

    def _create_fire(self) -> None:
        '''
        This function will initialize the rothermel fire strategies.
        '''
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
            headless=self.config.simulation.headless)
        self.fire_sprites = self.fire_manager.sprites

    def get_actions(self) -> Dict[str, int]:
        '''
        Return the action space for the Rothermel simulation.

        Arguments:
            None

        Returns:
            The action / mitgiation strategies available: Dict[str, int]
        '''
        return {
            'none': BurnStatus.UNBURNED,
            'fireline': BurnStatus.FIRELINE,
            'scratchline': BurnStatus.SCRATCHLINE,
            'wetline': BurnStatus.WETLINE
        }

    def get_attributes(self) -> Dict[str, np.ndarray]:
        '''
        Initialize and return the observation space for the simulation.

        Arguments:
            None

        Returns:
            The dictionary of observations containing NumPy arrays.
        '''
        return {
            'w0':
            np.array([[
                self.terrain.fuel_arrs[i][j].fuel.w_0
                for j in range(self.config.area.screen_size)
            ] for i in range(self.config.area.screen_size)]),
            'sigma':
            np.array([[
                self.terrain.fuel_arrs[i][j].fuel.sigma
                for j in range(self.config.area.screen_size)
            ] for i in range(self.config.area.screen_size)]),
            'delta':
            np.array([[
                self.terrain.fuel_arrs[i][j].fuel.delta
                for j in range(self.config.area.screen_size)
            ] for i in range(self.config.area.screen_size)]),
            'M_x':
            np.array([[
                self.terrain.fuel_arrs[i][j].fuel.M_x
                for j in range(self.config.area.screen_size)
            ] for i in range(self.config.area.screen_size)]),
            'elevation':
            self.terrain.elevations,
            'wind_speed':
            self.config.wind.speed,
            'wind_direction':
            self.config.wind.direction
        }

    def _correct_pos(self, position: np.ndarray) -> np.ndarray:
        '''
        '''
        pos = position.flatten()
        current_pos = np.where(pos == 1)[0]
        prev_pos = current_pos - 1
        pos[prev_pos] = 1
        pos[current_pos] = 0
        position = np.reshape(
            pos, (self.config.area.screen_size, self.config.area.screen_size))

        return position

    def _update_sprite_points(
        self,
        mitigation_state: np.ndarray,
        position_state: Optional[np.ndarray] = ([0], [0])
    ) -> None:
        '''
        Update sprite point list based on fire mitigation.

        Arguments:
            mitigation_state: Array of mitigation value(s). 0: No Control Line,
                              1: Control Line
            position_state: Array of position. Only used when rendering `inline`
            inline: Boolean value of whether or not to render at each step() or after
                    agent has placed control lines. If True, will use mitigation state,
                    position_state to add a new point to the fireline sprites group.
                    If False, loop through all mitigation_state array to get points to add
                    to fireline sprites group.

        Returns:
            None
        '''
        if self.config.render.inline:
            if mitigation_state[0] == BurnStatus.FIRELINE:
                self.points.add((position_state[0][0], position_state[1][0]))
            elif mitigation_state[0] == BurnStatus.SCRATCHLINE:
                self.points.add((position_state[0][0], position_state[1][0]))
            elif mitigation_state[0] == BurnStatus.WETLINE:
                self.points.add((position_state[0][0], position_state[1][0]))

        else:
            # update the location to pass to the sprite
            for i in range(self.config.area.screen_size):
                for j in range(self.config.area.screen_size):
                    if mitigation_state[(i, j)] == BurnStatus.FIRELINE:
                        self.points.add((i, j))
                    elif mitigation_state[(i, j)] == BurnStatus.SCRATCHLINE:
                        self.points.add((i, j))
                    elif mitigation_state[(i, j)] == BurnStatus.WETLINE:
                        self.points.add((i, j))

    def run(self, mitigation_state: np.ndarray, mitigation: bool) -> np.ndarray:
        '''
        Runs the simulation with or without mitigation lines.

        Use `self.terrain` to either:

          1. Place agent's mitigation lines and then spread fire
          2. Only spread fire, with no mitigation line
                (to compare for reward calculation)

        Arguments:
            mitigation_state: Array of mitigation value(s) as BurnStatus values.

            mitigation: Boolean for running the mitigation + fire spread which updates
                        `FirelineManager` sprites points or just the fire spread.


        Returns:
            The Burned/Unburned/ControlLine pixel map. Values range from [0, 6].
        '''
        # for updating sprite purposes turn the inline rendering "off"
        self.config.render.inline = False

        # reset the fire status to running
        self.fire_status = GameStatus.RUNNING
        # initialize fire strategy
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
            headless=self.config.simulation.headless)

        self.fire_map = np.full(
            (self.config.area.screen_size, self.config.area.screen_size),
            BurnStatus.UNBURNED)
        if mitigation:
            # update firemap with agent actions before initializing fire spread
            self._update_sprite_points(mitigation_state)
            self.fire_map = self.fireline_manager.update(self.fire_map, self.points)

        while self.fire_status == GameStatus.RUNNING:
            self.fire_sprites = self.fire_manager.sprites
            self.fire_map, self.fire_status = self.fire_manager.update(self.fire_map)
            if self.fire_status == GameStatus.QUIT:
                return self.fire_map

    def _render_inline(self, mitigation: np.ndarray, position: np.ndarray) -> None:
        '''
        Interact with the RL harness to display and update the simulation as the agent
        progresses through the simulation (if applicable, i.e AgentBasedHarness)

        TODO: position could change to a Dict[str, (int, int)] for multi-agent scenario

        Arguments:
            mitigation: The values of the mitigation array from the RL Harness, converted
                        to the simulation format.
            position: The position array of the agent.

        Returns:
            None
            '''
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
            headless=False)
        self.game = Game(self.config.area.screen_size)
        self.fire_map = self.game.fire_map

        position = np.where(self._correct_pos(position) == 1)
        mitigation = mitigation[position].astype(int)
        # mitigation map needs to be associated with correct BurnStatus types

        self._update_sprite_points(mitigation, position)
        if self.game_status == GameStatus.RUNNING:

            self.fire_map = self.fireline_manager.update(self.fire_map, self.points)
            self.fireline_sprites = self.fireline_manager.sprites
            self.game.fire_map = self.fire_map
            self.game_status = self.game.update(self.terrain, self.fire_sprites,
                                                self.fireline_sprites,
                                                self.config.wind.speed,
                                                self.config.wind.direction)

            self.fire_map = self.game.fire_map
            self.game.fire_map = self.fire_map
        self.points = set([])

    def _render_mitigations(self, mitigation: np.ndarray) -> None:
        '''
        This method will render the agent's actions after the final action.

        NOTE: this method doesn't seem really necessary -- might omit and use
        `_render_mitigation_fire_spread()` only

        Arguments:
            mitigation: The array of the final mitigation state from the RL Harness.

        Returns:
            None
        '''
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
            headless=False)
        self.game = Game(self.config.area.screen_size, headless=False)
        self.fire_map = self.game.fire_map

        self._update_sprite_points(mitigation)
        if self.game_status == GameStatus.RUNNING:

            self.fire_map = self.fireline_manager.update(self.fire_map, self.points)
            self.fireline_sprites = self.fireline_manager.sprites
            self.game.fire_map = self.fire_map
            self.game_status = self.game.update(self.terrain, self.fire_sprites,
                                                self.fireline_sprites,
                                                self.config.wind.speed,
                                                self.config.wind.direction)

            self.fire_map = self.game.fire_map
            self.game.fire_map = self.fire_map
        self.points = set([])

    def _render_mitigation_fire_spread(self, mitigation: np.ndarray) -> None:
        '''
        Render the agent's actions after the final action and the subsequent fire spread.

        Arguments:
            mitigation: The array of the final mitigation state from the RL Harness.

        Returns:
            None
        '''
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
            headless=False)
        self.game = Game(self.config.area.screen_size, headless=False)
        self.fire_map = self.game.fire_map

        self.fire_status = GameStatus.RUNNING
        self.game_status = GameStatus.RUNNING
        self._update_sprite_points(mitigation)
        self.fireline_sprites = self.fireline_manager.sprites
        self.fire_map = self.fireline_manager.update(self.fire_map, self.points)
        while self.game_status == GameStatus.RUNNING and \
                self.fire_status == GameStatus.RUNNING:
            self.fire_sprites = self.fire_manager.sprites
            self.game.fire_map = self.fire_map
            self.game_status = self.game.update(self.terrain, self.fire_sprites,
                                                self.fireline_sprites,
                                                self.config.wind.speed,
                                                self.config.wind.direction)
            self.fire_map, self.fire_status = self.fire_manager.update(self.fire_map)
            self.fire_map = self.game.fire_map
            self.game.fire_map = self.fire_map

        self.points = set([])

    def render(self, type: str, mitigation: np.ndarray,
               position: np.ndarray = ([0], [0])) -> None:
        '''
        This is a helper function that hands off to sub-functions for rendering.

        Arguments:
            type: The type of rendering being done
                  ('inline', 'post_agent', 'post agent with fire')
            mitigation: The values of the mitigation array from the RL Harness, converted
                        to the simulation format.
            position: The position array of the agent.

        Returns:
            None
        '''
        if type == 'inline':
            self._render_inline(mitigation, position)
        if type == 'post agent':
            self._render_mitigations(mitigation)
        if type == 'post agent with fire':
            self._render_mitigation_fire_spread(mitigation)

    def get_seeds(self) -> Dict[str, int]:
        '''
        Returns the available randomization seeds for the simulation.

        Arguments:
            None

        Returns:
            The dictionary with all available seeds to change and their values.
        '''
        seeds = {
            'elevation': self._get_elevation_seed(),
            'fuel': self._get_fuel_seed(),
            'wind_speed': self._get_wind_speed_seed(),
            'wind_direction': self._get_wind_direction_seed()
        }
        # Make sure to delete all the seeds that are None, so the user knows not to try
        # and set them
        del_keys = []
        for key, seed in seeds.items():
            if seed is None:
                del_keys.append(key)
        for key in del_keys:
            del seeds[key]

        return seeds

    def _get_elevation_seed(self) -> int:
        '''
        Returns the seed for the current elevation function.

        Only the 'perlin' option has a seed value associated with it.

        Arguments:
            None

        Returns:
            The seed for the currently configured elevation function.
        '''
        if 'perlin' in str(self.config.terrain.elevation_function).lower():
            return self.config.terrain.perlin.seed
        else:
            return None

    def _get_fuel_seed(self) -> int:
        '''
        Returns the seed for the current fuel array function.

        Only the 'chaparral' option has a seed value associated with it, because it's
        currently the only one.

        Arguments:
            None

        Returns:
            The seed for the currently configured fuel array function.
        '''
        if 'chaparral' in str(self.config.terrain.fuel_array_function).lower():
            return self.config.terrain.chaparral.seed
        else:
            return None

    def _get_wind_speed_seed(self) -> int:
        '''
        Returns the seed for the current wind speed function.

        Only the 'perlin' option has a seed value associated with it.

        Arguments:
            None

        Returns:
            The seed for the currently configured wind speed function.
        '''
        if 'perlin' in str(self.config.wind.wind_function).lower():
            return self.config.wind.perlin.speed.seed
        else:
            return None

    def _get_wind_direction_seed(self) -> int:
        '''
        Returns the seed for the current wind direction function.

        Only the 'perlin' option has a seed value associated with it.

        Arguments:
            None

        Returns:
            The seed for the currently configured wind direction function.
        '''
        if 'perlin' in str(self.config.wind.wind_function).lower():
            return self.config.wind.perlin.direction.seed
        else:
            return None

    def set_seeds(self, seeds: Dict[str, int]) -> bool:
        '''
        Sets the seeds for different available randomization parameters.

        Which randomization parameters can be  set depends on the simulator being used.
        Available seeds can be retreived by calling the `self.get_seeds` method.

        Arguments:
            seeds: The dictionary of seed names and the values they will be set to.

        Returns:
            Whether or not the method successfully set a seed value
        '''
        success = False
        keys = list(seeds.keys())
        if 'elevation' in keys:
            self.config.reset_elevation_function(seeds['elevation'])
            success = True
        if 'fuel' in keys:
            self.config.reset_fuel_array_function(seeds['fuel'])
            success = True
        if 'wind_speed' in keys and 'wind_direction' in keys:
            self.config.reset_wind_function(speed_seed=seeds['wind_speed'],
                                            direction_seed=seeds['wind_direction'])
            success = True
        if 'wind_speed' in keys and 'wind_direction' not in keys:
            self.config.reset_wind_function(speed_seed=seeds['wind_speed'])
            success = True
        if 'wind_speed' not in keys and 'wind_direction' in keys:
            self.config.reset_wind_function(direction_seed=seeds['wind_direction'])
            success = True

        valid_keys = list(self.get_seeds().keys())
        for key in keys:
            if key not in valid_keys:
                log.warn('No valid keys in the seeds dictionary were given to the '
                         'set_seeds method. No seeds will be changed. Valid keys are: '
                         f'{valid_keys}')
        return success
