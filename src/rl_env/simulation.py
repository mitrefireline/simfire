import numpy as np

from typing import Dict, Optional
from abc import ABC, abstractmethod

from ..game.game import Game
from ..utils.config import Config
from ..game.sprites import Terrain
from ..world.wind import WindController
from ..enums import GameStatus, BurnStatus
from ..game.managers.fire import RothermelFireManager
from ..world.parameters import Environment, FuelParticle
from ..game.managers.mitigation import (FireLineManager, ScratchLineManager,
                                        WetLineManager)


class Simulation(ABC):
    def __init__(self, config: Config) -> None:
        '''
        Initialize the Simulation object for interacting with the base
            simulation and the RL harness.

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
        Runs the simulation and displays the simulation's and agent's actions
        '''
        pass

    @abstractmethod
    def get_actions(self) -> Dict[str, int]:
        '''
        Returns the action space for the simulation
        '''
        pass

    @abstractmethod
    def get_attributes(self) -> Dict[str, np.ndarray]:
        '''
        Returns the observation space for the simulation
        '''
        pass


class RothermelSimulation(Simulation):
    def __init__(self, config: Config) -> None:
        '''
        Initialize the RothermelSimulation object for interacting with the base
        simulation and the RL harness.
        '''
        super().__init__(config)
        self.game_status = GameStatus.RUNNING
        self.fire_status = GameStatus.RUNNING
        self.points = set([])
        self._create_wind()
        self._create_terrain()
        self._create_fire()
        self._create_mitigations()

    def _create_terrain(self) -> None:
        '''
        Initialize the terrain
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
                                       self.wind_map.map_wind_speed,
                                       self.wind_map.map_wind_direction)

    def _create_mitigations(self) -> None:
        '''
        Initialize the mitigation strategies
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

    def _create_wind(self) -> None:
        '''
        This function will initialize the wind strategies
        '''
        self.wind_map = WindController()
        self.wind_map.init_wind_speed_generator(
            self.config.wind.speed.seed, self.config.wind.speed.scale,
            self.config.wind.speed.octaves, self.config.wind.speed.persistence,
            self.config.wind.speed.lacunarity, self.config.wind.speed.min,
            self.config.wind.speed.max, self.config.area.screen_size)
        self.wind_map.init_wind_direction_generator(
            self.config.wind.direction.seed, self.config.wind.direction.scale,
            self.config.wind.direction.octaves, self.config.wind.direction.persistence,
            self.config.wind.direction.lacunarity, self.config.wind.direction.min,
            self.config.wind.direction.max, self.config.area.screen_size)

    def _create_fire(self) -> None:
        '''
        This function will initialize the rothermel fire strategies
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
        This function will return the action space for the rothermel simulation

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
        This function will initialize and return the observation
            space for the simulation

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
            self.wind_map.map_wind_speed,
            'wind_direction':
            self.wind_map.map_wind_direction
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
        Runs the simulation with or without mitigation lines

        Use self.terrain to either:

          1. Place agent's mitigation lines and then spread fire
          2. Only spread fire, with no mitigation line
                (to compare for reward calculation)

        Arguments:
            mitigation_state: np.ndarray
                Array of mitigation value(s) as BurnStatus values.

            mitigation: bool
                Boolean for running the mitigation + fire spread which updates
                    fireline_manager sprites points or just the fire spread


        Returns:
            fire_map: Burned/Unburned/ControlLine pixel map. Values range from [0, 6]
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
            This method will interact with the RL harness to display and update the
                Rothermel simulation as the agent progresses through the simulation
                (if applicable, i.e AgentBasedHarness)

            TODO: position could change to a Dic[str, (int, int)]
                    for multi-agent scenario


            Arguments:
                mitigation: np.ndarray
                    The values of the mitigation array from the RL Harness, converted
                        to the simulation format

                position: np.ndarray
                    The position array of the agent

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
                                                self.wind_map.map_wind_speed,
                                                self.wind_map.map_wind_direction)

            self.fire_map = self.game.fire_map
            self.game.fire_map = self.fire_map
        self.points = set([])

    def _render_mitigations(self, mitigation: np.ndarray) -> None:
        '''
        This method will render the agent's actions after the final action.

        NOTE: this method doesn't seem really necessary -- might omit and
                use _render_mitigation_fire_spread() only

        Arguments:
            mitigation: np.ndarray
                The array of the final mitigation state from the RL Harness

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
                                                self.wind_map.map_wind_speed,
                                                self.wind_map.map_wind_direction)

            self.fire_map = self.game.fire_map
            self.game.fire_map = self.fire_map
        self.points = set([])

    def _render_mitigation_fire_spread(self, mitigation: np.ndarray) -> None:
        '''
        This method will render the agent's actions after the final action and
            the subsequent Rothermel Fire spread.

        Arguments:
            mitigation: np.ndarray
                The array of the final mitigation state from the RL Harness

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
                                                self.wind_map.map_wind_speed,
                                                self.wind_map.map_wind_direction)
            self.fire_map, self.fire_status = self.fire_manager.update(self.fire_map)
            self.fire_map = self.game.fire_map
            self.game.fire_map = self.fire_map

        self.points = set([])

    def render(self, type: str, mitigation: np.ndarray,
               position: np.ndarray = ([0], [0])) -> None:
        '''
        This is a helper function that hands off to sub-functions for rendering

        '''

        if type == 'inline':
            self._render_inline(mitigation, position)
        if type == 'post agent':
            self._render_mitigations(mitigation)
        if type == 'post agent with fire':
            self._render_mitigation_fire_spread(mitigation)
