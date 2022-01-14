import numpy as np

from typing import Dict
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
        Initialize the Simulation object for interacting with the base simulation and the
        RL harness.

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
        Runs the pygame simulation and displays the simulation's and agent's actions
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
        self.terrain = Terrain(self.fuel_arrs, self.config.terrain.elevation_function,
                               self.config.area.terrain_size,
                               self.config.area.screen_size)

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
            terrain=self.terrain)

        self.scratchline_manager = ScratchLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain)
        self.wetline_manager = WetLineManager(size=self.config.display.control_line_size,
                                              pixel_scale=self.config.area.pixel_scale,
                                              terrain=self.terrain)

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
            self.config.fire.fire_initial_position, self.config.display.fire_size,
            self.config.fire.max_fire_duration, self.config.area.pixel_scale,
            self.config.simulation.update_rate, self.fuel_particle, self.terrain,
            self.environment)
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

    def _update_sprite_points(self,
                              mitigation_state,
                              position_state,
                              inline: bool = False) -> None:
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
        if inline:
            if mitigation_state == BurnStatus.FIRELINE:
                self.points.add(position_state)
            elif mitigation_state == BurnStatus.SCRATCHLINE:
                self.points.add(position_state)
            elif mitigation_state == BurnStatus.WETLINE:
                self.points.add(position_state)

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

    def run(self,
            mitigation_state: np.ndarray,
            position_state: np.ndarray,
            mitigation: bool = False) -> np.ndarray:
        '''
        Runs the simulation with or without mitigation lines

        Use self.terrain to either:

          1. Place agent's mitigation lines and then spread fire
          2. Only spread fire, with no mitigation line
                (to compare for reward calculation)

        Arguments:
            mitigation_state: Array of mitigation value(s). 0: No Control Line,
                              1: Control Line
            position_state: Array of current agent position. Only used when rendering
                            `inline`.
            mitigation: Boolean value to update agent's mitigation staegy before fire
                        spread.

        Returns:
            fire_map: Burned/Unburned/ControlLine pixel map. Values range from [0, 6]
        '''
        # reset the fire status to running
        self.fire_status = GameStatus.RUNNING
        # initialize fire strategy
        self.fire_manager = RothermelFireManager(
            self.config.fire.fire_initial_position, self.config.display.fire_size,
            self.config.fire.max_fire_duration, self.config.area.pixel_scale,
            self.config.simulation.update_rate, self.fuel_particle, self.terrain,
            self.environment)

        self.fire_map = np.full(
            (self.config.area.screen_size, self.config.area.screen_size),
            BurnStatus.UNBURNED)
        if mitigation:
            # update firemap with agent actions before initializing fire spread
            self._update_sprite_points(mitigation_state, position_state)
            self.fire_map = self.fireline_manager.update(self.fire_map, self.points)

        while self.fire_status == GameStatus.RUNNING:
            self.fire_sprites = self.fire_manager.sprites
            self.fire_map, self.fire_status = self.fire_manager.update(self.fire_map)
            if self.fire_status == GameStatus.QUIT:
                return self.fire_map

    def render(self,
               mitigation_state: np.ndarray,
               position_state: np.ndarray,
               mitigation_only: bool = True,
               mitigation_and_fire_spread: bool = False,
               inline: bool = False) -> None:
        '''
        This will take the pygame update command and perform the display updates for the
        following scenarios:

            1. pro-active fire mitigation - inline (during step()) (no fire)
            2. pro-active fire mitigation (full traversal)
            3. pro-active fire mitigation (full traversal) + fire spread

        Arguments:
            mitigation_state: Array of either the current agent mitigation or all
                              mitigations.
            position_state: Array of either the current agent position only used when
                            `inline == True`.
            mitigation_only: Boolean value to only show agent mitigation stategy.
            mitigation_and_fire_spread: Boolean value to show agent mitigation
                                        stategy & fire spread.
                                        Only used when agent has traversed
                                        entire game board.
            inline: Boolean value to use rendering at each call to step().

        Returns:
            None
        '''
        self.fire_manager = RothermelFireManager(
            self.config.fire.fire_initial_position, self.config.display.fire_size,
            self.config.fire.max_fire_duration, self.config.area.pixel_scale,
            self.config.simulation.update_rate, self.fuel_particle, self.terrain,
            self.environment)
        self.game = Game(self.config.area.screen_size)
        self.fire_map = self.game.fire_map

        if mitigation_only:
            self._update_sprite_points(mitigation_state, position_state, inline)
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

        if mitigation_and_fire_spread:
            self.fire_status = GameStatus.RUNNING
            self.game_status = GameStatus.RUNNING
            self._update_sprite_points(mitigation_state, position_state, inline)
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

        # after rendering - reset mitigation points can always recover
        # these through the agent state info and _update_sprite_points()
        self.points = set([])
