# from copy import deepcopy
# from typing import Dict, Tuple

import numpy as np

from .. import config
from ..enums import GameStatus, BurnStatus
from ..game.game import Game
from ..game.sprites import Terrain
from ..world.wind import WindController
from ..game.managers.fire import RothermelFireManager
from ..world.parameters import Environment, FuelArray, FuelParticle, Tile
from ..game.managers.mitigation import (FireLineManager, ScratchLineManager,
                                        WetLineManager)
from ..utils.terrain import Chaparral, RandomSeedList


class FireLineEnv():
    def __init__(self, config: config, seed: int = None):

        self.config = config
        self.points = set([])

        if seed:
            np.random.seed(seed)
            seed_tuple = RandomSeedList(self.config.terrain_size, seed)
            self.config.terrain_map = tuple(
                tuple(
                    Chaparral(seed=seed_tuple[outer][inner])
                    for inner in range(self.config.terrain_size))
                for outer in range(self.config.terrain_size))
        else:
            self.config.terrain_map = tuple(
                tuple(Chaparral() for _ in range(self.config.terrain_size))
                for _ in range(self.config.terrain_size))

        self.fuel_particle = FuelParticle()
        self.fuel_arrs = [[
            FuelArray(Tile(j, i, self.config.terrain_scale, self.config.terrain_scale),
                      self.config.terrain_map[i][j])
            for j in range(self.config.terrain_size)
        ] for i in range(self.config.terrain_size)]
        self.terrain = Terrain(self.fuel_arrs, self.config.elevation_fn,
                               self.config.terrain_size, self.config.screen_size)
        self.wind_map = WindController()
        self.wind_map.init_wind_speed_generator(
            self.config.mw_seed, self.config.mw_scale, self.config.mw_octaves,
            self.config.mw_persistence, self.config.mw_lacunarity,
            self.config.mw_speed_min, self.config.mw_speed_max, self.config.screen_size)
        self.wind_map.init_wind_direction_generator(
            self.config.dw_seed, self.config.dw_scale, self.config.dw_octaves,
            self.config.dw_persistence, self.config.dw_lacunarity, self.config.dw_deg_min,
            self.config.dw_deg_max, self.config.screen_size)

        self.environment = Environment(self.config.M_f, self.wind_map.map_wind_speed,
                                       self.wind_map.map_wind_direction)

        # initialize all mitigation strategies
        self.fireline_manager = FireLineManager(size=self.config.control_line_size,
                                                pixel_scale=self.config.pixel_scale,
                                                terrain=self.terrain)

        self.scratchline_manager = ScratchLineManager(self.config.control_line_size,
                                                      self.config.pixel_scale,
                                                      self.terrain)
        self.wetline_manager = WetLineManager(size=self.config.control_line_size,
                                              pixel_scale=self.config.pixel_scale,
                                              terrain=self.terrain)

        self.fireline_sprites = self.fireline_manager.sprites
        self.fireline_sprites_empty = self.fireline_sprites.copy()
        self.scratchline_sprites = self.scratchline_manager.sprites
        self.wetline_sprites = self.wetline_manager.sprites

        self.fire_manager = RothermelFireManager(
            self.config.fire_init_pos, self.config.fire_size,
            self.config.max_fire_duration, self.config.pixel_scale,
            self.config.update_rate, self.fuel_particle, self.terrain, self.environment)
        self.fire_sprites = self.fire_manager.sprites

        self.game_status = GameStatus.RUNNING
        self.fire_status = GameStatus.RUNNING

    def _render(self,
                mitigation_state: np.ndarray,
                position_state: np.ndarray,
                mitigation_only: bool = True,
                mitigation_and_fire_spread: bool = False,
                inline: bool = False) -> None:
        '''
        This will take the pygame update command and perform the display updates
            for the following scenarios:
                1. pro-active fire mitigation - inline (during step()) (no fire)
                2. pro-active fire mitigation (full traversal)
                3. pro-active fire mitigation (full traversal) + fire spread

        Arguments:
            mitigation_state: Array of either the current agent mitigation or all
                              mitigations.
            position_state: Array of either the current agent position only used when
                            `inline == True`.

            mitigation_only: Boolean value to only show agent mitigation stategy.

            mitigation_and_fire_spread: Boolean value to show agent mitigation stategy and
                                        fire spread. Only used when agent has traversed
                                        entire game board.
            inline: Boolean value to use rendering at each call to step().

        Returns:
            None
        '''

        self.fire_manager = RothermelFireManager(
            self.config.fire_init_pos, self.config.fire_size,
            self.config.max_fire_duration, self.config.pixel_scale,
            self.config.update_rate, self.fuel_particle, self.terrain, self.environment)
        self.game = Game(self.config.screen_size)
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
            if mitigation_state == 1:
                self.points.add(position_state)

        else:
            # update the location to pass to the sprite
            for i in range(self.config.screen_size):
                for j in range(self.config.screen_size):
                    if mitigation_state[(i, j)] == 1:
                        self.points.add((i, j))

    def _run(self,
             mitigation_state: np.ndarray,
             position_state: np.ndarray,
             mitigation: bool = False):
        '''
        Runs the simulation with or without mitigation lines

        Use self.terrain to either:

          1. Place agent's mitigation lines and then spread fire
          2. Only spread fire, with no mitigation line (to compare for reward calculation)

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
            self.config.fire_init_pos, self.config.fire_size,
            self.config.max_fire_duration, self.config.pixel_scale,
            self.config.update_rate, self.fuel_particle, self.terrain, self.environment)

        self.fire_map = np.full((self.config.screen_size, self.config.screen_size),
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
