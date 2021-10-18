# from dataclasses import astuple
# from typing import List, Tuple

import pygame
import config

from .sprites import Terrain
from src.enums import GameStatus
from src.game.Game import Game
from src.world.parameters import Environment, FuelArray, FuelParticle, Tile

from managers import RothermelFireManager


class FireLineEnv():
    '''
        This Environment Catcher will record all environments/actions
            for the RL return states.
        Inlcuding: Observation, Reward (or Penalty), "done", and any meta-data.

        This class will incorporate all gamelogic, input and rendering.

        It will also incorporate everything related to pygame into the render()
            method and separate init() and init_render() methods.

        We can then render ML routines using the step() and reset() method
            w/o loading the pygame package each step - if the environment is loaded,
            the rendering is not needed for training (saves execution time).

    '''
    def __init__(self):
        '''
            Initialize the class by recording the state space.
            We need to make a copy:
                Need to step through the state space twice:
                    1. Let the agent step through the space and draw firelines
                    2. Let the environemnt progress w/o agent
                Compare the two state spaces.

        '''
        self.points = []
        self.fuel_particle = FuelParticle()
        self.fuel_arrs = [[
            FuelArray(Tile(j, i, 0, config.terrain_scale, config.terrain_scale),
                      config.terrain_map[i][j]) for j in range(config.terrain_size)
        ] for i in range(config.terrain_size)]
        self.terrain = Terrain(self.fuel_arrs)
        self.environment = Environment(config.M_f, config.U, config.U_dir)

        # self.fireline_manager = FireLineManager(points=self.points,
        #                             size=config.control_line_size,
        #                             pixel_scale=config.pixel_scale,
        #                             terrain=self.terrain)

        self.fire_manager = RothermelFireManager(config.fire_init_pos, config.fire_size,
                                                 config.max_fire_duration,
                                                 config.pixel_scale,
                                                 rl_environment.fuel_particle,
                                                 self.terrain, self.environment)

    def init_render(self):
        pygame.init()
        self.game = Game(config.screen_size)

    def step(self, action):
        '''

        Input:
        -------
        action: action[0]: no trench, action[1]: trench
                points: ((x1, y1), (x2, y2))

        Return:
        -------
        observation:
        reward:
        done: end simulation, continue simulation
        info: extra meta-data
        '''

    def render(self):
        '''
        This will take the pygame update command and perform the display updates

        '''

        fire_sprites = self.fire_manager.sprites
        # fireline_sprites = self.fireline_manager.sprites
        # running = self.game.update(self.terrain, fire_sprites, fireline_sprites)
        running = self.game.update(self.terrain, fire_sprites)
        fire_map = self.game.fire_map
        fire_map = self.fireline_manager.update(fire_map, self.points)
        fire_map = self.fire_manager.update(fire_map)
        self.game.fire_map = fire_map

        return running

    def reset(self):
        '''
        reset environment to initial state
        '''
        # return observation

    def compare_spaces(self, fireline_space, fire_space):
        '''
        At the end of stepping through both state spaces, compare agent actions
            and reward

        '''


if __name__ == '__main__':

    rl_environment = FireLineEnv()
    rl_environment.init_render()

    # points = line(100, 0, 100, 224)
    # y = points[0].tolist()
    # x = points[1].tolist()
    # points = list(zip(x, y))

    # fire_map = game.fire_map
    # fire_map = fireline_manager.update(fire_map, points)
    # game.fire_map = fire_map

    # running = True
    # while running:

    #     # action

    #     # step
    #     rl_environment.step()

    #     # render
    #     running = rl_environment.render()

    game_status = GameStatus.RUNNING
    fire_status = GameStatus.RUNNING
    while game_status == GameStatus.RUNNING and fire_status == GameStatus.RUNNING:
        rl_environment.step()
        rl_environment.render()

    pygame.quit()
