# from dataclasses import astuple
import numpy as np
import pygame
import gym
from typing import List
from gym.utils import seeding
from src import config
from ..game.sprites import Terrain
from src.game.Game import Game
from src.world.parameters import Environment, FuelArray, FuelParticle, Tile
from ..game.managers.fire import RothermelFireManager
from ..game.managers.mitigation import (FireLineManager, ScratchLineManager,
                                        WetLineManager)


class FireLineEnv(gym.Env):
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

        Observation:
        ------------
        a = [[[0,0,0,0,0] for _ in range(255)] for _ in range (255)]
        b = [[[1,5,1,3,6] for _ in range(255)] for _ in range(255)]

        Type: Box(low=a, high=b, shape=(255,255,3))
        Num    Observation              min     max
        0      Agent Position           0       1
        1      Fuel (type)              0       5
        2      Burned/Unburned          0       1
        3      Line Type                0       3
        4      Burn Stats               0       6

        Actions:
        --------
        Type: Discrete(4) -- real-valued (on / off)
        Num    Action
        0      None
        1      Fireline
        2      ScratchLine
        3      Wetline


        Reward:
        -------
        Reward of [+ difference]
        Reward of [-1 + difference]

        Starting State:
        ---------------
        The position of the agent always starts in the top right corner (0,0).


        Episode Termination:
        ---------------------
        The agent has traversed all pixels (screen_size, screen_size)

    TODO: pull out Game functionality or make it a parameter to pass in
            if we want to train w/o pygame rendering or if we do.

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
        pygame.init()
        self.game = Game(config.screen_size)
        self.fuel_particle = FuelParticle()
        self.fuel_arrs = [[
            FuelArray(Tile(j, i, config.terrain_scale, config.terrain_scale),
                      config.terrain_map[i][j]) for j in range(config.terrain_size)
        ] for i in range(config.terrain_size)]
        self.terrain = Terrain(self.fuel_arrs, config.elevation_fn)
        self.environment = Environment(config.M_f, config.U, config.U_dir)

        # initialize all mitigation strategies
        self.fireline_manager = FireLineManager(size=config.control_line_size,
                                                pixel_scale=config.pixel_scale,
                                                terrain=self.terrain)

        self.scratchline_manager = ScratchLineManager(size=config.control_line_size,
                                                      pixel_scale=config.pixel_scale,
                                                      terrain=self.terrain)
        self.wetline_manager = WetLineManager(size=config.control_line_size,
                                              pixel_scale=config.pixel_scale,
                                              terrain=self.terrain)

        self.mitigation = {
            0: 'None',
            1: self.fireline_manager,
            2: self.scratchline_manager,
            3: self.wetline_manager
        }
        # initialize fire strategy
        self.fire_manager = RothermelFireManager(config.fire_init_pos, config.fire_size,
                                                 config.max_fire_duration,
                                                 config.pixel_scale, self.fuel_particle,
                                                 self.terrain, self.environment)

        self.fireline_sprites = self.fireline_manager.sprites
        self.scratchline_sprites = self.scratchline_manager.sprites
        self.wetline_sprites = self.wetline_manager.sprites

        self.fire_sprites = self.fire_manager.sprites
        self.fire_map = self.game.fire_map

        self.points = []

        # at start, agent is not in screen
        self.current_agent_loc = (0, 0)

        self.action_space = gym.spaces.Discrete(2)

        self.low = [[[0, 0, 0, 0] for _ in range(255)] for _ in range(255)]
        self.high = [[[1, 5, 1, 3] for _ in range(255)] for _ in range(255)]
        self.observation_space = gym.spaces.Box(self.low,
                                                self.high,
                                                shape=(config.screen_size,
                                                       config.screen_size, 4),
                                                dtype=np.float32)

        # TODO check if we can use dict (dont think stable-baselines3 supports it)
        # observ_spaces = {
        #     'agent position': self.state_space,
        #     # gym.spaces.Discrete(1, shape=(config.screen_size, config.screen_size)),
        #     'fuel': self.terrain.fuel_arrs,
        #     # gym.spaces.Discrete(low=0, high=4, shape=(config.screen_size,
        #                           config.screen_size)),
        #     'burn status': self.fire_map
        #     # gym.spaces.Discrete(low=0, high=5, shape=(config.screen_size,
        #                               config.screen_size))
        # }
        # self.observation_space = observ_spaces

        # always keep the same if terrain and fire position are static
        self.seed(1234)

    def seed(self, seed=None):
        '''
        Set the seed for numpy random methods.

        Input:
        -------
        seed: Random seeding value

        Return:
        -------
        seed: Random seeding value
        '''

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        This function will apply the action.

        Calculate new agent/state_space position by reseting old position to zero,
            call the _update_current_agent_loc method and set new
            agent/state_space location to 1.

        Done: Occurs when the agent has traversed the entire game
                position: [screen_size, screen_size]


        Input:
        -------

            action: {0: 0, 1: 0, 2: 0, 3: 0}

        Return:
        -------
        observation: [screen_size, screen_size, 4]: terrain, agent pos, fuel,
        reward: -1 if trench, 0 if None
        done: end simulation, calculate state differences
        info: extra meta-data
        '''

        # Make the action on the env at agent's current location
        self.state[self.current_agent_loc[0], self.current_agent_loc[1], 4] = action

        # get the fire mitigation type (there could be more than 1)
        sprites_num = [k for k, v in action.items() if v >= 1]

        if len(sprites_num) > 0:
            # agent will put down some fire mitigation
            self._update_sprites(sprites_num)

        old_loc = self.current_agent_loc.copy()

        # update position
        self._update_current_agent_loc()

        # If the agent is at the last location (cannot move forward)
        done = old_loc == self.current_agent_loc

        # reward is calculated after final step(): done == True
        if done:
            # compare the state spaces
            difference, reward_agent = self.compare_spaces()
            if reward_agent:
                reward = -1 + difference
            else:
                reward = difference

        else:
            reward = 0 if action == 0 else -1

        return self.state, reward, done, {}

    def _update_sprites(self, sprites_num: List):

        # update the location to pass to the sprite
        self.points = self.current_agent_loc
        sprites = [self.mitigatoin[num] for num in sprites_num]
        for sprite in sprites:
            if sprite == self.fireline_manager:
                self.fireline_sprites = self.fireline_manager._add_point(
                    point=self.points)
            elif sprite == self.scratchline_manager:
                self.scratchine_sprites = self.scratchline_manager._add_point(
                    point=self.points)
            elif sprite == self.wetline_manager:
                self.wetline_sprites = self.wetline_manager._add_point(point=self.points)

    def _update_current_agent_loc(self):
        '''
        This function will help update the current agent position
            as it traverses the game.

        Check if the y-axis is less than the screen-size and the
            x-axis is greater than the screen size --> y-axis += 1, x-axis = 0

        Check if the x-axis is less than screen size and the y-axis is
            less than/equal to screen size --> y-axis = None, x-axis += 1

        '''
        x = self.current_agent_loc[0]
        y = self.current_agent_loc[1]

        # If moving forward one would bring us out of bounds and we can move to new row
        if x + 1 > (config.screen_size - 1) and y + 1 <= (config.screen_size - 1):
            self.current_agent_loc[0] = 0
            self.current_agent_loc[1] += 1

        # If moving forward keeps us in bounds
        elif x + 1 <= (config.screen_size - 1):
            self.current_agent_loc[1] += 1

    def render(self):
        '''
        This will take the pygame update command and perform the display updates

        '''

        game_status = self.game.update(self.terrain, self.fire_sprites,
                                       self.fireline_sprites)
        self.fire_map = self.game.fire_map
        if len(self.points) == 0:
            self.fire_map = self.fireline_manager.update(self.fire_map)
        else:
            self.fire_map = self.fireline_manager.update(self.fire_map, self.points)
        self.fire_map = self.fire_manager.update(self.fire_map)
        self.game.fire_map = self.fire_map

        return game_status

    def reset(self):
        '''
        Reset environment to initial state.
        NOTE: reset() must be called before you can call step()

        Terrain is received from the sim.
        Agent position matrix is assumed to be all 0's when received from sim.
            Updated to have agent at (0,0) on reset.

        '''
        # TODO this should be a new terrain from the sim - dummy for now.
        # ASSUMPTION: [:,:,0] (agent position matrix) are all 0s when
        #       received from sim
        self.state = np.zeros((255, 255, 5))

        # Place agent at location (0,0)
        self.state[0, 0, 0] = 1
        self.current_agent_loc = (0, 0)

        return np.array(self.state, dtype=np.float32)

    def compare_spaces(self):
        '''
        At the end of stepping through both state spaces, compare fianl agent action space
            and final observation space of burned terrain

        '''
        difference = np.diff(self.observation_space['burn status'], self.action_space)
        # need logic for whether or not to reward the agent based on some difference
        # criteria
        reward_agent = False
        return difference, reward_agent
