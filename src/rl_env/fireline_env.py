# from dataclasses import astuple
# from typing import List, Tuple
import numpy as np
import pygame
import config
from skimage.draw import line
from gym import spaces
from ..game.sprites import Terrain
from src.enums import GameStatus
from src.game.Game import Game
from src.world.parameters import Environment, FuelArray, FuelParticle, Tile
from ..game.managers.fire import RothermelFireManager
from ..game.managers.mitigation import FireLineManager


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

        Observation:
        ------------
        Type: Box(4)
        Num    Observation              min     max
        0      Agent Position           0       1
        1      Fuel (type)              0       5
        2      Burned/Unburned          0       1

        TODO: shape will need to fit Box(low=x, high=x, shape=x, dtype=x) where low/high are min/max values. Linear transformation?
        https://github.com/openai/gym/blob/3eb699228024f600e1e5e98218b15381f065abff/gym/spaces/box.py#L7
        Line 19 - Independent bound for each dimension

        Actions:
        --------
        Type: Discrete(4) -- real-valued (on / off)
        Num    Action
        0      None
        1      Trench
        2      ScratchLine
        3      WetLine

        Reward:
        -------
        Reward of 0 when 'None' action is taken and agent position is not the last tile.
        Reward of -1 when 'Trench, ScratchLine, WetLine' action is taken and agent position is not the last tile.
        Reward of (fire_burned - fireline_burned) when done.

        TODO: Will probably want to normalize (difference) to be [0,1] or something similar. reward values between [0,1] result in better training.

        Starting State:
        ---------------
        The position of the agent always starts in the top right corner (0,0).


        Episode Termination:
        ---------------------
        The agent has traversed all pixels (screen_size, screen_size)


    '''
    def __init__(self, mode: str):
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

        self.fireline_manager = FireLineManager(points=self.points,
                                                size=config.control_line_size,
                                                pixel_scale=config.pixel_scale,
                                                terrain=self.terrain)

        self.fire_manager = RothermelFireManager(config.fire_init_pos, config.fire_size,
                                                 config.max_fire_duration,
                                                 config.pixel_scale,
                                                 rl_environment.fuel_particle,
                                                 self.terrain, self.environment)

        if mode == 'static terrain':
            self.observation = self.terrain
        elif mode == 'dynamic terrain':
            self.observation = config.fire_init_pos

        self.action_space = spaces.Discrete(4)
        self.low = np.array([0, 0, 0])
        self.high = np.array([1, 5, 1])
        self.observation_space = spaces.Box(self.low,
                                            self.high,
                                            shape=(config.screen_size,
                                                   config.screen_size),
                                            dtype=np.float32)

        # defines the agen location in teh state at each step
        self.state_space = np.zeros(config.screen_size, config.screen_size)
        # at start, agent is in top left corner
        self.current_agent_loc = (0, 0)

    def init_render(self):
        pygame.init()
        self.game = Game(config.screen_size)

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
        action: action[0]: no trench, action[1]: trench
                points: ((x1, y1), (x2, y2))

        Return:
        -------
        observation: [screen_size, screen_size, 4]: terrain, agent pos, fuel,
        reward: -1 if trench, 0 if None
        done: end simulation, calculate state differences
        info: extra meta-data
        '''

        self.state_space[self.current_agent_loc] = 0
        self._update_current_agent_loc()
        self.state_space[self.current_agent_loc] = 1

        done = bool(self.current_agent_loc[0] == (config.screen_size - 1)
                    and self.current_agent_loc[1] == (config.screen_size - 1))

        return done

    def _update_current_agent_loc(self):
        '''
        This function will help update the current agent position
            as it traverses the game.

        Check if the y-axis is less than the screen-size and the
            x-axis is greater than the screen size --> y-axis += 1, x-axis = 0

        Check if the x-axis is less than screen size and the y-axis is
            less than/equal to screen size --> y-axis = None, x-axis += 1

        '''
        y = self.current_agent_loc[0]
        x = self.current_agent_loc[1]

        if x > (config.screen_size - 1) and y < (config.screen_size - 1):
            self.current_agent_loc[0] += 1
            self.current_agent_loc[1] = 0

        elif x < (config.screen_size - 1) and y <= (config.screen_size - 1):
            self.current_agent_loc[1] += 1

    def render(self):
        '''
        This will take the pygame update command and perform the display updates

        '''

        fire_sprites = self.fire_manager.sprites
        fireline_sprites = self.fireline_manager.sprites
        game_status = self.game.update(self.terrain, fire_sprites, fireline_sprites)
        fire_map = self.game.fire_map
        fire_map = self.fireline_manager.update(fire_map, self.points)
        fire_map = self.fire_manager.update(fire_map)
        self.game.fire_map = fire_map

        return game_status

    def reset(self):
        '''
        reset environment to initial state

        '''
        self.observation

    def compare_spaces(self, fireline_space, fire_space):
        '''
        At the end of stepping through both state spaces, compare agent actions
            and reward

        '''


def get_action():
    '''

    FOR TESTING ONLY

    Get the action from the rl agent.

    Return:
    -------
    action: {0: 'null', 1: points}
            action[0]: no trench,
            action[1]: trench
                points: ((x1, y1), (x2, y2))

    '''
    action = {}
    # for testing
    points = line(100, 0, 100, 224)
    y = points[0].tolist()
    x = points[1].tolist()
    points = list(zip(x, y))
    action
    return


if __name__ == '__main__':

    # initilize the rl_env()
    rl_environment = FireLineEnv()
    rl_environment.init_render()

    # fire_map = game.fire_map
    # fire_map = fireline_manager.update(fire_map, points)
    # game.fire_map = fire_map

    observation = rl_environment.reset()
    game_status = GameStatus.RUNNING
    fire_status = GameStatus.RUNNING
    while game_status == GameStatus.RUNNING and fire_status == GameStatus.RUNNING:
        action = get_action()
        rl_environment.render()
        rl_environment.step(action)

    pygame.quit()
