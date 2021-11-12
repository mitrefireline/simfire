from copy import deepcopy

import gym
import numpy as np
from gym.utils import seeding

from ..enums import GameStatus, BurnStatus
from .. import config
from ..game.sprites import Terrain
from ..game.game import Game
from ..world.parameters import Environment, FuelArray, FuelParticle, Tile
from ..game.managers.fire import RothermelFireManager
from ..game.managers.mitigation import (FireLineManager, ScratchLineManager,
                                        WetLineManager)


class FireLineEnv():
    def __init__(self, config: config):

        self.config = config
        self.points = set([])

        self.fuel_particle = FuelParticle()
        self.fuel_arrs = [[
            FuelArray(Tile(j, i, self.config.terrain_scale, self.config.terrain_scale),
                      self.config.terrain_map[i][j])
            for j in range(self.config.terrain_size)
        ] for i in range(self.config.terrain_size)]
        self.terrain = Terrain(self.fuel_arrs, self.config.elevation_fn,
                               self.config.terrain_size, self.config.screen_size)
        self.environment = Environment(self.config.M_f, self.config.U, self.config.U_dir)

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

        self.fire_manager = RothermelFireManager(self.config.fire_init_pos,
                                                 self.config.fire_size,
                                                 self.config.max_fire_duration,
                                                 self.config.pixel_scale,
                                                 self.fuel_particle, self.terrain,
                                                 self.environment)
        self.fire_sprites = self.fire_manager.sprites

        self.game_status = GameStatus.RUNNING
        self.fire_status = GameStatus.RUNNING

        self.actions = ['None', 'fireline']
        self.observ_spaces = {
            'position': (0, 1),
            'w_0': (0, 1),
            # 'sigma': (0, 1),
            # 'delta': (0, 1),
            # 'm_x': (0, 1),
            # 'm_f': (0, 1)
            'elevation': (0, 1),
            'mitigation': (0, len(self.actions) - 1)
        }

    def _render(self,
                mitigation_state: np.ndarray,
                position_state: np.ndarray,
                mitigation_only: bool = True,
                mitigation_and_fire_spread: bool = False,
                inline: bool = False):
        '''
        This will take the pygame update command and perform the display updates
            for the following scenarios:
                1. pro-active fire mitigation - inline (during step()) (no fire)
                2. pro-active fire mitigation (full traversal)
                3. pro-active fire mitigation (full traversal) + fire spread

        Parameters
        -----------
        mitigation_state: np.ndarray
                            Array of either the current agent mitigation
                                or all mitigations.

        position_state: np.ndarray
                        Array of either the current agent position only
                            used when 'inline' == True

        mitigation_only: bool = True
                            Boolean value to only show agent mitigation stategy

        mitigation_and_fire_spread: bool = False
                            Boolean value to show agent mitigation stategy and
                                fire spread. Only used when agent has traversed
                                etire game board.
        inline: bool = False
                Boolean value to use rendering at each call to step()

        Return
        -------
        None

        '''

        self.fire_manager = RothermelFireManager(self.config.fire_init_pos,
                                                 self.config.fire_size,
                                                 self.config.max_fire_duration,
                                                 self.config.pixel_scale,
                                                 self.fuel_particle, self.terrain,
                                                 self.environment)
        self.game = Game(self.config.screen_size)
        self.fire_map = self.game.fire_map

        if mitigation_only:
            self._update_sprite_points(mitigation_state, position_state, inline)
            if self.game_status == GameStatus.RUNNING:

                self.fire_map = self.fireline_manager.update(self.fire_map, self.points)
                self.fireline_sprites = self.fireline_manager.sprites
                self.game.fire_map = self.fire_map
                self.game_status = self.game.update(self.terrain, self.fire_sprites,
                                                    self.fireline_sprites)

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
                                                    self.fireline_sprites)
                self.fire_map, self.fire_status = self.fire_manager.update(self.fire_map)
                self.fire_map = self.game.fire_map
                self.game.fire_map = self.fire_map

        # after rendering - reset mitigation points can always recover
        # these through the agent state info and _update_sprite_points()
        self.points = set([])

    def _update_sprite_points(self,
                              mitigation_state,
                              position_state,
                              inline: bool = False):
        '''
        Update sprite point list based on fire mitigation.

        Parameters
        -----------
        mitigation_state: np.ndarray
                            Array of mitigation value(s).
                            0: No Control Line, 1: Control Line
        position_state: np.ndarray
                        Array of position. Only used when rendering `inline`
        inline: bool
                Boolean value of whether or not to render at each step() or after
                    agent has placed control lines
                If True, will use mitigation state, position_state to add a new
                    point to the fireline sprites group
                If False, loop through all mitigation_state array to get points
                    to add to fireline sprites group

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

        Use self.terrain to either:
            1. Place agent's mitigation lines and then spread fire
            2. Only spread fire, with no mitigation line
                    (to compare for reward calculation)

        Parameters
        ----------
        mitigation_state: np.ndarray
                            Array of mitigation value(s).
                            0: No Control Line, 1: Control Line
        position_state: np.ndarray
                        Array of current agent position.
                        Only used when rendering `inline`

        mitigation: bool
                    Boolean value to update agent's mitigation staegy before
                        fire spread.

        Return
        ------
        fire_map: np.ndarray
                    Burned/Unburned/ControlLine pixel map.
                    Values range from [0, 6]
        '''

        # reset the fire status to running
        self.fire_status = GameStatus.RUNNING
        # initialize fire strategy
        self.fire_manager = RothermelFireManager(self.config.fire_init_pos,
                                                 self.config.fire_size,
                                                 self.config.max_fire_duration,
                                                 self.config.pixel_scale,
                                                 self.fuel_particle, self.terrain,
                                                 self.environment)

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

    def _reset_state(self):
        '''
        This function will convert the initialized terrain
            to the gym.spaces.Box format.

        self.current_agent_loc --> [:,:,0]
        self.terrain.fuel_arrs.type --> [:,:,1[0]]
        self.terrain.fuel_arrs.w_0 --> [:,:,1[1]]
        self.terrain.fuel_arrs.sigma --> [:,:,1[2]]
        self.terrain.fuel_arrs.delta --> [:,:,1[3]]
        self.terrain.fuel_arrs.M_x --> [:,:,1[4]]
        self.elevation --> [:,:,2]
        self.mitigation --> [:,:,3]

        Parameters
        -----------
        None

        Return
        -------
        state: Dict[np.ndarray]


        '''
        reset_position = np.zeros([self.config.screen_size, self.config.screen_size])

        w_0_array = np.array([
            self.terrain.fuel_arrs[i][j].fuel.w_0 for j in range(self.config.screen_size)
            for i in range(self.config.screen_size)
        ]).reshape(self.config.screen_size, self.config.screen_size)
        # sigma_array = np.array([
        #     self.terrain.fuel_arrs[i][j].fuel.sigma
        #     for j in range(self.config.screen_size)
        #     for i in range(self.config.screen_size)
        # ]).reshape(self.config.screen_size, self.config.screen_size)
        # delta_array = np.array([
        #     self.terrain.fuel_arrs[i][j].fuel.delta
        #     for j in range(self.config.screen_size)
        #     for i in range(self.config.screen_size)
        # ]).reshape(self.config.screen_size, self.config.screen_size)
        # M_x_array = np.array([
        #    self.terrain.fuel_arrs[i][j].fuel.M_x for j in
        #           range(self.config.screen_size)
        #     for i in range(self.config.screen_size)
        # ]).reshape(self.config.screen_size, self.config.screen_size)
        reset_mitigation = np.zeros([self.config.screen_size, self.config.screen_size])

        # (screen_size, screen_size, 4)
        # terrain = np.stack((w_0_array, sigma_array, delta_array, M_x_array), axis=-1)
        state = np.stack((reset_position, w_0_array,
                          (self.terrain.elevations + self.config.noise_amplitude) /
                          (2 * self.config.noise_amplitude), reset_mitigation))

        return state

    def _compare_states(self, fire_map, fire_map_with_agent):
        '''
        Calculate the reward for the agent's actions.
        Reward is determined by:

        TYPE:                   min:     max:
        ------                  ----    ----
        Burn Stats               0       6

        0   Unburned
        1   Burning (this should never be used at the very end of a sim)
        2   Burned
        3   Fireline
        4   Scratchline
        5   Wetline

        firemap_with_agent: [2] -> reward = -1
        firemap_with_agent: [3] -> reward = -1
        firemap: [2] firemap_with_agent: [0] -> reward = +1
        else 0

        Input:
        ------
        fire_map [0, 2]
        fire_map_with_agent [0, 2, 3]

        Return:
        -------
        float: score / difference between the firemaps (normalized wrt screen size)

        '''
        reward = 0
        mod = 0
        unmod = 0
        for x in range(self.config.screen_size):
            for y in range(self.config.screen_size):
                modified = fire_map_with_agent[x][y]
                unmodified = fire_map[x][y]

                # How well did the unmodified map perform at this tile
                if unmodified == 0:
                    unmod_reward = 0
                if unmodified == 2:
                    unmod_reward = -1
                else:
                    unmod_reward = 0

                # How well did the modified map perform at this tile
                if modified == 0:
                    mod_reward = 0
                elif modified == 2:
                    mod_reward = -1
                elif modified == 3:
                    mod_reward = -1
                else:
                    mod_reward = 0

                reward += mod_reward - unmod_reward
                mod += mod_reward
                unmod += unmod_reward

        return reward / (self.config.screen_size * self.config.screen_size)


class RLEnv(gym.Env):
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
        Type: gym.spaces.Dict(Box(low=min, high=max, shape=(255,255,len(max))))


        Num    Observation              min     max
        0      position                 0       1
        1      Fuel (w_0)               0       1
        2      Elevation                0       1 (float)
        3      mitigation               0       1



            In Reactive Case:
            -----------------
            Type: gym.spaces.Dict(Box(low=min, high=max, shape=(255,255,len(max))))
            Num    Observation              min     max
            0      position                 0       1
            1      Fuel (w_0)               0       1
            2      Burned/Unburned          0       1
            3      mitigation               0       3
            4      Burn Stats               0       6


        Actions:
        --------
        Type: Discrete(4) -- real-valued (on / off)
        Num    Action
        0      None
        1      Fireline

        TODO:
        -------
        2      ScratchLine
        3      WetLine


        Reward:
        -------
        Reward of 0 when 'None' action is taken and position is not
                            the last tile.
        Reward of -1 when 'Trench, ScratchLine, WetLine' action is taken and agent
                            position is not the last tile.
        Reward of (fire_burned - fireline_burned) when done.

        Starting State:
        ---------------
        The position of the agent always starts in the top right corner (0,0).

        Episode Termination:
        ---------------------
        The agent has traversed all pixels (screen_size, screen_size)
    '''
    def __init__(self, simulation: FireLineEnv):
        '''
            Initialize the class by recording the state space.

            Using a self.configurable terrain and fire start position:
                Need to step through the state space twice:
                    1. Let the agent step through the space and draw firelines
                    2. Let the environemnt progress w/o agent
                Compare the two state spaces.

            Initialize the observation space and state space as described
                in the docstrings

        '''

        self.simulation = simulation
        self.action_space = gym.spaces.Discrete(len(self.simulation.actions))

        channel_lows = np.array([[[self.simulation.observ_spaces[channel][0]]]
                                 for channel in self.simulation.observ_spaces.keys()])
        channel_highs = np.array([[[self.simulation.observ_spaces[channel][1]]]
                                  for channel in self.simulation.observ_spaces.keys()])

        self.low = np.repeat(np.repeat(channel_lows,
                                       self.simulation.config.screen_size,
                                       axis=1),
                             self.simulation.config.screen_size,
                             axis=2)

        self.high = np.repeat(np.repeat(channel_highs,
                                        self.simulation.config.screen_size,
                                        axis=1),
                              self.simulation.config.screen_size,
                              axis=2)

        self.observation_space = gym.spaces.Box(
            np.float32(self.low),
            np.float32(self.high),
            shape=(len(self.simulation.observ_spaces.keys()),
                   self.simulation.config.screen_size,
                   self.simulation.config.screen_size),
            dtype=np.float64)

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
        This function will apply the action to the agent in the current state.

        Calculate new agent/state_space position by reseting old position to zero,
            call the _update_current_agent_loc() method and set new
            agent/state_space location to 1.

        Done: Occurs when the agent has traversed the entire game
                position: [screen_size, screen_size]


        Input:
        -------

        action: int

        Return:
        -------
        observation: Dict(
                        'position': (screen_size, screen_size, 1),
                        'terrain': (screen_size, screen_size, 5),
                        'elevation': (screen_size, screen_size, 1),
                        'mitigation': (screen_size, screen_size, 1)
                        )

        reward: -1 if trench, 0 if None
        done: end simulation, calculate state differences
        info: extra meta-data
        '''

        # Make the action on the env at agent's current location
        self.state[-1][self.current_agent_loc] = action

        # make a copy
        old_loc = deepcopy(self.current_agent_loc)

        # if we want to render as we step through the game
        if self.simulation.config.render_inline:
            self.simulation._render(self.state[-1][self.current_agent_loc],
                                    self.current_agent_loc,
                                    inline=True)

        # set the old position back to 0
        self.state[0][self.current_agent_loc] = 0

        # update position
        self._update_current_agent_loc()
        self.state[0][self.current_agent_loc] = 1

        reward = 0  # if action == 0 else
        # (-1 / (self.simulation.config.screen_size * self.simulation.config.screen_size))

        # If the agent is at the last location (cannot move forward)
        done = old_loc == self.current_agent_loc
        if done:
            # compare the state spaces
            fire_map = self.simulation._run(self.state[-1], self.state[0])
            # render only agent
            if self.simulation.config.render_post_agent:
                self.simulation._render(self.state[-1], self.state[0])

            fire_map_with_agent = self.simulation._run(self.state[-1], self.state[0],
                                                       True)
            # render fire with agents mitigation in place
            if self.simulation.config.render_post_agent_with_fire:
                self.simulation._render(self.state[-1],
                                        self.state[0],
                                        mitigation_only=False,
                                        mitigation_and_fire_spread=True)
            reward = self.simulation._compare_states(fire_map, fire_map_with_agent)

        return self.state, reward, done, {}

    def _update_current_agent_loc(self):
        '''
        This function will help update the current position
            as it traverses the game.

        Check if the y-axis is less than the screen-size and the
            x-axis is greater than the screen size --> y-axis += 1, x-axis = 0

        Check if the x-axis is less than screen size and the y-axis is
            less than/equal to screen size --> y-axis = None, x-axis += 1

        '''
        row = self.current_agent_loc[0]
        column = self.current_agent_loc[1]

        # If moving forward one would bring us out of bounds and we can move to new row
        if column + 1 > (self.simulation.config.screen_size -
                         1) and row + 1 <= (self.simulation.config.screen_size - 1):
            column = 0
            row += 1

        # If moving forward keeps us in bounds
        elif column + 1 <= (self.simulation.config.screen_size - 1):
            column += 1

        self.current_agent_loc = (row, column)

    def reset(self):
        '''
        Reset environment to initial state.
        NOTE: reset() must be called before you can call step() for the first time.

        Terrain is received from the sim.
        position matrix is assumed to be all 0's when received from sim.
            Updated to have agent at (0,0) on reset.

        '''

        self.state = self.simulation._reset_state()

        # Place agent at location (0,0)
        self.current_agent_loc = (0, 0)
        self.state[0][self.current_agent_loc] = 1

        return self.state
