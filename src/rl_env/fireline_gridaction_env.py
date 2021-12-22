import gym
import numpy as np
from gym.utils import seeding

from typing import Dict, Tuple, Any, List

from ..game.game import Game
from ..utils.config import Config
from .fireline_env import FireLineEnv
from ..enums import GameStatus, BurnStatus
from ..game.managers.fire import RothermelFireManager


class FireLineGridEnv(FireLineEnv):
    def __init__(self, config: Config, seed: int = None) -> None:
        super().__init__(config, seed)
        # Overwrite the super's observation space
        self.observ_spaces = {
            'w_0': (0, 1),
            # 'sigma': (0, 1),
            # 'delta': (0, 1),
            # 'm_x': (0, 1),
            # 'm_f': (0, 1)
            'elevation': (0, 1)
        }

    def _render(self,
                mitigation_state: np.ndarray,
                mitigation_only: bool = True,
                mitigation_and_fire_spread: bool = False):
        '''
        This will take the pygame update command and perform the display updates
            for the following scenarios:
                1. pro-active fire mitigation - inline (during step()) (no fire)
                2. pro-active fire mitigation (full traversal)
                3. pro-active fire mitigation (full traversal) + fire spread

        Arguments:
            mitigation_state: Array of either the current agent mitigation or all
                              mitigations.

            mitigation_only: Boolean value to only show agent mitigation stategy.

            mitigation_and_fire_spread: Boolean value to show agent mitigation stategy and
                                        fire spread. Only used when agent has traversed
                                        entire game board.

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
            self._update_sprite_points(mitigation_state)
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
            self._update_sprite_points(mitigation_state)
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

    def _update_sprite_points(self, mitigation_state: np.ndarray) -> None:
        '''
        Update sprite point list based on fire mitigation.

        Arguments:
            mitigation_state: Array of mitigation value(s). 0: No Control Line,
                              1: Control Line

        Returns:
            None
        '''
        # update the location to pass to the sprite
        for i in range(self.config.screen_size):
            for j in range(self.config.screen_size):
                if mitigation_state[(i, j)] == 1:
                    self.points.add((i, j))

    def _run(self, mitigation_state: np.ndarray, mitigation: bool = False) -> np.ndarray:
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
            self._update_sprite_points(mitigation_state)
            self.fire_map = self.fireline_manager.update(self.fire_map, self.points)

        while self.fire_status == GameStatus.RUNNING:
            self.fire_sprites = self.fire_manager.sprites
            self.fire_map, self.fire_status = self.fire_manager.update(self.fire_map)
            if self.fire_status == GameStatus.QUIT:
                return self.fire_map

    def _reset_state(self) -> np.ndarray:
        '''
        This function will convert the initialized terrain
            to the gym.spaces.Box format.

        ```
        self.current_agent_loc --> [:,:,0]
        self.terrain.fuel_arrs.type --> [:,:,1[0]]
        self.terrain.fuel_arrs.w_0 --> [:,:,1[1]]
        self.terrain.fuel_arrs.sigma --> [:,:,1[2]]
        self.terrain.fuel_arrs.delta --> [:,:,1[3]]
        self.terrain.fuel_arrs.M_x --> [:,:,1[4]]
        self.elevation --> [:,:,2]
        self.mitigation --> [:,:,3]
        ```

        Arguments:
            None

        Returns:
            state: The reset state of the simulation.
        '''
        reset_position = np.zeros(
            [self.config.area.screen_size, self.config.area.screen_size])

        w_0_array = np.array([
            self.terrain.fuel_arrs[i][j].fuel.w_0
            for j in range(self.config.area.screen_size)
            for i in range(self.config.area.screen_size)
        ]).reshape(self.config.area.screen_size, self.config.area.screen_size)

        elevations_zero_min = self.terrain.elevations - self.terrain.elevations.min()
        elevations_norm = elevations_zero_min / (elevations_zero_min.max() + 1e-6)

        reset_mitigation = np.zeros(
            [self.config.area.screen_size, self.config.area.screen_size])

        state = np.stack((reset_position, w_0_array, elevations_norm, reset_mitigation))

        return state

    def _compare_states(self, fire_map: np.ndarray,
                        fire_map_with_agent: np.ndarray) -> float:
        '''
        Calculate the reward for the agent's actions.

        Possible `fire_map` values:

        | Burn Status | Value |
        |-------------|-------|
        | Unburned    | 0     |
        | Burning     | 1     |
        | Burned      | 2     |
        | Fireline    | 3     |
        | Scratchline | 4     |
        | Wetline     | 5     |

        Reward is determined by:
        ```
            firemap_with_agent: [2] -> reward = -1
            firemap_with_agent: [3] -> reward = -1
            firemap: [2] firemap_with_agent: [0] -> reward = +1
            else 0
        ```

        Arguments:
            fire_map: `np.ndarray` with possible values [0, 2]
            fire_map_with_agent: `np.ndarray` with possible values [0, 2, 3]

        Returns:
            score / difference between the firemaps (normalized with respect to screen
            size).
        '''
        reward = 0
        mod = 0
        unmod = 0
        for x in range(self.config.area.screen_size):
            for y in range(self.config.area.screen_size):
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

        return reward / (self.config.area.screen_size * self.config.area.screen_size)


class RLGridEnv(gym.Env):
    '''
    This Environment Catcher will record all environments/actions for the RL return
    states.

    This includes: Observation, Reward (or Penalty), "done", and any meta-data.

    This class will incorporate all gamelogic, input and rendering.

    It will also incorporate everything related to pygame into the `render()` method and
    separate `__init__()` and `init_render()` methods.

    We can then render ML routines using the step() and reset() method w/o loading the
    pygame package each step - if the environment is loaded, the rendering is not needed
    for training (saves execution time).

    Observation:
    ------------
    Type: gym.spaces.Dict(Box(low=min, high=max, shape=(255,255,len(max))))

    | Number | Observation | min | max |
    |--------|-------------|-----|-----|
    | 0      | Position    | 0   | 1   |
    | 1      | Fuel (w_0)  | 0   | 1   |
    | 2      | Elevation   | 0   | 1.0 |
    | 3      | Mitigation  | 0   | 1   |

    In Reactive Case:
    -----------------
    Type: gym.spaces.Dict(Box(low=min, high=max, shape=(255,255,len(max))))

    | Number | Observation | min | max |
    |--------|-------------|-----|-----|
    | 0      | Position    | 0   | 1   |
    | 1      | Fuel (w_0)  | 0   | 1   |
    | 2      | Elevation   | 0   | 1   |
    | 3      | Mitigation  | 0   | 1   |

    Actions:
    --------
    Type: Discrete(4) -- real-valued (on / off)

    | Num | Action            |
    |-----|-------------------|
    | 0   | None              |
    | 1   | Fireline          |
    | 2   | Scratchline (TBD) |
    | 3   | Wetline (TBD)     |


    Reward:
    -------
    - Reward of 0 when 'None' action is taken and position is not the last tile.
    - Reward of -1 when 'Trench, ScratchLine, WetLine' action is taken and agent
      position is not the last tile.
    - Reward of (fire_burned - fireline_burned) when done.

    Starting State:
    ---------------
    The position of the agent always starts in the top right corner (0,0).

    Episode Termination:
    ---------------------
    The agent has traversed all pixels (screen_size, screen_size)
    '''
    def __init__(self, simulation: FireLineGridEnv) -> None:
        '''
        Initialize the class by recording the state space

        Using a configurable terrain and fire start position, compare the two state
        spaces.

        Need to step through the state space twice:

          1. Let the agent step through the space and draw firelines
          2. Let the environemnt progress w/o agent

        Initialize the observation space and state space as described in the docstrings.

        Arguments:
            simulation: A `FireLineGridEnv` environment that will be used by the agent.

        Returns:
            None
        '''

        self.simulation = simulation
        self.action_space = gym.spaces.Box(0,
                                           1,
                                           shape=(self.simulation.config.screen_size,
                                                  self.simulation.config.screen_size))

        channel_lows = np.array([
            self.simulation.observ_spaces[channel][0]
            for channel in self.simulation.observ_spaces.keys()
        ])
        channel_highs = np.array([
            self.simulation.observ_spaces[channel][1]
            for channel in self.simulation.observ_spaces.keys()
        ])

        self.low = np.array(
            [[channel_lows for _ in range(self.simulation.config.screen_size)]
             for _ in range(self.simulation.config.screen_size)])

        self.high = np.array(
            [[channel_highs for _ in range(self.simulation.config.screen_size)]
             for _ in range(self.simulation.config.screen_size)])

        self.observation_space = gym.spaces.Box(
            np.float32(self.low),
            np.float32(self.high),
            shape=(self.simulation.config.screen_size, self.simulation.config.screen_size,
                   len(self.simulation.observ_spaces.keys())),
            dtype=np.float64)

        # always keep the same if terrain and fire position are static
        self.seed(1234)

    def seed(self, seed: int = None) -> List[int]:
        '''
        Set the seed for numpy random methods.

        Arguments:
            seed: Random seeding value.

        Returns:
            seed: Random seeding value.
        '''

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple[Dict[str, Tuple], int, bool, Dict[str, Any]]:
        '''
        This function will apply the action to the agent in the current state.

        Calculate new agent/state_space position by reseting old position to zero,
            call the _update_current_agent_loc() method and set new
            agent/state_space location to 1.

        Done: Occurs when the agent has traversed the entire game
                position: [screen_size, screen_size]


        Arguments:
            action: 0 for "do nothing", 1 for "do something".

        Returns:
            observation:  A tuple of the `self.state` dictionary seen below:
                {'position': (screen_size, screen_size, 1),
                 'terrain': (screen_size, screen_size, 5),
                 'elevation': (screen_size, screen_size, 1),
                 'mitigation': (screen_size, screen_size, 1)}

            reward: -1 if trench, 0 if None

            done: end simulation, calculate state differences

            info: extra meta-data
        '''
        # compare the state spaces
        fire_map = self.simulation._run(action)
        # render only agent
        if self.simulation.config.render_post_agent:
            self.simulation._render(action)

        fire_map_with_agent = self.simulation._run(action, True)
        # render fire with agents mitigation in place
        if self.simulation.config.render_post_agent_with_fire:
            self.simulation._render(action,
                                    mitigation_only=False,
                                    mitigation_and_fire_spread=True)
        reward = self.simulation._compare_states(fire_map, fire_map_with_agent)

        return self.state, reward, True, {}

    def reset(self) -> Dict[str, Tuple]:
        '''
        Reset environment to initial state.

        NOTE: reset() must be called before you can call step() for the first time.

        Terrain is received from the sim. Position matrix is assumed to be all 0's when
        received from sim. Updated to have agent at (0,0) on reset.

        Arguments:
            None

        Returns:
            `self.state`, a dictionary with the following structure:

            {'position': (screen_size, screen_size, 1),
             'terrain': (screen_size, screen_size, 5),
             'elevation': (screen_size, screen_size, 1),
             'mitigation': (screen_size, screen_size, 1)}
        '''
        self.state = self.simulation._reset_state()

        return self.state
