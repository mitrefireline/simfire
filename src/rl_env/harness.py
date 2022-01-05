import gym
from copy import deepcopy
from typing import Dict, Tuple, List
import numpy as np
from abc import ABC, abstractmethod

from .simulation import Simulation
from .simulation import RothermelSimulation
from .harness_utils import (SimulationConversion, ActionsToInt, HarnessConversion)


class RLHarness(gym.Env, ABC):
    @abstractmethod
    def __init__(self, simulation: Simulation, actions: List[str],
                 attributes: List[str]) -> None:
        self.simulation = simulation
        self.actions = actions
        self.attributes = attributes

    @abstractmethod
    def step(action) -> Tuple[gym.spaces.Box, float, bool, Dict]:
        pass

    @abstractmethod
    def reset() -> gym.spaces.Box:
        pass


class AgentBasedHarness(RLHarness):
    '''

    This Environment Catcher will record all environments/actions for the RL return
    states.

    This includes: Observation, Reward (or Penalty), "done", and any meta-data.

    We can then render ML routines using the step() and reset() method w/o loading the
    pygame package each step - if the environment is loaded, the rendering is not needed
    for training (saves execution time).

    Observation:
    ------------
    Type: gym.spaces.Dict(Box(low=min, high=max, shape=(255,255,len(max))))

    | Number | Observation | min | max |
    |--------|-------------|-----|-----|
    | 0      | Position    | 0   | 1   |
    | 1      | Mitigation  | 0   | 1   |
    | 2      | w0          | 0   | 1   |
    | 3      | Elevation   | 0   | 1   |
    | 4      | wind speed  | 0   | 1   |
    | 5      | wind dir    | 0   | 1   |


    Actions:
    --------
    Type: Discrete(4) -- real-valued (on / off)

    | Num                   | Action            |
    |-----                  |-------------------|
    | BurnStatus.UNBURNED   | none              |
    | BurnSatus.FIRELINE    | fireline          |
    | BurnSatus.SCRATCHLINE | scratchline       |
    | BurnSatus.WETLINE     | wetline           |


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
    def __init__(self, simulation: RothermelSimulation, actions: List[str],
                    attributes: List[str]) -> None:
        '''
        Initialize the class by recording the state space

        Using a configurable terrain and fire start position, compare the two state
        spaces.

        Need to step through the state space twice:

          1. Let the agent step through the space and draw firelines
          2. Let the environemnt progress w/o agent

        NOTE: self.state will be in SAME order of `actions` list provided

        Arguments:
            simulation: Simulation()
                A Simulation() base class
            actions: List[str]
                The actions the RL harness will be recording
            attributes: List[str]
                The attributes that the RL harness will inlcude in the
                    obersvation space

        Returns:
            None
        '''
        super().__init__(simulation, actions, attributes)

        self.observ_spaces = {'position': (0, 1)}
        self.observ_spaces.update({attribute: (0, 1) for attribute in self.attributes})

        channel_lows = np.array([[[self.observ_spaces[channel][0]]]
                                 for channel in self.observ_spaces.keys()])
        channel_highs = np.array([[[self.observ_spaces[channel][1]]]
                                  for channel in self.observ_spaces.keys()])

        self.low = np.repeat(np.repeat(channel_lows,
                                       self.simulation.config.area.screen_size,
                                       axis=1),
                             self.simulation.config.area.screen_size,
                             axis=2)

        self.high = np.repeat(np.repeat(channel_highs,
                                        self.simulation.config.area.screen_size,
                                        axis=1),
                              self.simulation.config.area.screen_size,
                              axis=2)

        self.observation_space = gym.spaces.Box(
            np.float32(self.low),
            np.float32(self.high),
            shape=(len(self.observ_spaces.keys()),
                   self.simulation.config.area.screen_size,
                   self.simulation.config.area.screen_size),
            dtype=np.float64)

        self.actions_as_ints = ActionsToInt(self.actions)
        self.action_space = gym.spaces.Box(
            min(self.actions_as_ints),
            max(self.actions_as_ints),
            shape=(self.simulation.config.area.screen_size,
                   self.simulation.config.area.screen_size))

    def step(self, action) -> Tuple[gym.spaces.Box, float, bool, Dict]:
        '''
        This function will apply the action to the agent in the current state.

        Calculate new agent/state_space position by reseting old position to zero, call
        the `_update_current_agent_loc()` method and set new `agent`/`state_space`
        location to 1.

        `done` occurs when the agent has traversed the entire game position:
        `[screen_size, screen_size]`

        Arguments:
            action: A dictionary of action location.

        Returns:
            A tuple of the `self.state` dictionary seen below:

            {'position': (screen_size, screen_size),
             'mitigation': (screen_size, screen_size),
             'w0': (screen_size, screen_size),
             'elevation': (screen_size, screen_size),
             'wind_speed': (screen_size, screen_size),
             'wind_direction': (screen_size, screen_size)
             }

            The reward for the step, whether or not the simulation is `done`, and a
            dictionary containing metadata.
        '''
        # Make the mitigation action on the env at agent's current location
        self.state[1][self.current_agent_loc] = action

        # make a copy
        old_loc = deepcopy(self.current_agent_loc)

        # if we want to render as we step through the game
        if self.simulation.config.render.inline:
            # convert mitigation map to correct simulation format
            sim_mitigation_map = HarnessConversion(self.state[1], self.sim_actions,
                                                   self.actions)
            self.simulation.render(sim_mitigation_map[self.current_agent_loc],
                                   self.current_agent_loc,
                                   inline=True)

        # set the old position back to 0
        self.state[0][self.current_agent_loc] = 0

        # update position
        self._update_current_agent_loc()
        self.state[0][self.current_agent_loc] = 1

        reward = 0  # if action == 0 else

        # If the agent is at the last location (cannot move forward)
        done = old_loc == self.current_agent_loc
        if done:
            # convert mitigation map to correct simulation format
            sim_mitigation_map = HarnessConversion(self.state[1], self.sim_actions,
                                                   self.actions)
            # compare the state spaces
            fire_map = self.simulation.run(sim_mitigation_map, self.state[0])
            # render only agent
            if self.simulation.config.render.post_agent:
                self.simulation.render(sim_mitigation_map, self.state[0])

            fire_map_with_agent = self.simulation.run(sim_mitigation_map, self.state[0],
                                                      True)
            # render fire with agents mitigation in place
            if self.simulation.config.render.post_agent_with_fire:
                self.simulation.render(sim_mitigation_map,
                                       self.state[0],
                                       mitigation_only=False,
                                       mitigation_and_fire_spread=True)
            reward = self._compare_states(fire_map, fire_map_with_agent)

        return self.state, reward, done, {}

    def _update_current_agent_loc(self) -> None:
        '''
        This function will help update the current position as it traverses the game.

        Check if the y-axis is less than the screen-size and the x-axis is greater than
        the screen size --> y-axis += 1, x-axis = 0

        Check if the y-axis is less than the screen-size and the x-axis is greater than
        the screen size --> y-axis += 1, x-axis = 0

        Check if the x-axis is less than screen size and the y-axis is less than/equal
        to screen size --> y-axis = None, x-axis += 1

        Arguments:
            None

        Returns:
            None
        '''
        row = self.current_agent_loc[0]
        column = self.current_agent_loc[1]

        # If moving forward one would bring us out of bounds and we can move to new row
        if column + 1 > (self.simulation.config.area.screen_size -
                         1) and row + 1 <= (self.simulation.config.area.screen_size - 1):
            column = 0
            row += 1

        # If moving forward keeps us in bounds
        elif column + 1 <= (self.simulation.config.area.screen_size - 1):
            column += 1

        self.current_agent_loc = (row, column)

    def reset(self) -> gym.spaces.Box:
        '''
        Reset environment to initial state.

        NOTE: reset() must be called before you can call step() for the first time.

        Terrain is received from the sim. Position matrix is assumed to be all 0's when
        received from sim. Updated to have agent at (0,0) on reset.

        Arguments:
            None

        Returns:
            `self.state`, a dictionary with the following structure:

        '''
        self.state = self._reset_state()

        # Place agent at location (0,0)
        self.current_agent_loc = (0, 0)
        self.state[0][self.current_agent_loc] = 1

        return self.state

    def _reset_state(self) -> np.ndarray:
        '''
        This function will convert the initialized terrain
            to the gym.spaces.Box format.

        Arguments:
            None

        Returns:
            state: The reset state of the simulation.
        '''
        reset_position = np.zeros([
            self.simulation.config.area.screen_size,
            self.simulation.config.area.screen_size
        ])
        reset_position = np.expand_dims(reset_position, axis=0)

        self.sim_attributes = self.simulation.get_attributes()
        self.sim_actions = self.simulation.get_actions()

        observations = SimulationConversion(self.sim_attributes, self.attributes)

        state = np.vstack((reset_position, observations))
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
        for x in range(self.simulation.config.area.screen_size):
            for y in range(self.simulation.config.area.screen_size):
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

        return reward / (self.simulation.config.area.screen_size *
                         self.simulation.config.area.screen_size)
