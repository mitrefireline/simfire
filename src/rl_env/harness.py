import gym
from copy import deepcopy
from typing import Dict, Tuple, List
import numpy as np
from abc import ABC, abstractmethod

from .simulation import Simulation


class RLHarness(gym.Env, ABC):
    @abstractmethod
    def __init__(self, simulation: Simulation, actions: List[str],
                 attributes: List[str]) -> None:
        self.simulation = simulation
        self.actions = actions
        self.attributes = attributes

        self.sim_attributes = self.simulation.get_attributes()
        self.min_maxes, self.sim_attributes = self.simulation_conversion(['elevation'])
        self.add_nonsim_min_max()

        channel_lows = np.array([[[self.min_maxes[channel][0]]]
                                 for channel in self.attributes])
        channel_highs = np.array([[[self.min_maxes[channel][1]]]
                                  for channel in self.attributes])

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
            shape=(len(self.attributes), self.simulation.config.area.screen_size,
                   self.simulation.config.area.screen_size),
            dtype=np.float64)

        self.actions_as_ints = [int(x) for x in range(len(self.actions))]
        self.sim_actions = self.simulation.get_actions()
        self.action_space = gym.spaces.Box(
            min(self.actions_as_ints),
            max(self.actions_as_ints),
            shape=(self.simulation.config.area.screen_size,
                   self.simulation.config.area.screen_size))

    @abstractmethod
    def reset(self) -> gym.spaces.Box:
        self.sim_attributes = self.simulation.get_attributes()
        _, observations = self.simulation_conversion(['elevation'])
        observations.update(self.add_nonsim_attributes())

        obs = []
        for attribute in self.attributes:
            obs.append(observations[attribute])

        self.state = np.stack(obs, axis=0)

        return self.state

    @abstractmethod
    def step(self, action) -> Tuple[gym.spaces.Box, float, bool, Dict]:
        pass

    @abstractmethod
    def simulation_conversion(
        self, normalize_attributes: List[str]
    ) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, np.ndarray]]:
        '''
        This function will convert the returns of the Simulation.get_attributes()
            to the RL harness np.ndarray structure

        Attributes:

            normalize_attributes: List[str]
                A list of strings of the desired attributes to normalize

        Returns:
            np.ndarray
                A numpy array of the converted attributes for the RL harness to use

        '''
        def normalize(x: np.ndarray) -> np.ndarray:
            '''
            Function to normalize array to [0,1]
            '''
            norm = (x - x.min()) / (x.max() - x.min())
            return norm

        res = {}
        min_maxes = {}
        for harness_attr in self.attributes:
            if harness_attr in self.sim_attributes.keys():
                self.sim_attributes[harness_attr] = np.asarray(
                    self.sim_attributes[harness_attr])
                if harness_attr in normalize_attributes:
                    self.sim_attributes[harness_attr] = normalize(
                        self.sim_attributes[harness_attr])
                res[harness_attr] = self.sim_attributes[harness_attr]
                min_maxes[harness_attr] = (self.sim_attributes[harness_attr].min(),
                                           self.sim_attributes[harness_attr].max())

        return min_maxes, res

    @abstractmethod
    def harness_conversion(self, mitigation_map: np.ndarray) -> np.ndarray:
        '''
        This function will convert the returns of the Simulation.get_actions()
            to the RL harness List of ints structure where the simulation action
            integer starts at index 0 for the RL harness

        Example:    mitigation_map = (0, 1, 1, 0)
                    sim_action = {'none': 0, 'fireline':1, 'scratchline':2, 'wetline':3}
                    harness_actions = ['none', 'scratchline']

                Harness                  Simulation
                ---------               ------------
                'none': 0           -->   'none': 0
                'scratchline': 1    -->   'scratchline': 2

                return (0, 2, 2, 0)

        Attributes:
            mitigation_map: np.ndarray
                A np.ndarray of the harness mitigation map

        Returns:
            np.ndarray
                A np.ndarray of the converted mitigation map from RL harness
                    to the correct Simulation BurnStatus types

        '''
        harness_ints = np.unique(mitigation_map)
        harness_dict = {
            self.actions[i]: harness_ints[i]
            for i in range(len(harness_ints))
        }

        sim_mitigation_map = []
        for mitigation_i in mitigation_map:
            for mitigation_j in mitigation_i:
                action = [
                    key for key, value in harness_dict.items() if value == mitigation_j
                ]
                sim_mitigation_map.append(self.sim_actions[action[0]])

        return np.asarray(sim_mitigation_map).reshape(len(mitigation_map[0]),
                                                      len(mitigation_map[1]))

    @abstractmethod
    def add_nonsim_min_max(self) -> None:
        pass

    @abstractmethod
    def add_nonsim_attributes(self) -> Dict[str, np.ndarray]:
        return {}


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
    def __init__(self, simulation: Simulation, actions: List[str],
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

        # set the old position back to 0
        self.state[0][self.current_agent_loc] = 0

        # update position
        self._update_current_agent_loc()
        self.state[0][self.current_agent_loc] = 1

        reward = 0

        # If the agent is at the last location (cannot move forward)
        done = old_loc == self.current_agent_loc
        if done:
            # convert mitigation map to correct simulation format
            sim_mitigation_map = self.harness_conversion(self.state[1])

            # run simulation with agent actions + fire burning
            fire_map = self.simulation.run(sim_mitigation_map, False)
            fire_map_with_agent = self.simulation.run(sim_mitigation_map, True)

            # compare the state spaces
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

    def simulation_conversion(
        self, normalize_attributes: List[str]
    ) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, np.ndarray]]:

        return super().simulation_conversion(normalize_attributes)

    def harness_conversion(self, mitigation_map: np.ndarray) -> np.ndarray:

        return super().harness_conversion(mitigation_map)

    def reset(self) -> gym.spaces.Box:
        '''
        For AgentBasedHarness we need agent position as they traverse.
        Need to include in the state space with a start position in the
            upper left corner of the simulation.

        Need to override RLHarness.reset() method
        '''
        position = np.zeros((self.simulation.config.area.screen_size,
                             self.simulation.config.area.screen_size))
        position = np.expand_dims(position, axis=0)
        position[0, 0, 0] = 1
        self.current_agent_loc = (0, 0)

        self.sim_attributes = self.simulation.get_attributes()
        _, observations = self.simulation_conversion(['elevation'])
        observations.update(self.add_nonsim_attributes())

        obs = []
        for attribute in self.attributes:
            obs.append(observations[attribute])

        self.state = np.vstack((position, obs))

        return self.state

    def add_nonsim_min_max(self) -> None:
        """
        Add the min and max values for attributes not found in the simulation.

        Raises:
            ValueError: If an non-simulation attribute is not supported for this harness.
        """
        keys = self.min_maxes.keys()
        self.nonsim_attributes = []
        for attribute in self.attributes:
            if attribute not in keys:
                self.nonsim_attributes.append(attribute)
                if attribute == 'position':
                    self.min_maxes[attribute] = (0, 1)
                elif attribute == 'mitigation':
                    self.min_maxes[attribute] = (0, len(self.actions))
                else:
                    raise ValueError(f'Attribute {attribute} is not supported!')

    def add_nonsim_attributes(self) -> Dict[str, np.ndarray]:
        """
        Generate the observation channels for non-simulation attributes.

        Raises:
            ValueError: If a non-simulation attribute is not supported for this harness.

        Returns:
            Dict[str, np.ndarray]: Collection of non-simulation attributes
                and their observation channel representations.
        """
        res = {}
        for attribute in self.nonsim_attributes:
            if attribute == 'position':
                res[attribute] = np.zeros([
                    self.simulation.config.area.screen_size,
                    self.simulation.config.area.screen_size
                ])
                self.current_agent_loc = (0, 0)
                res[attribute][self.current_agent_loc] = 1
            elif attribute == 'mitigation':
                res[attribute] = np.full((self.simulation.config.area.screen_size,
                                          self.simulation.config.area.screen_size),
                                         self.sim_actions['none'])
            else:
                raise ValueError(f'Attribute {attribute} is not supported!')

        return res
