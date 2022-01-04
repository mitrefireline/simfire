import unittest
import numpy as np
from src.rl_env.simulation import RothermalSimulation
from ...utils.config import Config
from ...rl_env.harness import RLEnvironment


class RLEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        '''
        '''

        self.config = Config('./src/rl_env/_tests/test_config.yml')
        self.actions = ['none', 'fireline']
        self.attributes = ['mitigation', 'w0', 'elevation']
        self.simulation = RothermalSimulation(self.config)
        self.rl_harness = RLEnvironment(self.simulation, self.actions, self.attributes)

        self.action = 1

        self.current_agent_loc = (1, 1)

    def test_step(self) -> None:
        '''
        Test that the call to `step()` runs through properly.

        `step()` calls a handful of sub-functions, which also get tested.
        '''
        self.rl_harness.reset()
        state, reward, done, _ = self.rl_harness.step(self.action)

        agent_pos = np.where(state[0] == 1)

        self.assertEqual(
            agent_pos, (0, 1),
            msg=(f'The returned state of agent position of the game is {agent_pos}, '
                 f'but it should be (1, 0)'))

        self.assertEqual(
            reward,
            0,
            msg=(f'The returned state of reward of the game is {reward}, but it '
                 f'should be -1'))

        self.assertEqual(done,
                         False,
                         msg=(f'The returned state of the game is {done}, but it '
                              f'should be False'))

    def test__update_current_agent_loc(self) -> None:
        '''
        Test that the call to `_update_current_agent_loc()` runs through properly.

        The agent position should only be at a single location per `step()`.
        '''

        self.rl_harness.reset()
        self.rl_harness.current_agent_loc = self.current_agent_loc
        x = self.current_agent_loc[0]
        y = self.current_agent_loc[1]
        new_agent_loc = (x, y + 1)
        self.rl_harness._update_current_agent_loc()

        self.assertEqual(
            self.rl_harness.current_agent_loc,
            new_agent_loc,
            msg=f'The returned agent location is {self.current_agent_loc}, but it '
            f'should be {new_agent_loc}.')

    def test_reset(self) -> None:
        '''
        Test that the call to `reset()` runs through properly.

        `reset()` calls a handful of sub-functions, which also get tested.

        Assert agent position is returned to upper left corner (1,0) of game.

        '''
        state = self.rl_harness.reset()

        # check agent position is reset to top corner
        agent_pos = np.where(state[0] == 1)
        self.assertEqual(
            agent_pos, (0, 0),
            msg=(f'The returned state of the agent position is {agent_pos}, but it '
                 f'should be [0, 0]'))

        # check elevation
        elevation = state[3]
        elevation_zero_min = self.simulation.terrain.elevations - \
                             self.simulation.terrain.elevations.min()
        valid_elevation = elevation_zero_min / (elevation_zero_min.max())
        self.assertTrue(
            (elevation == valid_elevation).all(),
            msg=('The returned state of the terrain elevation map is not the same '
                 'as the initialized terrain elevation map'))

    def test__reset_state(self) -> None:
        '''
        Test that the call to _reset_state() runs properly.
        `_reset_state()` calls a handful of sub-functions, which also get tested.

        '''
        state = self.rl_harness._reset_state()

        # check state only has desired attributes +position array
        harness_attributes = len(state)
        test_attributes = len(self.attributes) + 1
        self.assertEqual(
            harness_attributes,
            test_attributes,
            msg=(f'The returned state of the harness attributes is length '
                 f'{harness_attributes}, but it should be length {test_attributes}'))

    def test_update_current_agent_loc(self) -> None:
        '''
        Test that the call to `_update_current_agent_loc()` runs through properly.

        The agent position should only be at a single location per `step()`.
        '''

        self.rl_harness.reset()
        self.rl_harness.current_agent_loc = self.current_agent_loc
        x = self.current_agent_loc[0]
        y = self.current_agent_loc[1]
        new_agent_loc = (x, y + 1)
        self.rl_harness._update_current_agent_loc()

        self.assertEqual(
            self.rl_harness.current_agent_loc,
            new_agent_loc,
            msg=f'The returned agent location is {self.current_agent_loc}, but it '
            f'should be {new_agent_loc}.')

    def test__compare_states(self) -> None:
        '''
        Test that the call to `_compare_states()` runs the comparison of state spaces
        properly.

        This function returns the overall reward.
        '''
        screen_size = self.simulation.config.area.screen_size
        # create array of BURNED pixels
        fire_map = np.full((screen_size, screen_size), 2)
        # create array of agent mitigation + fire spread (BURNED pixels)
        fire_map_with_agent = np.full((screen_size, screen_size), 3)

        unmodified_reward = -1 * (screen_size * screen_size)
        modified_reward = -1 * (screen_size * screen_size)
        test_reward = (modified_reward - unmodified_reward) / \
                        (screen_size * screen_size)
        reward = self.rl_harness._compare_states(fire_map, fire_map_with_agent)

        # assert rewards are the same
        self.assertEqual(reward,
                         test_reward,
                         msg=(f'The returned reward of the game is {reward}, but it '
                              f'should be {test_reward}'))
