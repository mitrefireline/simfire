import unittest
import numpy as np
from ...rl_env.simulation import RothermelSimulation
from ...utils.config import Config
from ...rl_env.harness import AgentBasedHarness


class AgentBasedHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        '''
        '''

        self.config = Config('./src/rl_env/_tests/test_config.yml')
        self.actions = ['none', 'fireline']
        self.attributes = ['position', 'mitigation', 'w0', 'elevation']
        self.simulation = RothermelSimulation(self.config)
        self.rl_harness = AgentBasedHarness(self.simulation, self.actions,
                                            self.attributes)
        self.mitigation_map = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

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

        Assert agent position is returned to upper left corner (0,0) of game.

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
            msg=(f'The returned agent location is {self.current_agent_loc}, but it '
                 f'should be {new_agent_loc}.'))

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

    def test_add_nonsim_min_max(self) -> None:
        """
        Test that the non-simulation attributes min/max values are added.
        """
        min_maxes, _ = self.rl_harness.simulation_conversion([])
        self.rl_harness.min_maxes = min_maxes

        self.assertTrue('position' not in self.rl_harness.min_maxes,
                        msg=(f'Position should not be in to min_maxes'
                             f' but is: {self.rl_harness.min_maxes}'))

        self.assertTrue('mitigation' not in self.rl_harness.min_maxes,
                        msg=(f'Mitigation should not be in to min_maxes'
                             f' but is: {self.rl_harness.min_maxes}'))

        self.rl_harness.add_nonsim_min_max()

        self.assertTrue('position' in self.rl_harness.min_maxes,
                        msg=(f'Position should be added to min_maxes'
                             f' but is not in {self.rl_harness.min_maxes}'))

        self.assertTrue(self.rl_harness.min_maxes['position'] == (0, 1),
                        msg=(f'Position min_max should be (0,1) but is '
                             f'{self.rl_harness.min_maxes["position"]}'))

        self.assertTrue('mitigation' in self.rl_harness.min_maxes,
                        msg=(f'Mitigation should be added to min_maxes '
                             f'but is not in {self.rl_harness.min_maxes}'))

        self.assertTrue(
            self.rl_harness.min_maxes['mitigation'] == (0, len(self.rl_harness.actions)),
            msg=(
                f'Mitigation min_max should be (0,{len(self.rl_harness.actions)}) but is '
                f'{self.rl_harness.min_maxes["mitigation"]}'))

        self.assertTrue(
            'w0' in self.rl_harness.min_maxes,
            msg=(f'w0 should be in to min_maxes but is not: {self.rl_harness.min_maxes}'))

        self.assertTrue('elevation' in self.rl_harness.min_maxes,
                        msg=(f'elevation should be in to min_maxes '
                             f'but is not: {self.rl_harness.min_maxes}'))

    def test_add_nonsim_attributes(self):
        """
        Test that the non-simulation attribute channels are created.
        """
        res = self.rl_harness.add_nonsim_attributes()
        self.assertTrue('position' in res,
                        msg=(f'Position should be added to res but is not in {res}'))

        self.assertTrue('mitigation' in res,
                        msg=(f'Mitigation should be added to res but is not in {res}'))
        self.assertTrue(
            len(res) == 2,
            msg=(
                f'Mitigation and Position should be the only attributes but found {res}'))

    def test_harness_conversion(self) -> None:
        '''
        Test that the harness_conversion() function correctly assigns the RL Harness
            action with the Simulation.attributes BurnStatus.
        '''
        self.rl_harness.actions = ['none', 'scratchline']
        sim_mitigation_map = np.array([[0, 4, 0], [4, 0, 4], [0, 4, 0]])

        test_sim_mitigation_map = self.rl_harness.harness_conversion(self.mitigation_map)

        compare = sim_mitigation_map == test_sim_mitigation_map
        self.assertTrue(compare.all(),
                        msg='The output Simulation mitigation map should be '
                        'corrected for the correct BurnStatus')

    def test_simulation_conversion(self) -> None:
        '''
        Test that the simulation_conversion() function correctly converts all
        attributes from the simulation to ones the harness can use.
        '''
        converted_attributes = np.stack((np.zeros((2, 2)), np.zeros((2, 2))))

        test_converted_attributes = self.rl_harness.simulation_conversion([])
        # test that incorrect attributes are not inlcuded
        self.assertEqual(len(converted_attributes),
                         len(test_converted_attributes),
                         msg='The output conversion should only have '
                         f'{len(converted_attributes)} attributes, but has '
                         f'{len(test_converted_attributes)} attributes.')
