import unittest
import numpy as np
from ..harness_utils import (HarnessConversion, SimulationConversion, ActionsToInt)


class HarnessConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        '''
        '''
        self.mitigation_map = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        self.sim_action = {'none': 0, 'fireline': 1, 'scratchline': 2, 'wetline': 3}
        self.harness_actions = ['none', 'scratchline']

    def test_HarnessConversion(self) -> None:
        '''
        Test that the HarnessConversion() function correctly assigns the RL Harness
            action with the Simulation.attributes BurnStatus.
        '''

        sim_mitigation_map = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]])

        test_sim_mitigation_map = HarnessConversion(self.mitigation_map, self.sim_action,
                                                    self.harness_actions)

        compare = sim_mitigation_map == test_sim_mitigation_map
        self.assertTrue(compare.all(),
                        msg='The output Simulation mitigation map should be '
                        'corrected for the correct BurnStatus')


class SimulationConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        '''
        '''
        self.sim_attributes = {
            'x': np.zeros((2, 2)),
            'y': np.zeros((2, 2)),
            'z': np.zeros((2, 2))
        }
        self.harness_attributes = ['z', 'x', 'b']

    def test_SimulationConversion(self) -> None:
        '''
        '''
        converted_attributes = np.stack((np.zeros((2, 2)), np.zeros((2, 2))))

        test_converted_attributes = SimulationConversion(self.sim_attributes,
                                                         self.harness_attributes)

        # test that incorrect attributes are not inlcuded
        self.assertEqual(len(converted_attributes),
                         len(test_converted_attributes),
                         msg='The output conversion should only have '
                         f'{len(converted_attributes)} attributes, but has '
                         f'{len(test_converted_attributes)} attributes.')


class ActionsToIntTest(unittest.TestCase):
    def setUp(self) -> None:
        '''
        '''
        self.test_actions = [0, 1, 2]

    def test_ActionsToInt(self) -> None:
        '''
        '''
        harness_actions = ['x', 'y', 'z']
        test_harness_actions = ActionsToInt(harness_actions)

        self.assertEqual(self.test_actions,
                         test_harness_actions,
                         msg=f'The output should have {len(self.test_actions)} action '
                         'possibilities.')
