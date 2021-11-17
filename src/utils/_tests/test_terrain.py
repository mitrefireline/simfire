import unittest
import numpy as np

from src import config
from src.utils.terrain import Chaparral
from ...world.parameters import Fuel


class ChaparralTest(unittest.TestCase):
    def setUp(self) -> None:
        '''


        '''
        self.config = config
        self.config.terrain_size = 5

    def test_Chaparral(self) -> None:
        '''
        Test that the seeding runs propoerly when creating a terrain map.

        '''
        # assert the same seed
        seed = 1111
        np.random.seed(seed)
        fuel = Fuel(w_0=np.random.uniform(.2, .6), delta=6.000, M_x=0.2000, sigma=1739)
        chaparral_fuel = Chaparral(seed=seed)

        self.assertEqual(fuel.w_0,
                         chaparral_fuel.w_0,
                         msg='The seed value should produce the '
                         'same Fuel map.')

        # assert a different seed
        np.random.seed(seed)
        fuel = Fuel(w_0=np.random.uniform(.2, .6), delta=6.000, M_x=0.2000, sigma=1739)
        chaparral_fuel = Chaparral()

        self.assertNotEqual(fuel.w_0,
                            chaparral_fuel.w_0,
                            msg='The seed value should produce a '
                            'different Fuel map.')
