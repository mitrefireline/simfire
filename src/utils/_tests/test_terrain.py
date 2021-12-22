import unittest
import numpy as np

from src import config
from src.utils.terrain import chaparral, random_seed_list
from ...world.parameters import Fuel


class ChaparralTest(unittest.TestCase):
    def setUp(self) -> None:
        '''


        '''
        self.config = config
        self.config.terrain_size = 5

    def test_chaparral(self) -> None:
        '''
        Test that the seeding runs propoerly when creating a terrain map.

        '''
        # assert the same seed
        seed = 1111
        np.random.seed(seed)
        fuel = Fuel(w_0=np.random.uniform(.2, .6), delta=6.000, M_x=0.2000, sigma=1739)
        chaparral_fuel = chaparral(seed=seed)

        self.assertEqual(fuel.w_0,
                         chaparral_fuel.w_0,
                         msg='The seed value should produce the '
                         'same Fuel map.')

        # assert a different seed
        np.random.seed(seed)
        fuel = Fuel(w_0=np.random.uniform(.2, .6), delta=6.000, M_x=0.2000, sigma=1739)
        chaparral_fuel = chaparral()

        self.assertNotEqual(fuel.w_0,
                            chaparral_fuel.w_0,
                            msg='The seed value should produce a '
                            'different Fuel map.')


class RandomSeedListTest(unittest.TestCase):
    def setUp(self) -> None:
        self.length = 3

    def test_random_seed_list(self) -> None:
        seed = 1111
        np.random.seed(seed)
        terrain_random_seed_tuple = tuple(
            tuple(np.random.randint(0, 99999) for _ in range(self.length))
            for _ in range(self.length))
        test_terrain_random_seed_tuple = random_seed_list(self.length, seed=seed)

        self.assertEqual(terrain_random_seed_tuple,
                         test_terrain_random_seed_tuple,
                         msg='Randomly generated tuple should be the same.')

        # remove seed and assert the tuple of tuples is not the same (random)
        test_terrain_random_seed_tuple = random_seed_list(self.length)
        self.assertNotEqual(terrain_random_seed_tuple,
                            test_terrain_random_seed_tuple,
                            msg='Randomly generated tuple should be the same.')
