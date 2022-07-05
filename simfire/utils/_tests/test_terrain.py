import unittest

import numpy as np

from ...world.parameters import Fuel
from ..config import Config
from ..terrain import chaparral, random_seed_list, w_0_seed


class TerrainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.length = 3

    def test_chaparral(self) -> None:
        """
        Test that the seeding runs propoerly when creating a terrain map.
        """
        # assert the same seed
        seed = 1111
        np.random.seed(seed)
        fuel = Fuel(w_0=np.random.uniform(0.2, 0.6), delta=6.000, M_x=0.2000, sigma=1739)
        chaparral_fuel = chaparral(seed=seed)

        self.assertEqual(
            fuel.w_0,
            chaparral_fuel.w_0,
            msg="The seed value should produce the " "same Fuel map.",
        )

        # assert a different seed
        np.random.seed(seed)
        fuel = Fuel(w_0=np.random.uniform(0.2, 0.6), delta=6.000, M_x=0.2000, sigma=1739)
        chaparral_fuel = chaparral()

        self.assertNotEqual(
            fuel.w_0,
            chaparral_fuel.w_0,
            msg="The seed value should produce a " "different Fuel map.",
        )

    def test_w_0_seed(self) -> None:
        """
        Test creating a random float value for w_0 (moisture content)
        """
        seed = 1111
        w_0 = w_0_seed(seed)
        self.assertIsInstance(w_0, float)
        # If this range changes in the function itself, update this test
        self.assertGreaterEqual(w_0, 0.2)
        self.assertLessEqual(w_0, 0.6)

    def test_random_seed_list(self) -> None:
        """
        Test that random seed lists are created correctly
        """
        seed = 1111
        np.random.seed(seed)
        terrain_random_seed_tuple = tuple(
            tuple(np.random.randint(0, 99999) for _ in range(self.length))
            for _ in range(self.length)
        )
        test_terrain_random_seed_tuple = random_seed_list(self.length, seed=seed)

        self.assertEqual(
            terrain_random_seed_tuple,
            test_terrain_random_seed_tuple,
            msg="Randomly generated tuple should be the same.",
        )

        # remove seed and assert the tuple of tuples is not the same (random)
        test_terrain_random_seed_tuple = random_seed_list(self.length)
        self.assertNotEqual(
            terrain_random_seed_tuple,
            test_terrain_random_seed_tuple,
            msg="Randomly generated tuple should be the same.",
        )
