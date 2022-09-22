import unittest
from pathlib import Path
from typing import Callable

from ...utils.config import Config
from ..elevation_functions import ElevationFn, flat, gaussian, perlin


class TestElevationFunctions(unittest.TestCase):
    def setUp(self) -> None:
        """Load the config"""
        self.yaml = Path("./simfire/utils/_tests/test_configs/test_config.yml")
        self.cfg = Config(self.yaml)
        return super().setUp()

    def test_flat(self) -> None:
        """Test that the call to flat() returns the correct Callable and value"""
        fn = flat()

        self.assertIsInstance(
            fn,
            Callable,
            msg="The retuned function should be of type "
            f"{ElevationFn}, but is of type {type(fn)}",
        )

        z = fn(0, 0)
        z_valid = 0
        self.assertAlmostEqual(
            z,
            z_valid,
            msg=f"The returned value should be {z_valid}, " f"but is actually {z}",
        )

    def test_gaussian(self) -> None:
        """Test that the call to gaussian() returns the correct Callable and value"""
        amplitude = 1
        mu_x = 1
        mu_y = 1
        sigma_x = 1
        sigma_y = 1
        fn = gaussian(amplitude, mu_x, mu_y, sigma_x, sigma_y)

        self.assertIsInstance(
            fn,
            Callable,
            msg="The retuned function should be of type "
            f"{ElevationFn}, but is of type {type(fn)}",
        )

        z = fn(0, 0)
        z_valid = 0.6065306597126334
        self.assertAlmostEqual(
            z,
            z_valid,
            msg=f"The returned value should be {z_valid}, " f"but is actually {z}",
        )

    def test_perlin(self) -> None:
        """Test that the call to perlin() returns the correct Callable and value"""
        octaves = 1
        persistence = 0.5
        lacunarity = 2.0
        seed = 827
        range_min = 100
        range_max = 300
        fn = perlin(octaves, persistence, lacunarity, seed, range_min, range_max)

        self.assertIsInstance(
            fn,
            Callable,
            msg="The retuned function should be of type "
            f"{ElevationFn}, but is of type {type(fn)}",
        )

        z = fn(0, 0)
        z_valid = 188.19449469447136
        self.assertAlmostEqual(
            z,
            z_valid,
            msg=f"The returned value should be {z_valid}, " f"but is actually {z}",
        )
