import unittest

from ..wind_mechanics.perlin_wind import WindNoise


class TestWindNoise(unittest.TestCase):
    def setUp(self) -> None:
        self.seed: int = 2345
        self.scale: int = 400  # High values = Lower Resolution, less noisy
        self.octaves: int = 3
        self.persistence: float = 0.7
        self.lacunarity: float = 2.0
        self.range_min: float = 7
        self.range_max: float = 47
        self.screen_size = 150
        self.test_wind = WindNoise()
        self.test_wind.set_noise_parameters(
            self.seed,
            self.scale,
            self.octaves,
            self.persistence,
            self.lacunarity,
            self.range_min,
            self.range_max,
        )

    def test_generate_map_array(self) -> None:
        wind_map = self.test_wind.generate_map_array(self.screen_size)
        wind_map_row = wind_map[0]

        self.assertEqual(len(wind_map), self.screen_size)
        self.assertEqual(len(wind_map_row), self.screen_size)

        for column_list in wind_map:
            for row_value in column_list:
                self.assertTrue(row_value <= self.range_max)
                self.assertTrue(row_value >= self.range_min)

    def test_denormalize_noise_value(self) -> None:
        noise_value = 0.5

        new_value = self.test_wind._denormalize_noise_value(noise_value)
        self.assertTrue(new_value >= self.range_min)
        self.assertTrue(new_value <= self.range_max)
