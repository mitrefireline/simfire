import yaml
import unittest
from pathlib import Path

from numpy.testing import assert_array_almost_equal

from ...utils.units import mph_to_ftpm
from ..config import ConfigType, Config
from ...world.wind_mechanics.wind_controller import WindController


class ConfigTypeTest(unittest.TestCase):
    def setUp(self) -> None:
        yaml_dict = {
            'level_0_a': 0,
            'level_0_b': {
                'level_1_a': 1,
                'level_1_b': {
                    'level_2_a': 2,
                    'level_2_b': 3,
                    'level_2_c': '(1, 2, 3)'
                }
            }
        }
        self.cfg_type = ConfigType(**yaml_dict)

    def test__init__(self) -> None:
        '''
        Test creating a nested ConfigType class
        '''
        msg = 'The nesting of dictionary attributes did not work correctly.'

        self.assertEqual(self.cfg_type.level_0_a, 0)
        self.assertEqual(self.cfg_type.level_0_b.level_1_a,
                         1,
                         msg=f'{msg} level_0_b.level_1_a should equal {1} when '
                         f'it is set to {self.cfg_type.level_0_b.level_1_a}')
        self.assertEqual(self.cfg_type.level_0_b.level_1_b.level_2_a,
                         2,
                         msg=f'{msg} level_0_b.level_1_b.level_2_a should equal {2} when '
                         f'it is set to {self.cfg_type.level_0_b.level_1_b.level_2_a}')
        self.assertEqual(self.cfg_type.level_0_b.level_1_b.level_2_b,
                         3,
                         msg=f'{msg} level_0_b.level_1_b.level_2_b should equal {3} when '
                         f'it is set to {self.cfg_type.level_0_b.level_1_b.level_2_b}')

    def test__type(self):
        '''
        Test typing a string of a tuple into a tuple
        '''
        self.assertEqual((1, 2, 3), self.cfg_type.level_0_b.level_1_b.level_2_c)


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.yaml = Path('./src/utils/_tests/test_configs/test_config.yml')
        self.cfg = Config(self.yaml)
        self.cfg_flat_simple = Config(self.yaml.parent / 'test_config_flat_simple.yml')
        self.cfg_gaussian = Config(self.yaml.parent / 'test_config_gaussian.yml')
        with open(self.yaml, 'r') as f:
            self.data = yaml.safe_load(f)

    def test__set_attributes(self) -> None:
        '''
        Test setting all attributes from the YAML file
        '''
        self.assertEqual(self.data,
                         self.cfg.data,
                         msg=f'The YAML at {self.yaml} was loaded into the Config class '
                         'incorrectly')
        self.assertEqual(self.cfg.area.screen_size, self.data['area']['screen_size'])
        self.assertEqual(self.cfg.terrain.topography.functional.perlin.shape, (9, 9))

    def test__set_terrain_scale(self) -> None:
        '''
        Test correctly setting the terrain scale based on `pixel_scale` and `terrain_size`
        '''
        self.assertEqual(
            self.data['area']['terrain_size'] * self.data['area']['pixel_scale'],
            self.cfg.area.terrain_scale)

    def test__set_topography_layer(self) -> None:
        '''
        Test assigning elevation Python function based on a config string
        '''
        x, y = (5, 5)
        # This is the only way I could really come up with to make the correct function
        # was assigned - mdoyle
        self.assertEqual(self.cfg.terrain.topography.layer.name,
                         'perlin',
                         msg='The name for the set config terrain.topography.layer.name '
                         f'({self.cfg.terrain.topography.layer.name}) does not match '
                         '"perlin"')

        self.assertEqual(self.cfg_flat_simple.terrain.topography.layer.data[y, x],
                         0,
                         msg='The output of the flat terrain function does not equal 0')

        self.assertEqual(int(self.cfg_gaussian.terrain.topography.layer.data[y, x]),
                         333,
                         msg=f'The output of the gaussian terrain function at ({x}, {y}) '
                         'does not equal 333')

    def test__set_fuel_layer(self) -> None:
        '''
        Test assigning fuel array Python function based on config string
        '''
        self.assertEqual(self.cfg.terrain.fuel.layer.name,
                         'chaparral',
                         msg='The name for the set terrain.fuel_array_function '
                         f'({self.cfg.terrain.fuel.layer.name}) does not match '
                         '"chaparral"')

    def test__set_wind_function(self) -> None:
        '''
        Test assigning wind speed and direction arrays based on config string
        '''
        x, y = (5, 5)
        speed_min = mph_to_ftpm(self.data['wind']['perlin']['speed']['min'])
        speed_max = mph_to_ftpm(self.data['wind']['perlin']['speed']['max'])
        wind_map = WindController()
        wind_map.init_wind_speed_generator(
            self.data['wind']['perlin']['speed']['seed'],
            self.data['wind']['perlin']['speed']['scale'],
            self.data['wind']['perlin']['speed']['octaves'],
            self.data['wind']['perlin']['speed']['persistence'],
            self.data['wind']['perlin']['speed']['lacunarity'], speed_min, speed_max,
            self.data['area']['screen_size'])
        wind_map.init_wind_direction_generator(
            self.data['wind']['perlin']['direction']['seed'],
            self.data['wind']['perlin']['direction']['scale'],
            self.data['wind']['perlin']['direction']['octaves'],
            self.data['wind']['perlin']['direction']['persistence'],
            self.data['wind']['perlin']['direction']['lacunarity'],
            self.data['wind']['perlin']['direction']['min'],
            self.data['wind']['perlin']['direction']['max'],
            self.data['area']['screen_size'])
        assert_array_almost_equal(wind_map.map_wind_speed,
                                  self.cfg.wind.speed,
                                  err_msg='The speed array set by config.py:Config does '
                                  'not match the values straight from test_config.yml. '
                                  'The Config class is not loading the wind speed '
                                  'correctly.')

        assert_array_almost_equal(wind_map.map_wind_direction,
                                  self.cfg.wind.direction,
                                  err_msg='The direction array set by config.py:Config '
                                  'does not match the values straight from '
                                  'test_config.yml. The Config class is notloading the '
                                  'wind direction correctly.')

        self.assertEqual(self.cfg_flat_simple.wind.speed[y][x],
                         616,
                         msg=f'The simple wind speed at ({x}, {y}) does not equal 7 mph '
                         '(or 616.0 ft/min)')

        self.assertEqual(self.cfg_flat_simple.wind.direction[y][x],
                         90.0,
                         msg=f'The simple wind direction at ({x}, {y}) does not headed '
                         'due East')

    def test_reset_topography_layer(self) -> None:
        '''
        Test resetting the seed for the elevation function and returning a different map
        '''
        # test_config.yml has a seed of 1111
        seed = 1234
        x, y = (5, 5)
        old_elevation = self.cfg.terrain.topography.layer.data[y, x]
        self.cfg.reset_topography_layer(seed=seed)
        new_elevation = self.cfg.terrain.topography.layer.data[y, x]

        # The seeds should be updated after calling the reset method
        self.assertEqual(seed,
                         self.cfg.terrain.topography.functional.perlin.seed,
                         msg='The assigned seed of '
                         f'{self.cfg.terrain.topography.functional.perlin.seed} does '
                         f'not match the test seed of {seed}')

        # The elevation should be different at the same location now that the seed has
        # changed (most likely)
        self.assertNotEqual(old_elevation,
                            new_elevation,
                            msg='The elevation before resetting the elevation function '
                            f'({old_elevation}) and the new elevation ({new_elevation}) '
                            'should be different')

    def test_reset_fuel_layer(self) -> None:
        '''
        Test resetting the seed for the fuel array function and returning a different map
        '''
        # test_config.yml has a seed of 1111
        seed = 1234
        x, y = (5, 5)
        old_fuel = self.cfg.terrain.fuel.layer.data[y, x]
        self.cfg.reset_fuel_layer(seed=seed)
        new_fuel = self.cfg.terrain.fuel.layer.data[y, x]

        # The seeds should be updated after calling the reset method
        self.assertEqual(seed,
                         self.cfg.terrain.fuel.functional.chaparral.seed,
                         msg='The assigned seed of '
                         f'{self.cfg.terrain.fuel.functional.chaparral.seed} '
                         f'does not match the test seed of {seed}')

        # The fuel should be different at the same location now that the seed has changed
        # (most likely)
        self.assertNotEqual(old_fuel,
                            new_fuel,
                            msg='The fuel before resetting the fuel array function '
                            f'({old_fuel}) and the new fuel ({new_fuel}) '
                            'should be different')

    def test_reset_wind_function(self) -> None:
        '''
        Test resetting the seed for the wind function and returning a different map
        '''
        # test_config.yml has a speed seed of 2345
        speed_seed = 1234
        # test_config.yml has a direction seed of 650
        direction_seed = 1234
        x, y = (5, 5)

        old_wind_speed = self.cfg.wind.speed[y][x]
        old_wind_direction = self.cfg.wind.direction[y][x]
        self.cfg.reset_wind_function(speed_seed, direction_seed)
        new_wind_speed = self.cfg.wind.speed[y][x]
        new_wind_direction = self.cfg.wind.direction[y][x]

        # The seeds should be updated after calling the reset method
        self.assertEqual(speed_seed,
                         self.cfg.wind.perlin.speed.seed,
                         msg=f'The assigned seed of {self.cfg.wind.perlin.speed.seed} '
                         f'does not match the test seed of {speed_seed}')

        # The seeds should be updated after calling the reset method
        self.assertEqual(speed_seed,
                         self.cfg.wind.perlin.speed.seed,
                         msg=f'The assigned seed of {self.cfg.wind.perlin.speed.seed} '
                         f'does not match the test seed of {speed_seed}')

        # The wind speed and direction should be different at the same location now that
        # the seed has changed (most likely)
        self.assertNotEqual(old_wind_speed,
                            new_wind_speed,
                            msg='The wind speed before resetting the wind function '
                            f'({old_wind_speed}) and the new wind speed '
                            f'({new_wind_speed}) should be different')

        self.assertNotEqual(old_wind_direction,
                            new_wind_direction,
                            msg='The wind direction before resetting the wind function '
                            f'({old_wind_direction}) and the new wind direction '
                            f'({new_wind_direction}) should be different')

    def test_save(self) -> None:
        '''
        Test saving the config's data and making sure it matches the original YAML
        '''
        save_path = self.yaml.parent / 'save_config.yml'
        self.cfg.save(save_path)
        with open(save_path, 'r') as f:
            save_data = yaml.safe_load(f)
        save_path.unlink()
        self.assertEqual(self.cfg.data,
                         save_data,
                         msg=f'The data in the saved YAML at {save_path} does not match '
                         f'the data in the test YAML at {self.yaml}')
