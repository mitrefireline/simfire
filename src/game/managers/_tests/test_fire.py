import unittest

import numpy as np

from .... import config as cfg
from ....enums import BurnStatus, GameStatus
from ...sprites import Fire, Terrain
from ....world.elevation_functions import flat
from ....world.parameters import Environment, FuelArray, FuelParticle, Tile
from ..fire import ConstantSpreadFireManager, FireManager, RothermelFireManager
from src.world.wind import WindController


class TestFireManager(unittest.TestCase):
    def setUp(self) -> None:
        self.init_pos = (cfg.screen_size // 2, cfg.screen_size // 2)
        self.fire_size = cfg.fire_size
        self.max_fire_duration = cfg.max_fire_duration

        self.fire_manager = FireManager(self.init_pos, self.fire_size,
                                        self.max_fire_duration)

    def test_prune_sprites(self) -> None:
        '''
        Test that the sprites are pruned correctly.
        '''
        fire_map = np.full((cfg.screen_size, cfg.screen_size), BurnStatus.UNBURNED)
        # Create a sprite that is past the duration
        new_sprite = Fire(self.init_pos, self.fire_size)
        sprites = [new_sprite]
        durations = [cfg.max_fire_duration + 1]

        self.fire_manager.sprites = sprites
        self.fire_manager.durations = durations
        self.fire_manager._prune_sprites(fire_map)

        self.assertEqual(len(self.fire_manager.sprites),
                         0,
                         msg=('The fire manager did not prune the Fire sprite '
                              'with a duration greater than the max duration'))

    def test_get_new_locs(self) -> None:
        '''
        Test that new locations can be retrieved correctly.
        '''
        fire_map = np.full((cfg.screen_size, cfg.screen_size), BurnStatus.UNBURNED)
        # Use a location that is too large and out-of-bounds
        x, y = (cfg.screen_size, cfg.screen_size)
        new_locs = self.fire_manager._get_new_locs(x, y, fire_map)
        valid_locs = ((x - 1, y - 1), )
        self.assertTupleEqual(new_locs,
                              valid_locs,
                              msg=(f'The new locations: {new_locs} do not match '
                                   f'the valid locations: {valid_locs} when the '
                                   'location is too large'))

        # Use a location that is too small and out-of-bounds
        x, y = (0, 0)
        new_locs = self.fire_manager._get_new_locs(x, y, fire_map)
        valid_locs = ((x + 1, y), (x + 1, y + 1), (x, y + 1))
        self.assertTupleEqual(new_locs,
                              valid_locs,
                              msg=(f'The new locations: {new_locs} do not match '
                                   f'the valid locations: {valid_locs} when the '
                                   'location is too small'))

        # Use a location that is BURNED
        x, y = (cfg.screen_size // 2, cfg.screen_size // 2)
        fire_map[y, x + 1] = BurnStatus.BURNED
        new_locs = self.fire_manager._get_new_locs(x, y, fire_map)
        # All 8-connected points except (x+1, y)
        valid_locs = ((x + 1, y + 1), (x, y + 1), (x - 1, y + 1), (x - 1, y),
                      (x - 1, y - 1), (x, y - 1), (x + 1, y - 1))
        self.assertTupleEqual(new_locs,
                              valid_locs,
                              msg=(f'The new locations: {new_locs} do not match '
                                   f'the valid locations: {valid_locs} when a '
                                   'new location is BURNED'))


class TestRothermelFireManager(unittest.TestCase):
    def setUp(self) -> None:
        self.init_pos = (cfg.screen_size // 3, cfg.screen_size // 4)
        self.fire_size = cfg.fire_size
        self.max_fire_duration = cfg.max_fire_duration
        self.pixel_scale = cfg.pixel_scale
        self.fuel_particle = FuelParticle()

        fuel_arrs = [[
            FuelArray(Tile(j, i, cfg.terrain_scale, cfg.terrain_scale),
                      cfg.terrain_map[i][j]) for j in range(cfg.terrain_size)
        ] for i in range(cfg.terrain_size)]
        self.terrain = Terrain(fuel_arrs, flat(), cfg.terrain_size, cfg.screen_size)

        self.wind_map = WindController()
        self.wind_map.init_wind_speed_generator(cfg.mw_seed, cfg.mw_scale, cfg.mw_octaves,
                                                cfg.mw_persistence, cfg.mw_lacunarity,
                                                cfg.mw_speed_min, cfg.mw_speed_max,
                                                cfg.screen_size)
        self.wind_map.init_wind_direction_generator(cfg.dw_seed, cfg.dw_scale,
                                                    cfg.dw_octaves, cfg.dw_persistence,
                                                    cfg.dw_lacunarity, cfg.dw_deg_min,
                                                    cfg.dw_deg_max, cfg.screen_size)

        self.environment = Environment(cfg.M_f, self.wind_map.map_wind_speed,
                                       self.wind_map.map_wind_direction)

        self.fire_manager = RothermelFireManager(self.init_pos, self.fire_size,
                                                 self.max_fire_duration, self.pixel_scale,
                                                 self.fuel_particle, self.terrain,
                                                 self.environment)

    def test_update(self) -> None:
        '''
        Test that the RothermelFireManager will update correctly. There is no need to
        check the Rothermel rate of spread calculation since that has its own unit test.
        Instead, check that the fire will spread correctly once enough time has passed.
        '''
        # Create simulation parameters that will guarantee fire spread
        fire_map = np.full_like(self.terrain.fuel_arrs, BurnStatus.UNBURNED)
        self.fire_manager.pixel_scale = 0
        new_locs = self.fire_manager._get_new_locs(self.init_pos[0], self.init_pos[1],
                                                   fire_map)
        new_locs_uzip = tuple(zip(*new_locs))
        self.fire_manager.burn_amounts[new_locs_uzip] = -1

        # Update the manager and get the locations that are now burning
        # These should match new_locs since those locations are guaranteed
        # to burn with a pixel_scale of -1
        fire_map, status = self.fire_manager.update(fire_map)
        burning_locs = np.where(
            self.fire_manager.burn_amounts > self.fire_manager.pixel_scale)
        burning_locs = tuple(map(tuple, burning_locs[::-1]))
        burning_locs = tuple(zip(*burning_locs))

        self.assertEqual(status,
                         GameStatus.RUNNING,
                         msg=('The game status should be "RUNNING"'))

        self.assertCountEqual(burning_locs,
                              new_locs,
                              msg=('The locations that are now burning are: '
                                   f'{burning_locs}, but they should be: {new_locs}'))

        # Test that the RothermelFireManager sucessfully created the new Fire sprites
        # The 0th entry is the initial fire, so don't include it
        new_sprites = self.fire_manager.sprites[1:]
        num_new_sprites = len(new_sprites)
        num_new_locs = len(new_locs)

        self.assertEqual(num_new_sprites,
                         num_new_locs,
                         msg=('The RothermelFireManager should have created '
                              f'{num_new_locs} Fires, but only {num_new_sprites} were '
                              'created'))

        # Test that the RothermelFireManager created the sprites in the correct locations
        for i, sprite in enumerate(new_sprites):
            with self.subTest(i=i):
                x, y, _, _ = sprite.rect
                self.assertIn((x, y),
                              new_locs,
                              msg=(f'A sprite was created at (x, y) location ({x}, {y}), '
                                   'but that location is not valid for initital '
                                   f'location ({x}, {y}) and new locations {new_locs}'))


class TestConstantSpreadFireManager(unittest.TestCase):
    def setUp(self) -> None:
        self.init_pos = (cfg.screen_size // 5, cfg.screen_size // 7)
        self.fire_size = cfg.fire_size
        self.max_fire_duration = cfg.max_fire_duration
        self.rate_of_spread = self.max_fire_duration - 1

        self.fire_manager = ConstantSpreadFireManager(self.init_pos, self.fire_size,
                                                      self.max_fire_duration,
                                                      self.rate_of_spread)

    def test_update(self) -> None:
        '''
        Test that the ConstantSpreadFireManager will update correctly. This will
        make sure that no fires are spread before max_fire_duration updates, and that the
        correct number of new fires are created at the correct locations.
        '''
        fire_map = np.zeros((cfg.screen_size, cfg.screen_size))
        # Get the locations where Fires should be created
        new_locs = self.fire_manager._get_new_locs(self.init_pos[0], self.init_pos[1],
                                                   fire_map)

        # Update the simulation until 1 update before any spreading
        for i in range(self.rate_of_spread):
            fire_map = self.fire_manager.update(fire_map)
            # Call the fire sprite update here since it's normally handled by PyGame
            # This is needeed to increment the duration
            self.fire_manager.sprites[0].update()
            with self.subTest(update=i):
                # Verify that only the initial Fire sprite exists
                self.assertEqual(len(self.fire_manager.sprites), 1)

        # Update to make the ConstantSpreadFireManager create new Fire sprites
        self.fire_manager.update(fire_map)

        # The 0th entry is the initial fire, so don't include it
        new_sprites = self.fire_manager.sprites[1:]

        # Verify that the correct number of sprites were created
        num_new_sprites = len(new_sprites)
        num_new_locs = len(new_locs)
        self.assertEqual(num_new_sprites,
                         num_new_locs,
                         msg=('The ConstantSpreadFireManager should have created '
                              f'{num_new_locs} Fires, but {num_new_sprites} were '
                              'created'))

        sprite_locs = tuple(tuple(s.rect[:2]) for s in new_sprites)
        # Verify that the locations of the new Fire sprites are correct
        self.assertCountEqual(sprite_locs,
                              new_locs,
                              msg=('The locations that are now burning are: '
                                   f'{sprite_locs}, but they should be: {new_locs}'))
