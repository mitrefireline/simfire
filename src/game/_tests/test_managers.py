import unittest

import numpy as np

from ..managers import ConstantSpreadFireManager, FireManager, RothermelFireManager
from ..sprites import Fire, Terrain
from ... import config as cfg
from ...enums import BurnStatus
from ...world.elevation_functions import flat
from ...world.parameters import Environment, FuelArray, FuelParticle, Tile


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
        # Create a sprite that is past the duration
        new_sprite = Fire(self.init_pos, self.fire_size)
        new_sprite.duration = cfg.max_fire_duration + 1
        sprites = [new_sprite]

        self.fire_manager.sprites = sprites
        self.fire_manager._prune_sprites()

        self.assertEqual(len(self.fire_manager.sprites),
                         0,
                         msg=('The fire manager did not prune the Fire sprite '
                              'with a duration greater than the max duration'))

    def test_get_new_locs(self) -> None:
        '''
        Test that new locations can be retrieved correctly.
        '''
        # Use a location that is too large and out-of-bounds
        x, y = (cfg.screen_size, cfg.screen_size)
        new_locs = self.fire_manager._get_new_locs(x, y)
        valid_locs = ((x - 1, y - 1), )
        self.assertTupleEqual(new_locs,
                              valid_locs,
                              msg=(f'The new locations: {new_locs} do not match '
                                   f'the valid locations: {valid_locs} when the '
                                   'location is too large'))

        # Use a location that is too small and out-of-bounds
        x, y = (0, 0)
        new_locs = self.fire_manager._get_new_locs(x, y)
        valid_locs = ((x + 1, y), (x + 1, y + 1), (x, y + 1))
        self.assertTupleEqual(new_locs,
                              valid_locs,
                              msg=(f'The new locations: {new_locs} do not match '
                                   f'the valid locations: {valid_locs} when the '
                                   'location is too small'))

        # Use a location that is BURNED
        x, y = (cfg.screen_size // 2, cfg.screen_size // 2)
        self.fire_manager.fire_map[y, x + 1] = BurnStatus.BURNED
        new_locs = self.fire_manager._get_new_locs(x, y)
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

        self.tiles = [[
            FuelArray(Tile(j, i, cfg.terrain_scale, cfg.terrain_scale),
                      cfg.terrain_map[i][j]) for j in range(cfg.terrain_size)
        ] for i in range(cfg.terrain_size)]
        self.terrain = Terrain(self.tiles, flat())

        self.environment = Environment(cfg.M_f, cfg.U, cfg.U_dir)

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
        self.fire_manager.pixel_scale = 0
        new_locs = self.fire_manager._get_new_locs(self.init_pos[0], self.init_pos[1])
        new_locs_uzip = tuple(zip(*new_locs))
        self.fire_manager.burn_amounts[new_locs_uzip] = -1

        # Update the manager and get the locations that are now burning
        # These should match new_locs since those locations are guaranteed
        # to burn with a pixel_scale of -1
        self.fire_manager.update()
        burning_locs = np.where(
            self.fire_manager.burn_amounts > self.fire_manager.pixel_scale)
        burning_locs = tuple(map(tuple, burning_locs[::-1]))
        burning_locs = tuple(zip(*burning_locs))

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
        # Get the locations where Fires should be created
        new_locs = self.fire_manager._get_new_locs(self.init_pos[0], self.init_pos[1])

        # Update the simulation until 1 update before any spreading
        for i in range(self.rate_of_spread):
            self.fire_manager.update()
            # Call the fire sprite update here since it's normally handled by PyGame
            # This is needeed to increment the duration
            self.fire_manager.sprites[0].update()
            with self.subTest(update=i):
                # Verify that only the initial Fire sprite exists
                self.assertEqual(len(self.fire_manager.sprites), 1)

        # Update to make the ConstantSpreadFireManager create new Fire sprites
        self.fire_manager.update()

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
