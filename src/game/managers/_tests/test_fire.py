import unittest

import numpy as np

from ....enums import BurnStatus, GameStatus
from ....game._tests import DummyFuelLayer, DummyTopographyLayer
from ....game.managers.mitigation import FireLineManager
from ...sprites import Fire, Terrain
from ....utils.config import Config
from ....utils.units import mph_to_ftpm
from ....world.parameters import Environment, FuelParticle
from ..fire import ConstantSpreadFireManager, FireManager, RothermelFireManager


class TestFireManager(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config('./config.yml')
        self.init_pos = (self.config.area.screen_size // 2,
                         self.config.area.screen_size // 2)
        self.fire_size = self.config.display.fire_size
        self.max_fire_duration = self.config.fire.max_fire_duration

        self.fire_manager = FireManager(self.init_pos, self.fire_size,
                                        self.max_fire_duration)

    def test_prune_sprites(self) -> None:
        '''
        Test that the sprites are pruned correctly.
        '''
        fire_map = np.full((self.config.area.screen_size, self.config.area.screen_size),
                           BurnStatus.UNBURNED)
        # Create a sprite that is past the duration
        new_sprite = Fire(self.init_pos, self.fire_size)
        sprites = [new_sprite]
        durations = [self.max_fire_duration + 1]

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
        fire_map = np.full((self.config.area.screen_size, self.config.area.screen_size),
                           BurnStatus.UNBURNED)
        # Use a location that is too large and out-of-bounds
        x, y = (self.config.area.screen_size, self.config.area.screen_size)
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
        x, y = (self.config.area.screen_size // 2, self.config.area.screen_size // 2)
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

    def test_update_rate_of_spread(self) -> None:
        '''
        Test updating the rate of spread based on locations of control lines
        '''
        fire_map = np.full((self.config.area.screen_size, self.config.area.screen_size),
                           BurnStatus.UNBURNED)
        rate_of_spread = np.zeros_like(fire_map)

        fireline_y_coords = [100, 100]
        fireline_x_coords = [100, 101]
        scratchline_y_coords = [100, 100]
        scratchline_x_coords = [102, 103]
        wetline_y_coords = [100, 100]
        wetline_x_coords = [104, 105]

        y_coords = [100, 100, 100, 100, 100, 100]
        x_coords = [100, 101, 102, 103, 104, 105]

        fire_map[fireline_y_coords, fireline_x_coords] = BurnStatus.FIRELINE
        fire_map[scratchline_y_coords, scratchline_x_coords] = BurnStatus.SCRATCHLINE
        fire_map[wetline_y_coords, wetline_x_coords] = BurnStatus.WETLINE
        rate_of_spread[y_coords, x_coords] = 1

        # First, test the True case
        self.fire_manager.attenuate_line_ros = True
        rate_of_spread_true = self.fire_manager._update_rate_of_spread(
            rate_of_spread, fire_map)
        # Check to make sure that the rate of spread was changed to a smaller value
        self.assertLess(np.sum(rate_of_spread_true), 5)

        # Then, test the False case
        self.fire_manager.attenuate_line_ros = False
        rate_of_spread_false = self.fire_manager._update_rate_of_spread(
            rate_of_spread, fire_map)
        self.assertEqual(np.sum(rate_of_spread_false), 0)


class TestRothermelFireManager(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config('./config.yml')
        self.screen_size = (32, 32)
        fuel_particle = FuelParticle()
        topo_layer = DummyTopographyLayer(self.screen_size)
        fuel_layer = DummyFuelLayer(self.screen_size)
        pixel_scale = 50  # Just an arbitrary number
        self.terrain = Terrain(fuel_layer, topo_layer, self.screen_size)
        # Use simple/constant wind speed
        self.wind_speed = mph_to_ftpm(1)
        wind_speed_arr = np.full(self.screen_size, self.wind_speed)
        environment = Environment(self.config.environment.moisture, wind_speed_arr,
                                  self.config.wind.simple.direction)
        self.fireline_manager = FireLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=pixel_scale,
            terrain=self.terrain)
        self.fire_init_pos = (self.screen_size[0] // 2, self.screen_size[1] // 2)
        self.fire_manager = RothermelFireManager(self.fire_init_pos,
                                                 self.config.display.fire_size,
                                                 self.config.fire.max_fire_duration,
                                                 pixel_scale,
                                                 self.config.simulation.update_rate,
                                                 fuel_particle,
                                                 self.terrain,
                                                 environment,
                                                 max_time=self.config.simulation.runtime)

    def test_update(self) -> None:
        '''
        Test that the RothermelFireManager will update correctly. There is no need to
        check the Rothermel rate of spread calculation since that has its own unit test.
        Instead, check that the fire will spread correctly once enough time has passed.
        '''
        # Create simulation parameters that will guarantee fire spread
        fire_map = np.full_like(self.terrain.fuels, BurnStatus.UNBURNED)
        self.fire_manager.pixel_scale = 0
        new_locs = self.fire_manager._get_new_locs(self.fire_init_pos[0],
                                                   self.fire_init_pos[1], fire_map)
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

        # Output the draw graph for QA
        fig = self.fire_manager.draw_spread_graph()
        fig.savefig('./assets/fire_spread_graph.png')


class TestConstantSpreadFireManager(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config('./config.yml')
        self.init_pos = (self.config.area.screen_size // 5,
                         self.config.area.screen_size // 7)
        self.fire_size = self.config.display.fire_size
        self.max_fire_duration = self.config.fire.max_fire_duration
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
        fire_map = np.zeros((self.config.area.screen_size, self.config.area.screen_size))
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
