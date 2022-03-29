import pygame

import os
import unittest
from unittest import mock
from multiprocessing import get_context

from ..game import Game
from ..sprites import Terrain
from ...enums import GameStatus
from ...utils.config import Config
from ..managers.fire import RothermelFireManager
from ..managers.mitigation import FireLineManager
from ...world.parameters import Environment, FuelParticle


@mock.patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy'})
class TestGame(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config('./config.yml')
        self.screen_size = self.config.area.screen_size
        self.game = Game(self.screen_size)

    def test__toggle_wind_magnitude_display(self) -> None:
        '''
        Test that when function is called, `game.show_wind_magnitude` is inverted
        '''
        self.game.show_wind_magnitude = False
        self.game._toggle_wind_magnitude_display()
        self.assertTrue(self.game.show_wind_magnitude,
                        msg='Game().show_wind_magnitude was not toggled from False to '
                        'True')
        self.game._toggle_wind_magnitude_display()
        self.assertFalse(self.game.show_wind_magnitude,
                         msg='Game().show_wind_magnitude was not toggled from True to '
                         'False')

    def test__toggle_wind_direction_display(self) -> None:
        '''
        Test that when function is called, `game.show_wind_direction` is inverted
        '''
        self.game.show_wind_direction = False
        self.game._toggle_wind_direction_display()
        self.assertTrue(self.game.show_wind_direction,
                        msg='Game().show_wind_direction was not toggled from False to '
                        'True')
        self.game._toggle_wind_direction_display()
        self.assertFalse(self.game.show_wind_direction,
                         msg='Game().show_wind_direction was not toggled from True to '
                         'False')

    def test__disable_wind_magnitude_display(self) -> None:
        '''
        Test that when function is called, `game.show_wind_magnitude` is disabled
        '''
        self.game.show_wind_magnitude = True
        self.game._disable_wind_magnitude_display()
        self.assertFalse(self.game.show_wind_magnitude,
                         msg='Game().show_wind_magnitude was not disabled and changed '
                         'from True to False')

    def test__disable_wind_direction_display(self) -> None:
        '''
        Test that when function is called, `game.show_wind_direction` is disabled
        '''
        self.game.show_wind_direction = True
        self.game._disable_wind_direction_display()
        self.assertFalse(self.game.show_wind_direction,
                         msg='Game().show_wind_direction was not disabled and changed '
                         'from True to False')

    def test__get_wind_direction_color(self) -> None:
        '''
        Test getting the color of the wind direction
        '''
        # North
        direction = 0.0
        rgb = (255, 0, 0)
        returned_rgb = self.game._get_wind_direction_color(direction)
        self.assertEqual(rgb,
                         returned_rgb,
                         msg=f'Direction angle of {direction} should return color of '
                         f'{rgb} when {returned_rgb} was returned')
        # East
        direction = 90.0
        rgb = (128, 255, 0)
        returned_rgb = self.game._get_wind_direction_color(direction)
        self.assertEqual(rgb,
                         returned_rgb,
                         msg=f'Direction angle of {direction} should return color of '
                         f'{rgb} when {returned_rgb} was returned')
        # South
        direction = 180.0
        rgb = (0, 255, 255)
        returned_rgb = self.game._get_wind_direction_color(direction)
        self.assertEqual(rgb,
                         returned_rgb,
                         msg=f'Direction angle of {direction} should return color of '
                         f'{rgb} when {returned_rgb} was returned')
        # West
        direction = 270.0
        rgb = (128, 0, 0)
        returned_rgb = self.game._get_wind_direction_color(direction)
        self.assertEqual(rgb,
                         returned_rgb,
                         msg=f'Direction angle of {direction} should return color of '
                         f'{rgb} when {returned_rgb} was returned')

    def test__get_wind_mag_surf(self) -> None:
        '''
        Test getting the wind magnitude PyGame surface
        '''
        surface = self.game._get_wind_mag_surf(self.config.wind.speed)
        surface_size = surface.get_size()
        config_size = (self.config.area.screen_size, self.config.area.screen_size)
        self.assertIsInstance(surface,
                              pygame.Surface,
                              msg='The object returned from Game()._get_wind_mag_surf '
                              f'is a {type(surface)} when it should be a pygame.Surface')
        self.assertEqual(surface_size,
                         config_size,
                         msg='The size of the surface returned in '
                         f'Game()._get_wind_mag_surf is {surface_size} when it should be '
                         f'{config_size}')

    def test__get_wind_dir_surf(self) -> None:
        '''
        Test getting the wind direction PyGame surface
        '''
        surface = self.game._get_wind_dir_surf(self.config.wind.direction)
        surface_size = surface.get_size()
        config_size = (self.config.area.screen_size, self.config.area.screen_size)
        self.assertIsInstance(surface,
                              pygame.Surface,
                              msg='The object returned from Game()._get_wind_dir_surf '
                              f'is a {type(surface)} when it should be a pygame.Surface')
        self.assertEqual(surface_size,
                         config_size,
                         msg='The size of the surface returned in '
                         f'Game()._get_wind_dir_surf is {surface_size} when it should be '
                         f'{config_size}')

    def test_update(self) -> None:
        '''
        Test that the call to update() runs through properly. There's not much to check
        since the update method only calls sprite and manager update methods. In theory,
        if all the other unit tests pass, then this one should pass.
        '''
        init_pos = (self.config.area.screen_size // 3, self.config.area.screen_size // 4)
        fire_size = self.config.display.fire_size
        max_fire_duration = self.config.fire.max_fire_duration
        pixel_scale = self.config.area.pixel_scale
        update_rate = self.config.simulation.update_rate
        fuel_particle = FuelParticle()

        tiles = [[
            self.config.terrain.fuel_array_function(x, y)
            for x in range(self.config.area.terrain_size)
        ] for y in range(self.config.area.terrain_size)]
        terrain = Terrain(tiles, self.config.terrain.elevation_function,
                          self.config.area.terrain_size, self.config.area.screen_size)

        environment = Environment(self.config.environment.moisture,
                                  self.config.wind.speed, self.config.wind.direction)

        fire_manager = RothermelFireManager(init_pos, fire_size, max_fire_duration,
                                            pixel_scale, update_rate, fuel_particle,
                                            terrain, environment)

        fireline_manager = FireLineManager(size=self.config.display.control_line_size,
                                           pixel_scale=self.config.area.pixel_scale,
                                           terrain=terrain)
        fireline_sprites = fireline_manager.sprites
        status = self.game.update(terrain, fire_manager.sprites, fireline_sprites,
                                  self.config.wind.speed, self.config.wind.direction)

        self.assertEqual(status,
                         GameStatus.RUNNING,
                         msg=(f'The returned status of the game is {status}, but it '
                              f'should be {GameStatus.RUNNING}'))

    def test_headless(self) -> None:
        '''
        Test that the game can run in a headless state with no PyGame assets loaded.
        This will also allow for the game to be pickle-able and used with multiprocessing.
        '''
        game = Game(self.screen_size, headless=True)

        init_pos = (self.config.area.screen_size // 3, self.config.area.screen_size // 4)
        fire_size = self.config.display.fire_size
        max_fire_duration = self.config.fire.max_fire_duration
        pixel_scale = self.config.area.pixel_scale
        update_rate = self.config.simulation.update_rate
        fuel_particle = FuelParticle()

        tiles = [[
            self.config.terrain.fuel_array_function(x, y)
            for x in range(self.config.area.terrain_size)
        ] for y in range(self.config.area.terrain_size)]
        terrain = Terrain(tiles,
                          self.config.terrain.elevation_function,
                          self.config.area.terrain_size,
                          self.config.area.screen_size,
                          headless=True)

        environment = Environment(self.config.environment.moisture,
                                  self.config.wind.speed, self.config.wind.direction)

        fire_manager = RothermelFireManager(init_pos,
                                            fire_size,
                                            max_fire_duration,
                                            pixel_scale,
                                            update_rate,
                                            fuel_particle,
                                            terrain,
                                            environment,
                                            headless=True)

        fireline_manager = FireLineManager(size=self.config.display.control_line_size,
                                           pixel_scale=self.config.area.pixel_scale,
                                           terrain=terrain,
                                           headless=True)
        fireline_sprites = fireline_manager.sprites
        status = game.update(terrain, fire_manager.sprites, fireline_sprites,
                             self.config.wind.speed, self.config.wind.direction)

        self.assertEqual(status,
                         GameStatus.RUNNING,
                         msg=(f'The returned status of the game is {status}, but it '
                              f'should be {GameStatus.RUNNING}'))

    def test_multiprocessing(self) -> None:
        '''
        Test that the game will run with a multiprocessing pool.
        This requires that all objects called are pickle-able and that the game is
        run in a headless state.
        '''
        game = Game(self.screen_size, headless=True)

        init_pos = (self.config.area.screen_size // 3, self.config.area.screen_size // 4)
        fire_size = self.config.display.fire_size
        max_fire_duration = self.config.fire.max_fire_duration
        pixel_scale = self.config.area.pixel_scale
        update_rate = self.config.simulation.update_rate
        fuel_particle = FuelParticle()

        tiles = [[
            self.config.terrain.fuel_array_function(x, y)
            for x in range(self.config.area.terrain_size)
        ] for y in range(self.config.area.terrain_size)]
        terrain = Terrain(tiles,
                          self.config.terrain.elevation_function,
                          self.config.area.terrain_size,
                          self.config.area.screen_size,
                          headless=True)

        environment = Environment(self.config.environment.moisture,
                                  self.config.wind.speed, self.config.wind.direction)

        fire_manager = RothermelFireManager(init_pos,
                                            fire_size,
                                            max_fire_duration,
                                            pixel_scale,
                                            update_rate,
                                            fuel_particle,
                                            terrain,
                                            environment,
                                            headless=True)

        fireline_manager = FireLineManager(size=self.config.display.control_line_size,
                                           pixel_scale=self.config.area.pixel_scale,
                                           terrain=terrain,
                                           headless=True)
        fireline_sprites = fireline_manager.sprites

        pool_size = 4
        inputs = (terrain, fire_manager.sprites, fireline_sprites, self.config.wind.speed,
                  self.config.wind.direction)
        inputs = [inputs] * pool_size
        with get_context('spawn').Pool(pool_size) as p:
            status = p.starmap(game.update, inputs)

        valid_status = [GameStatus.RUNNING] * pool_size
        self.assertCountEqual(status,
                              valid_status,
                              msg=(f'The returned status of the games is {status}, but '
                                   f'should be {valid_status}'))
