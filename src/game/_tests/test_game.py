import os
import unittest
from unittest import mock
from multiprocessing import Pool

from ..game import Game
from ..sprites import Terrain
from ...enums import GameStatus
from ...utils.config import Config
from ...world.wind import WindController
from ..managers.fire import RothermelFireManager
from ..managers.mitigation import FireLineManager
from ...world.parameters import Environment, FuelParticle


@mock.patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy'})
class TestGame(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config('./config.yml')
        self.screen_size = self.config.area.screen_size
        self.game = Game(self.screen_size)

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

        wind_map = WindController()
        wind_map.init_wind_speed_generator(
            self.config.wind.speed.seed, self.config.wind.speed.scale,
            self.config.wind.speed.octaves, self.config.wind.speed.persistence,
            self.config.wind.speed.lacunarity, self.config.wind.speed.min,
            self.config.wind.speed.max, self.config.area.screen_size)
        wind_map.init_wind_direction_generator(
            self.config.wind.direction.seed, self.config.wind.direction.scale,
            self.config.wind.direction.octaves, self.config.wind.direction.persistence,
            self.config.wind.direction.lacunarity, self.config.wind.direction.min,
            self.config.wind.direction.max, self.config.area.screen_size)

        environment = Environment(self.config.environment.moisture,
                                  wind_map.map_wind_speed, wind_map.map_wind_direction)

        fire_manager = RothermelFireManager(init_pos, fire_size, max_fire_duration,
                                            pixel_scale, update_rate, fuel_particle,
                                            terrain, environment)

        fireline_manager = FireLineManager(size=self.config.display.control_line_size,
                                           pixel_scale=self.config.area.pixel_scale,
                                           terrain=terrain)
        fireline_sprites = fireline_manager.sprites
        status = self.game.update(terrain, fire_manager.sprites, fireline_sprites,
                                  wind_map.map_wind_speed, wind_map.map_wind_direction)

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
        terrain = Terrain(tiles, self.config.terrain.elevation_function,
                          self.config.area.terrain_size, self.config.area.screen_size)

        wind_map = WindController()
        wind_map.init_wind_speed_generator(
            self.config.wind.speed.seed, self.config.wind.speed.scale,
            self.config.wind.speed.octaves, self.config.wind.speed.persistence,
            self.config.wind.speed.lacunarity, self.config.wind.speed.min,
            self.config.wind.speed.max, self.config.area.screen_size)
        wind_map.init_wind_direction_generator(
            self.config.wind.direction.seed, self.config.wind.direction.scale,
            self.config.wind.direction.octaves, self.config.wind.direction.persistence,
            self.config.wind.direction.lacunarity, self.config.wind.direction.min,
            self.config.wind.direction.max, self.config.area.screen_size)

        environment = Environment(self.config.environment.moisture,
                                  wind_map.map_wind_speed, wind_map.map_wind_direction)

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
                             wind_map.map_wind_speed, wind_map.map_wind_direction)

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
        terrain = Terrain(tiles, self.config.terrain.elevation_function,
                          self.config.area.terrain_size, self.config.area.screen_size)

        wind_map = WindController()
        wind_map.init_wind_speed_generator(
            self.config.wind.speed.seed, self.config.wind.speed.scale,
            self.config.wind.speed.octaves, self.config.wind.speed.persistence,
            self.config.wind.speed.lacunarity, self.config.wind.speed.min,
            self.config.wind.speed.max, self.config.area.screen_size)
        wind_map.init_wind_direction_generator(
            self.config.wind.direction.seed, self.config.wind.direction.scale,
            self.config.wind.direction.octaves, self.config.wind.direction.persistence,
            self.config.wind.direction.lacunarity, self.config.wind.direction.min,
            self.config.wind.direction.max, self.config.area.screen_size)

        environment = Environment(self.config.environment.moisture,
                                  wind_map.map_wind_speed, wind_map.map_wind_direction)

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
        inputs = (terrain, fire_manager.sprites, fireline_sprites,
                  wind_map.map_wind_speed, wind_map.map_wind_direction)
        inputs = [inputs] * pool_size
        with Pool(pool_size) as p:
            status = p.starmap(game.update, inputs)

        valid_status = [GameStatus.RUNNING] * pool_size
        self.assertCountEqual(status,
                              valid_status,
                              msg=(f'The returned status of the games is {status}, but '
                                   f'should be {valid_status}'))
