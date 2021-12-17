from multiprocessing import Pool
import os
import unittest
from unittest import mock

from ..game import Game
from ..sprites import Terrain
from ... import config as cfg
from ...enums import GameStatus
from ..managers.fire import RothermelFireManager
from ..managers.mitigation import FireLineManager
from ...world.parameters import Environment, FuelArray, FuelParticle, Tile
from ...world.wind import WindController


@mock.patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy'})
class TestGame(unittest.TestCase):
    def setUp(self) -> None:
        self.screen_size = cfg.screen_size

    def test_update(self) -> None:
        '''
        Test that the call to update() runs through properly. There's not much to check
        since the update method only calls sprite and manager update methods. In theory,
        if all the other unit tests pass, then this one should pass.
        '''
        game = Game(self.screen_size)

        init_pos = (cfg.screen_size // 3, cfg.screen_size // 4)
        fire_size = cfg.fire_size
        max_fire_duration = cfg.max_fire_duration
        pixel_scale = cfg.pixel_scale
        update_rate = cfg.update_rate
        fuel_particle = FuelParticle()

        tiles = [[
            FuelArray(Tile(j, i, cfg.terrain_scale, cfg.terrain_scale),
                      cfg.terrain_map[i][j]) for j in range(cfg.terrain_size)
        ] for i in range(cfg.terrain_size)]
        terrain = Terrain(tiles, cfg.elevation_fn, cfg.terrain_size, cfg.screen_size)

        wind_map = WindController()
        wind_map.init_wind_speed_generator(cfg.mw_seed, cfg.mw_scale, cfg.mw_octaves,
                                           cfg.mw_persistence, cfg.mw_lacunarity,
                                           cfg.mw_speed_min, cfg.mw_speed_max,
                                           cfg.screen_size)
        wind_map.init_wind_direction_generator(cfg.dw_seed, cfg.dw_scale, cfg.dw_octaves,
                                               cfg.dw_persistence, cfg.dw_lacunarity,
                                               cfg.dw_deg_min, cfg.dw_deg_max,
                                               cfg.screen_size)

        environment = Environment(cfg.M_f, wind_map.map_wind_speed,
                                  wind_map.map_wind_direction)

        fire_manager = RothermelFireManager(init_pos, fire_size, max_fire_duration,
                                            pixel_scale, update_rate, fuel_particle,
                                            terrain, environment)

        fireline_manager = FireLineManager(size=cfg.control_line_size,
                                           pixel_scale=cfg.pixel_scale,
                                           terrain=terrain)
        fireline_sprites = fireline_manager.sprites
        status = game.update(terrain, fire_manager.sprites, fireline_sprites,
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

        init_pos = (cfg.screen_size // 3, cfg.screen_size // 4)
        fire_size = cfg.fire_size
        max_fire_duration = cfg.max_fire_duration
        pixel_scale = cfg.pixel_scale
        update_rate = cfg.update_rate
        fuel_particle = FuelParticle()

        tiles = [[
            FuelArray(Tile(j, i, cfg.terrain_scale, cfg.terrain_scale),
                      cfg.terrain_map[i][j]) for j in range(cfg.terrain_size)
        ] for i in range(cfg.terrain_size)]
        terrain = Terrain(tiles, cfg.elevation_fn, cfg.terrain_size, cfg.screen_size)

        wind_map = WindController()
        wind_map.init_wind_speed_generator(cfg.mw_seed, cfg.mw_scale, cfg.mw_octaves,
                                           cfg.mw_persistence, cfg.mw_lacunarity,
                                           cfg.mw_speed_min, cfg.mw_speed_max,
                                           cfg.screen_size)
        wind_map.init_wind_direction_generator(cfg.dw_seed, cfg.dw_scale, cfg.dw_octaves,
                                               cfg.dw_persistence, cfg.dw_lacunarity,
                                               cfg.dw_deg_min, cfg.dw_deg_max,
                                               cfg.screen_size)

        environment = Environment(cfg.M_f, wind_map.map_wind_speed,
                                  wind_map.map_wind_direction)

        fire_manager = RothermelFireManager(init_pos,
                                            fire_size,
                                            max_fire_duration,
                                            pixel_scale,
                                            update_rate,
                                            fuel_particle,
                                            terrain,
                                            environment,
                                            headless=True)

        fireline_manager = FireLineManager(size=cfg.control_line_size,
                                           pixel_scale=cfg.pixel_scale,
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

        init_pos = (cfg.screen_size // 3, cfg.screen_size // 4)
        fire_size = cfg.fire_size
        max_fire_duration = cfg.max_fire_duration
        pixel_scale = cfg.pixel_scale
        update_rate = cfg.update_rate
        fuel_particle = FuelParticle()

        tiles = [[
            FuelArray(Tile(j, i, cfg.terrain_scale, cfg.terrain_scale),
                      cfg.terrain_map[i][j]) for j in range(cfg.terrain_size)
        ] for i in range(cfg.terrain_size)]
        terrain = Terrain(tiles,
                          cfg.elevation_fn,
                          cfg.terrain_size,
                          cfg.screen_size,
                          headless=True)

        wind_map = WindController()
        wind_map.init_wind_speed_generator(cfg.mw_seed, cfg.mw_scale, cfg.mw_octaves,
                                           cfg.mw_persistence, cfg.mw_lacunarity,
                                           cfg.mw_speed_min, cfg.mw_speed_max,
                                           cfg.screen_size)
        wind_map.init_wind_direction_generator(cfg.dw_seed, cfg.dw_scale, cfg.dw_octaves,
                                               cfg.dw_persistence, cfg.dw_lacunarity,
                                               cfg.dw_deg_min, cfg.dw_deg_max,
                                               cfg.screen_size)

        environment = Environment(cfg.M_f, wind_map.map_wind_speed,
                                  wind_map.map_wind_direction)

        fire_manager = RothermelFireManager(init_pos,
                                            fire_size,
                                            max_fire_duration,
                                            pixel_scale,
                                            update_rate,
                                            fuel_particle,
                                            terrain,
                                            environment,
                                            headless=True)

        fireline_manager = FireLineManager(size=cfg.control_line_size,
                                           pixel_scale=cfg.pixel_scale,
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
