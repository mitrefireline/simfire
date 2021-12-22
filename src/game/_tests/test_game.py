import unittest

from ..game import Game
from ..sprites import Terrain
from ...enums import GameStatus
from ...utils.config import Config
from ...world.wind import WindController
from ..managers.fire import RothermelFireManager
from ..managers.mitigation import FireLineManager
from ...world.parameters import Environment, FuelArray, FuelParticle, Tile


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
            FuelArray(
                Tile(j, i, self.config.area.terrain_scale,
                     self.config.area.terrain_scale), self.config.area.terrain_map[i][j])
            for j in range(self.config.area.terrain_size)
        ] for i in range(self.config.area.terrain_size)]
        terrain = Terrain(tiles, self.config.terrain.elevation_function,
                          self.config.area.terrain_size, self.config.area.screen_size)

        wind_map = WindController()
        wind_map.init_wind_speed_generator(
            self.config.wind.speed.seed, self.config.wind.speed.scale,
            self.config.wind.speed.octaves, self.config.wind.speed.persistence,
            self.config.wind.speed.lacunarity, self.config.wind.speed.speed_min,
            self.config.wind.speed.speed_max, self.config.wind.speed.een_size)
        wind_map.init_wind_direction_generator(
            self.config.wind.direction.seed, self.config.wind.direction.scale,
            self.config.wind.direction.octaves, self.config.wind.direction.persistence,
            self.config.wind.direction.lacunarity, self.config.wind.direction.deg_min,
            self.config.wind.direction.deg_max, self.config.wind.direction.een_size)

        environment = Environment(self.config.environment.moisture,
                                  wind_map.map_wind_speed, wind_map.map_wind_direction)

        fire_manager = RothermelFireManager(init_pos, fire_size, max_fire_duration,
                                            pixel_scale, update_rate, fuel_particle,
                                            terrain, environment)

        fireline_manager = FireLineManager(size=self.config.display.control_line_size,
                                           pixel_scale=self.config.display.pixel_scale,
                                           terrain=terrain)
        fireline_sprites = fireline_manager.sprites
        status = self.game.update(terrain, fire_manager.sprites, fireline_sprites,
                                  wind_map.map_wind_speed, wind_map.map_wind_direction)

        self.assertEqual(status,
                         GameStatus.RUNNING,
                         msg=(f'The returned status of the game is {status}, but it '
                              f'should be {GameStatus.RUNNING}'))
