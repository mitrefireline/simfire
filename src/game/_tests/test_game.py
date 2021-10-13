import unittest

from ..Game import Game
from ..managers import RothermelFireManager
from ..sprites import Terrain
from ... import config as cfg
from ...enums import GameStatus
from ...world.parameters import Environment, FuelArray, FuelParticle, Tile


class TestGame(unittest.TestCase):
    def setUp(self) -> None:
        self.screen_size = cfg.screen_size
        self.game = Game(self.screen_size)

    def test_update(self) -> None:
        '''
        Test that the call to update() runs through properly. There's not much to check
        since the update method only calls sprite and manager update methods. In theory,
        if all the other unit tests pass, then this one should pass.
        '''
        init_pos = (cfg.screen_size // 3, cfg.screen_size // 4)
        fire_size = cfg.fire_size
        max_fire_duration = cfg.max_fire_duration
        pixel_scale = cfg.pixel_scale
        fuel_particle = FuelParticle()

        tiles = [[
            FuelArray(Tile(j, i, 0, cfg.terrain_scale, cfg.terrain_scale),
                      cfg.terrain_map[i][j]) for j in range(cfg.terrain_size)
        ] for i in range(cfg.terrain_size)]
        terrain = Terrain(tiles)

        environment = Environment(cfg.M_f, cfg.U, cfg.U_dir)

        fire_manager = RothermelFireManager(init_pos, fire_size, max_fire_duration,
                                            pixel_scale, fuel_particle, terrain,
                                            environment)

        status = self.game.update(terrain, fire_manager.sprites, fire_manager.fire_map)

        self.assertEqual(status,
                         GameStatus.RUNNING,
                         msg=(f'The returned status of the game is {status}, but it '
                              f'should be {GameStatus.RUNNING}'))
