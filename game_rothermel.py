from pathlib import Path

from skimage.draw import line

from src.game.game import Game
from src.enums import GameStatus
from src.utils.config import Config
from src.game.sprites import Terrain
from src.world.wind import WindController
from src.game.managers.fire import RothermelFireManager
from src.game.managers.mitigation import FireLineManager
from src.world.parameters import Environment, FuelArray, FuelParticle, Tile


def main():

    config_path = Path('./config.yml')

    cfg = Config(config_path)

    game = Game(cfg.area.screen_size)

    fuel_particle = FuelParticle()

    fuel_arrs = [[
        FuelArray(Tile(j, i, cfg.terrain_scale, cfg.terrain_scale), cfg.terrain_map[i][j])
        for j in range(cfg.terrain_size)
    ] for i in range(cfg.terrain_size)]
    terrain = Terrain(fuel_arrs, cfg.elevation_fn, cfg.terrain_size, cfg.screen_size)

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

    points = line(100, 15, 100, 200)
    y = points[0].tolist()
    x = points[1].tolist()
    points = list(zip(x, y))

    fireline_manager = FireLineManager(size=cfg.control_line_size,
                                       pixel_scale=cfg.pixel_scale,
                                       terrain=terrain)

    fire_map = game.fire_map
    fire_map = fireline_manager.update(fire_map, points)
    game.fire_map = fire_map

    fire_manager = RothermelFireManager(cfg.fire_init_pos, cfg.fire_size,
                                        cfg.max_fire_duration, cfg.pixel_scale,
                                        cfg.update_rate, fuel_particle, terrain,
                                        environment)

    game_status = GameStatus.RUNNING
    fire_status = GameStatus.RUNNING
    while game_status == GameStatus.RUNNING and fire_status == GameStatus.RUNNING:
        fire_sprites = fire_manager.sprites
        fireline_sprites = fireline_manager.sprites
        game_status = game.update(terrain, fire_sprites, fireline_sprites,
                                  wind_map.map_wind_speed, wind_map.map_wind_direction)
        fire_map = game.fire_map
        fire_map = fireline_manager.update(fire_map)
        fire_map, fire_status = fire_manager.update(fire_map)
        game.fire_map = fire_map


if __name__ == '__main__':
    main()
