from skimage.draw import line
from src.enums import GameStatus
from src.game.game import Game
from src.game.managers.fire import RothermelFireManager
from src.game.managers.mitigation import FireLineManager
from src.game.sprites import Terrain
from src.utils.config import Config
from src.world.parameters import Environment, FuelParticle
from src.world.wind import WindController


def main():

    cfg_path = 'config.yml'
    cfg = Config(cfg_path)

    game = Game(cfg.area.screen_size)

    fuel_particle = FuelParticle()

    fuel_arrs = [[
        cfg.terrain.fuel_array_function(x, y) for x in range(cfg.area.terrain_size)
    ] for y in range(cfg.area.terrain_size)]
    terrain = Terrain(fuel_arrs, cfg.terrain.elevation_function, cfg.area.terrain_size,
                      cfg.area.screen_size)

    wind_map = WindController()
    wind_map.init_wind_speed_generator(cfg.wind.speed.seed, cfg.wind.speed.scale,
                                       cfg.wind.speed.octaves, cfg.wind.speed.persistence,
                                       cfg.wind.speed.lacunarity, cfg.wind.speed.min,
                                       cfg.wind.speed.max, cfg.area.screen_size)
    wind_map.init_wind_direction_generator(
        cfg.wind.direction.seed, cfg.wind.direction.scale, cfg.wind.direction.octaves,
        cfg.wind.direction.persistence, cfg.wind.direction.lacunarity,
        cfg.wind.direction.min, cfg.wind.direction.max, cfg.area.screen_size)

    environment = Environment(cfg.environment.moisture, wind_map.map_wind_speed,
                              wind_map.map_wind_direction)

    points = line(100, 15, 100, 200)
    y = points[0].tolist()
    x = points[1].tolist()
    points = list(zip(x, y))

    fireline_manager = FireLineManager(size=cfg.display.control_line_size,
                                       pixel_scale=cfg.area.pixel_scale,
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
