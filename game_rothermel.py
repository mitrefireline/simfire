from pathlib import Path

import numpy as np
from skimage.draw import line

from src.enums import GameStatus
from src.game.game import Game
from src.game.managers.fire import RothermelFireManager
from src.game.managers.mitigation import FireLineManager
from src.game.sprites import Terrain
from src.utils.config import Config
from src.utils.layers import DataLayer, FuelLayer, TopographyLayer
from src.utils.units import mph_to_ftpm
from src.world.parameters import Environment, FuelParticle


def main():
    '''
    Initialize the layers.
    Create the Game, Terrain, and Environment
    Create the Managers
    '''
    cfg_path = Path('./config.yml')
    cfg = Config(cfg_path)

    fuel_particle = FuelParticle()

    center = (33.5, 116.8)
    height, width = 5000, 5000
    resolution = 30
    data_layer = DataLayer(center, height, width, resolution)
    topo_layer = TopographyLayer(data_layer)
    fuel_layer = FuelLayer(data_layer)

    # Compute how many meters each pixel represents (should be close to resolution)
    pixel_scale = height / topo_layer.data.shape[0]
    # Convert to feet for use with rothermel
    pixel_scale = 3.28084 * pixel_scale

    game = Game(topo_layer.data.shape[:2])

    terrain = Terrain(fuel_layer, topo_layer, game.screen_size)

    # Use simple/constant wind speed for now
    wind_speed = mph_to_ftpm(cfg.wind.simple.speed)
    wind_speed_arr = np.full(game.screen_size, wind_speed)
    environment = Environment(cfg.environment.moisture, wind_speed_arr,
                              cfg.wind.simple.direction)

    # Need to create two lines to "double up" since the fire can spread
    # to 8-connected squares
    r0, c0 = (0, game.screen_size[1] // 4)
    r1, c1 = (game.screen_size[1] // 4, 0)
    points1 = line(r0, c0, r1, c1)
    r0, c0 = (game.screen_size[1] // 4 - 1, 0)
    r1, c1 = (0, game.screen_size[1] // 4 - 1)
    points2 = line(r0, c0, r1, c1)
    points = tuple(np.concatenate((p1, p2)) for p1, p2 in zip(points1, points2))

    y = points[0].tolist()
    x = points[1].tolist()
    points = list(zip(x, y))

    fireline_manager = FireLineManager(size=cfg.display.control_line_size,
                                       pixel_scale=pixel_scale,
                                       terrain=terrain)

    fire_map = game.fire_map
    fire_map = fireline_manager.update(fire_map, points)
    game.fire_map = fire_map

    fire_manager = RothermelFireManager(cfg.fire.fire_initial_position,
                                        cfg.display.fire_size,
                                        cfg.fire.max_fire_duration,
                                        pixel_scale,
                                        cfg.simulation.update_rate,
                                        fuel_particle,
                                        terrain,
                                        environment,
                                        max_time=cfg.simulation.runtime)

    game_status = GameStatus.RUNNING
    fire_status = GameStatus.RUNNING
    while game_status == GameStatus.RUNNING and fire_status == GameStatus.RUNNING:
        fire_sprites = fire_manager.sprites
        fireline_sprites = fireline_manager.sprites
        game_status = game.update(terrain, fire_sprites, fireline_sprites, cfg.wind.speed,
                                  cfg.wind.direction)
        fire_map = game.fire_map
        fire_map = fireline_manager.update(fire_map)
        fire_map, fire_status = fire_manager.update(fire_map)
        game.fire_map = fire_map


if __name__ == '__main__':
    main()
